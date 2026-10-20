"""
Bewegungsvorhersage mit HUD-Overlay: der erste Baustein des Vorhersage-Auges.

Das Programm verfolgt ein Objekt im Kamerabild und berechnet, wo es in
0,1 / 0,2 / ... / 1,0 Sekunden sein wird. Die 10 Vorhersagepunkte werden
als HUD eingeblendet. Gleichzeitig wird gemessen, wie weit jede Vorhersage
danebenlag, sobald die Zukunft "eingetroffen" ist.

Drei Vorhersagemodelle werden live gegeneinander getestet:
    Stillstand      Objekt bleibt, wo es ist (dümmste Annahme, Basislinie)
    Geradeaus       geglättete Geschwindigkeit aus dem Kalman-Filter, gerade weiter
    Beschleunigt    zusätzlich die geschätzte Beschleunigung (klingt ab)
Das HUD zeigt immer die Vorhersage des Modells, das bisher am besten trifft.

Starten:
    python predict_motion.py --synthetic        # ohne Kamera, simuliertes Objekt
    python predict_motion.py                    # Webcam, verfolgt was sich bewegt
    python predict_motion.py --color green      # Webcam, verfolgt etwas Grünes
    python predict_motion.py --video clip.mp4   # Videodatei

Tasten: q = beenden, r = Statistik zurücksetzen
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

STEP = 0.1        # Vorhersage-Schritt in Sekunden
N_STEPS = 10      # 10 x 0,1 s = 1 s in die Zukunft
HORIZONS = [STEP * (i + 1) for i in range(N_STEPS)]

HSV_PRESETS = {   # (untere Grenze, obere Grenze) im HSV-Farbraum
    "green": ((40, 70, 70), (85, 255, 255)),
    "blue": ((95, 120, 70), (130, 255, 255)),
    "orange": ((5, 130, 120), (22, 255, 255)),
    "red": ((170, 120, 90), (180, 255, 255)),
}

HUD = (120, 255, 160)      # Grün (BGR)
HUD_DIM = (60, 140, 90)
WARN = (60, 180, 255)


# --------------------------------------------------------------------------
# Vorhersage
# --------------------------------------------------------------------------

class KalmanPredictor:
    """Kalman-Filter mit Zustand [x, y, vx, vy, ax, ay].

    update() glättet die verrauschte Messung, predict_path() rechnet den
    Zustand in 0,1-s-Schritten nach vorne. Die Beschleunigung klingt dabei
    mit der Zeitkonstante tau ab, weil Menschen und Objekte selten eine
    ganze Sekunde lang gleich stark beschleunigen.
    """

    def __init__(self, meas_noise=2.0, jerk=8000.0, tau=0.15):
        self.x = None
        self.P = np.eye(6) * 500.0
        self.R = np.eye(2) * meas_noise ** 2
        self.jerk = jerk
        self.tau = tau
        self.H = np.zeros((2, 6))
        self.H[0, 0] = self.H[1, 1] = 1.0

    @staticmethod
    def _F(dt):
        F = np.eye(6)
        for i in range(2):
            F[i, i + 2] = dt
            F[i, i + 4] = 0.5 * dt * dt
            F[i + 2, i + 4] = dt
        return F

    def _Q(self, dt):
        # vereinfachtes Prozessrauschen: Ruck wirkt auf Position/Geschw./Beschl.
        q1 = np.array([dt ** 5 / 20, dt ** 3 / 3, dt])
        return np.diag(np.repeat(q1, 2)) * self.jerk

    def update(self, z, dt):
        z = np.asarray(z, dtype=float)
        if self.x is None:
            self.x = np.array([z[0], z[1], 0, 0, 0, 0], dtype=float)
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def ready(self):
        return self.x is not None

    def predict_path(self):
        """10 Punkte: Position nach 0,1 s, 0,2 s, ..., 1,0 s."""
        s = self.x.copy()
        F = self._F(STEP)
        decay = np.exp(-STEP / self.tau)
        path = []
        for _ in range(N_STEPS):
            s = F @ s
            s[4:] *= decay
            path.append(s[:2].copy())
        return np.array(path)

    def baseline_paths(self):
        pos, vel = self.x[:2], self.x[2:4]
        still = np.repeat(pos[None], N_STEPS, axis=0)
        straight = np.array([pos + vel * h for h in HORIZONS])
        return still, straight


# --------------------------------------------------------------------------
# Fehlermessung: Vorhersage vs. was wirklich passiert ist
# --------------------------------------------------------------------------

class Evaluator:
    MODELS = ("Stillstand", "Geradeaus", "Beschleunigt")

    def __init__(self):
        self.reset()

    def reset(self):
        self.pending = deque()                   # (t_jetzt, {modell: pfad})
        self.track = deque(maxlen=600)           # (t, x, y) echte Messungen
        self.errors = {m: [[] for _ in HORIZONS] for m in self.MODELS}

    def add_measurement(self, t, pos):
        self.track.append((t, pos[0], pos[1]))

    def add_prediction(self, t, paths):
        self.pending.append((t, paths))

    def _actual_at(self, t):
        """Echte Position zum Zeitpunkt t (linear interpoliert)."""
        tr = self.track
        for i in range(len(tr) - 1, 0, -1):
            t0, x0, y0 = tr[i - 1]
            t1, x1, y1 = tr[i]
            if t0 <= t <= t1:
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return np.array([x0 + a * (x1 - x0), y0 + a * (y1 - y0)])
        return None

    def evaluate(self, t_now):
        # Vorhersagen erst auswerten, wenn ihr ganzer Horizont vorbei ist
        while self.pending and self.pending[0][0] + HORIZONS[-1] <= t_now:
            t0, paths = self.pending.popleft()
            for k, h in enumerate(HORIZONS):
                actual = self._actual_at(t0 + h)
                if actual is None:
                    continue
                for m in self.MODELS:
                    self.errors[m][k].append(np.linalg.norm(paths[m][k] - actual))

    def mean_error(self, model, k=N_STEPS - 1):
        e = self.errors[model][k]
        return float(np.mean(e)) if e else float("nan")

    def best_model(self, min_samples=30):
        """Modell mit dem kleinsten 1-s-Fehler (vorher: Geradeaus)."""
        if self.count() < min_samples:
            return "Geradeaus"
        return min(self.MODELS, key=self.mean_error)

    def count(self):
        return len(self.errors["Beschleunigt"][-1])


# --------------------------------------------------------------------------
# Objekt finden
# --------------------------------------------------------------------------

class Detector:
    def __init__(self, color=None):
        self.color = color
        self.bg = None if color else cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def find(self, frame):
        if self.color:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lo, hi = HSV_PRESETS[self.color]
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        else:
            mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 300:
            return None
        m = cv2.moments(c)
        return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


# --------------------------------------------------------------------------
# Simuliertes Objekt (zum Testen ohne Kamera)
# --------------------------------------------------------------------------

class SyntheticScene:
    """Ein Ball, der sich wie ein Mensch bewegt: meistens flüssig, aber
    mit Richtungswechseln, die man nicht exakt vorhersehen kann."""

    def __init__(self, w=960, h=540, fps=30, seed=7):
        self.w, self.h, self.fps = w, h, fps
        self.rng = np.random.default_rng(seed)
        self.pos = np.array([w * 0.3, h * 0.5])
        self.vel = np.array([180.0, -60.0])
        self.i = 0

    def read(self):
        dt = 1.0 / self.fps
        # Geschwindigkeit driftet zufällig (Ornstein-Uhlenbeck-Prozess)
        target = np.array([np.cos(self.i / 70.0), np.sin(self.i / 45.0)]) * 220
        self.vel += (target - self.vel) * 0.8 * dt
        self.vel += self.rng.normal(0, 260, 2) * dt
        self.pos += self.vel * dt
        for d, lim in ((0, self.w), (1, self.h)):          # an Wänden abprallen
            if not 40 < self.pos[d] < lim - 40:
                self.vel[d] *= -1
                self.pos[d] = np.clip(self.pos[d], 40, lim - 40)
        self.i += 1

        frame = np.full((self.h, self.w, 3), (28, 22, 18), np.uint8)
        for x in range(0, self.w, 60):
            cv2.line(frame, (x, 0), (x, self.h), (40, 34, 30), 1)
        for y in range(0, self.h, 60):
            cv2.line(frame, (0, y), (self.w, y), (40, 34, 30), 1)
        cv2.circle(frame, tuple(self.pos.astype(int)), 22, (60, 200, 80), -1)
        noise = self.rng.normal(0, 6, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return True, frame


# --------------------------------------------------------------------------
# HUD zeichnen
# --------------------------------------------------------------------------

def draw_hud(frame, pos, path, ev, fps_shown):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if pos is not None:
        p = tuple(pos.astype(int))
        cv2.circle(overlay, p, 28, HUD, 2)
        cv2.line(overlay, (p[0] - 38, p[1]), (p[0] - 30, p[1]), HUD, 2)
        cv2.line(overlay, (p[0] + 30, p[1]), (p[0] + 38, p[1]), HUD, 2)

    if path is not None:
        prev = tuple(pos.astype(int)) if pos is not None else None
        for k, pt in enumerate(path):
            q = tuple(np.clip(pt, 0, [w - 1, h - 1]).astype(int))
            fade = 1.0 - k / N_STEPS * 0.7
            color = tuple(int(c * fade) for c in HUD)
            if prev:
                cv2.line(overlay, prev, q, color, 2, cv2.LINE_AA)
            cv2.circle(overlay, q, 4 if k < N_STEPS - 1 else 7, color, -1)
            prev = q
        end = tuple(np.clip(path[-1], 0, [w - 1, h - 1]).astype(int))
        cv2.drawMarker(overlay, end, WARN, cv2.MARKER_CROSS, 26, 2)
        cv2.putText(overlay, "+1.0 s", (end[0] + 12, end[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WARN, 1, cv2.LINE_AA)

    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # Infokasten oben links
    cv2.rectangle(frame, (12, 12), (330, 128), (0, 0, 0), -1)
    cv2.rectangle(frame, (12, 12), (330, 128), HUD_DIM, 1)
    best = ev.best_model()
    lines = [f"PREDICT-EYE v0.1   {fps_shown:4.1f} fps",
             f"Fehler nach 1 s ({ev.count()} Messungen):"]
    for m in Evaluator.MODELS:
        mark = "<" if m == best else " "
        lines.append(f" {mark}{m:<12}{ev.mean_error(m):7.1f} px")
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (22, 36 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, HUD, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Auswertung als Grafik
# --------------------------------------------------------------------------

def save_error_plot(ev, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="white")
    styles = {"Stillstand": ("#9aa5b1", "--"), "Geradeaus": ("#2f855a", "-"),
              "Beschleunigt": ("#e09f3e", "-.")}
    for m in Evaluator.MODELS:
        ys = [np.mean(e) if e else np.nan for e in ev.errors[m]]
        color, ls = styles[m]
        ax.plot(HORIZONS, ys, ls, marker="o", color=color, lw=2, label=m)
    ax.set_xlabel("Vorhersage-Horizont (s)")
    ax.set_ylabel("mittlerer Fehler (Pixel)")
    ax.set_title("Wie weit liegt die Vorhersage daneben?", loc="left")
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Bewegungsvorhersage mit HUD")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true",
                     help="ohne Kamera: simuliertes Objekt")
    src.add_argument("--video", help="Videodatei statt Webcam")
    src.add_argument("--camera", type=int, default=0, help="Kamera-Index")
    p.add_argument("--color", choices=HSV_PRESETS.keys(),
                   help="Objekt nach Farbe verfolgen statt nach Bewegung")
    p.add_argument("--headless", action="store_true",
                   help="ohne Fenster (z. B. auf einem Server)")
    p.add_argument("--frames", type=int, default=0,
                   help="nach N Bildern beenden (0 = unbegrenzt)")
    p.add_argument("--out", help="Ergebnisvideo speichern, z. B. demo.mp4")
    p.add_argument("--plot", default="fehler_vs_horizont.png",
                   help="Fehlergrafik beim Beenden speichern")
    return p.parse_args()


def main():
    args = parse_args()

    if args.synthetic:
        cap = SyntheticScene()
        detector = Detector(color="green")
        fps_nominal = cap.fps
    else:
        cap = cv2.VideoCapture(args.video if args.video else args.camera)
        if not cap.isOpened():
            raise SystemExit("Kamera/Video konnte nicht geöffnet werden.")
        detector = Detector(color=args.color)
        fps_nominal = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Bei Datei/Simulation die Zeit aus der Bildnummer berechnen,
    # bei der Webcam die echte Uhrzeit verwenden.
    use_clock = not (args.synthetic or args.video)

    kf = KalmanPredictor()
    ev = Evaluator()
    writer = None
    n, t_prev, t_start = 0, None, time.monotonic()
    fps_shown = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.monotonic() - t_start if use_clock else n / fps_nominal
        dt = (t - t_prev) if t_prev is not None else 1.0 / fps_nominal
        t_prev = t
        fps_shown = 0.9 * fps_shown + 0.1 * (1.0 / max(dt, 1e-3))

        pos = detector.find(frame)
        path = None
        if pos is not None:
            kf.update(pos, dt)
            ev.add_measurement(t, pos)
            if kf.ready:
                path = kf.predict_path()
                still, straight = kf.baseline_paths()
                paths = {"Stillstand": still, "Geradeaus": straight,
                         "Beschleunigt": path}
                ev.add_prediction(t, paths)
                path = paths[ev.best_model()]
        ev.evaluate(t)

        draw_hud(frame, pos, path, ev, fps_shown)

        if args.out:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_nominal, (w, h))
            writer.write(frame)

        if not args.headless:
            cv2.imshow("Predict-Eye", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                ev.reset()

        n += 1
        if args.frames and n >= args.frames:
            break

    if writer:
        writer.release()
    if not args.synthetic:
        cap.release()
    cv2.destroyAllWindows()

    print(f"\n{n} Bilder verarbeitet, {ev.count()} Vorhersagen ausgewertet.")
    print("Mittlerer Fehler in Pixel:")
    print("  Horizont    " + "".join(f"{h:>7.1f}s" for h in HORIZONS))
    for m in Evaluator.MODELS:
        row = "".join(f"{np.mean(e) if e else float('nan'):8.1f}"
                      for e in ev.errors[m])
        print(f"  {m:<12}{row}")
    best, s_err = ev.best_model(), ev.mean_error("Stillstand")
    if s_err == s_err and s_err > 0:
        b_err = ev.mean_error(best)
        print(f"\nBestes Modell: {best}. Nach 1 s ist es "
              f"{100 * (1 - b_err / s_err):.0f} % genauer als 'Objekt bleibt stehen'.")
    if args.plot and ev.count():
        save_error_plot(ev, args.plot)
        print(f"Grafik gespeichert: {args.plot}")


if __name__ == "__main__":
    main()

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

import cv2
import numpy as np

from predictor import (HORIZONS, N_STEPS, Evaluator, KalmanPredictor,
                       SimulatedMotion)


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
    """Zeichnet die simulierte Bewegung aus predictor.SimulatedMotion als
    Kamerabild, damit die echte Erkennung (Detector) getestet wird."""

    def __init__(self, w=960, h=540, fps=30, seed=7):
        self.w, self.h, self.fps = w, h, fps
        self.motion = SimulatedMotion(w, h, fps, seed)
        self.rng = np.random.default_rng(seed + 1000)   # nur für Bildrauschen

    def read(self):
        pos = self.motion.step()
        frame = np.full((self.h, self.w, 3), (28, 22, 18), np.uint8)
        for x in range(0, self.w, 60):
            cv2.line(frame, (x, 0), (x, self.h), (40, 34, 30), 1)
        for y in range(0, self.h, 60):
            cv2.line(frame, (0, y), (self.w, y), (40, 34, 30), 1)
        cv2.circle(frame, tuple(pos.astype(int)), 22, (60, 200, 80), -1)
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
                path = paths[ev.best_model(Evaluator.MOVING)]
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

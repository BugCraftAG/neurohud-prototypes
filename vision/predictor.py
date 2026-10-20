"""
Kern der Bewegungsvorhersage: reines Python + numpy, ohne OpenCV.

Dieses Modul wird von zwei Programmen benutzt:
  - vision/predict_motion.py   (Desktop, Webcam über OpenCV)
  - web/                        (Browser-Demo, läuft über Pyodide im Browser)

So ist sichergestellt, dass beide genau dieselbe Vorhersage rechnen.
"""

from collections import deque

import numpy as np

STEP = 0.1        # Vorhersage-Schritt in Sekunden
N_STEPS = 10      # 10 x 0,1 s = 1 s in die Zukunft
HORIZONS = [STEP * (i + 1) for i in range(N_STEPS)]


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
# Simulierte Bewegung (zum Testen ohne Kamera)
# --------------------------------------------------------------------------

class SimulatedMotion:
    """Ein Objekt, das sich wie ein Mensch bewegt: meistens flüssig, aber
    mit Richtungswechseln, die man nicht exakt vorhersehen kann."""

    def __init__(self, w=960, h=540, fps=30, seed=7):
        self.w, self.h, self.fps = w, h, fps
        self.rng = np.random.default_rng(seed)
        self.pos = np.array([w * 0.3, h * 0.5])
        self.vel = np.array([180.0, -60.0])
        self.i = 0

    def step(self):
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
        return self.pos.copy()


# --------------------------------------------------------------------------
# Alles zusammen: messen -> glätten -> vorhersagen -> überprüfen
# --------------------------------------------------------------------------

class MotionPredictor:
    """Ein Aufruf pro Kamerabild. Gibt den Pfad des besten Modells zurück."""

    def __init__(self):
        self.kf = KalmanPredictor()
        self.ev = Evaluator()

    def reset(self):
        self.kf = KalmanPredictor()
        self.ev.reset()

    def step(self, t, dt, pos):
        path = None
        if pos is not None:
            self.kf.update(pos, dt)
            self.ev.add_measurement(t, pos)
            still, straight = self.kf.baseline_paths()
            paths = {"Stillstand": still, "Geradeaus": straight,
                     "Beschleunigt": self.kf.predict_path()}
            self.ev.add_prediction(t, paths)
            path = paths[self.ev.best_model()]
        self.ev.evaluate(t)
        return path

    def stats(self):
        """Fehler je Modell und Horizont, für Anzeige und Grafik."""
        return {
            "count": self.ev.count(),
            "best": self.ev.best_model(),
            "errors": {m: [float(np.mean(e)) if e else None
                           for e in self.ev.errors[m]]
                       for m in Evaluator.MODELS},
        }

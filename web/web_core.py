"""
Python-Teil der Browser-Demo. Läuft über Pyodide direkt im Browser.

JavaScript macht nur zwei Dinge: Kamerabild holen und das HUD zeichnen.
Alles dazwischen passiert hier in Python:
    Bewegung im Bild finden -> Kalman-Filter -> 1 s vorhersagen -> Fehler messen

Die Vorhersage selbst kommt aus predictor.py, derselben Datei, die auch
das Desktop-Programm vision/predict_motion.py benutzt.
"""

import numpy as np

from predictor import MotionPredictor, SimulatedMotion


class MotionDetector:
    """Findet Bewegung durch Vergleich mit einem gleitenden Hintergrundbild.

    Das ist eine vereinfachte Version der Hintergrund-Subtraktion, die das
    Desktop-Programm mit OpenCV (MOG2) macht. Hier reicht numpy.
    """

    def __init__(self, threshold=28, learn_rate=0.08, min_fraction=0.004):
        self.bg = None
        self.threshold = threshold
        self.learn_rate = learn_rate
        self.min_fraction = min_fraction
        self.last_mask = None

    def find(self, rgba, w, h):
        img = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        gray = img[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114],
                                                          dtype=np.float32)
        if self.bg is None:
            self.bg = gray
            return None
        diff = np.abs(gray - self.bg)
        self.bg += self.learn_rate * (gray - self.bg)

        mask = diff > self.threshold
        self.last_mask = mask
        if mask.mean() < self.min_fraction:
            return None
        # Schwerpunkt der bewegten Pixel, gewichtet mit der Stärke der Änderung
        ys, xs = np.nonzero(mask)
        weights = diff[ys, xs]
        return np.array([np.average(xs, weights=weights),
                         np.average(ys, weights=weights)])


class Demo:
    def __init__(self):
        self.predictor = MotionPredictor()
        self.detector = MotionDetector()
        self.sim = None
        self.noise = np.random.default_rng(1)

    def reset(self, mode):
        self.predictor.reset()
        self.detector = MotionDetector()
        self.sim = SimulatedMotion(w=960, h=540, fps=30, seed=7) if mode == "sim" else None

    def camera_frame(self, t, dt, rgba, w, h):
        """Ein Kamerabild (verkleinert, RGBA) -> Position und Vorhersage."""
        pos = self.detector.find(rgba.to_py(), w, h)
        path = self.predictor.step(t, dt, pos)
        return self._result(pos, path)

    def sim_frame(self, t, dt):
        """Ein Schritt der Simulation. Die Messung bekommt etwas Rauschen,
        wie bei einer echten Erkennung."""
        truth = self.sim.step()
        measured = truth + self.noise.normal(0, 2.0, 2)
        path = self.predictor.step(t, dt, measured)
        return self._result(truth, path)

    def stats(self):
        return self.predictor.stats()

    @staticmethod
    def _result(pos, path):
        return {
            "pos": None if pos is None else [float(pos[0]), float(pos[1])],
            "path": None if path is None else [[float(x), float(y)] for x, y in path],
        }


demo = Demo()

"""
EEG-Live-Viewer für OpenBCI (Cyton, Cyton+Daisy, Ganglion) über BrainFlow.

Zeigt die EEG-Kanäle gefiltert in Echtzeit an und berechnet die Leistung
in den klassischen Frequenzbändern (Delta, Theta, Alpha, Beta, Gamma).

Ohne Hardware testen (künstliche Signale):
    python eeg_viewer.py

Mit OpenBCI Cyton am USB-Dongle:
    python eeg_viewer.py --board cyton --port COM3          # Windows
    python eeg_viewer.py --board cyton --port /dev/ttyUSB0  # Linux

Aufnahme als CSV speichern:
    python eeg_viewer.py --record aufnahme.csv

Kleiner Test: Augen schließen -> der Alpha-Balken (8-13 Hz) steigt deutlich.
"""

import argparse
import sys
import time

import numpy as np
import matplotlib

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    FilterTypes,
    NoiseTypes,
    WindowOperations,
)

BOARDS = {
    "synthetic": BoardIds.SYNTHETIC_BOARD,
    "cyton": BoardIds.CYTON_BOARD,
    "cyton-daisy": BoardIds.CYTON_DAISY_BOARD,
    "ganglion": BoardIds.GANGLION_BOARD,
}

BANDS = [
    ("Delta", 1.0, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta", 13.0, 30.0),
    ("Gamma", 30.0, 45.0),
]

WINDOW_SECONDS = 5      # so viel Signal ist im Plot sichtbar
MAX_PLOT_CHANNELS = 8   # bei Daisy (16 Kanäle) nur die ersten 8 zeigen


def parse_args():
    p = argparse.ArgumentParser(description="EEG-Live-Viewer für OpenBCI")
    p.add_argument("--board", choices=BOARDS.keys(), default="synthetic",
                   help="Welches Board (Standard: synthetic = ohne Hardware)")
    p.add_argument("--port", default="",
                   help="Serieller Port des Dongles, z. B. COM3 oder /dev/ttyUSB0")
    p.add_argument("--mac", default="", help="MAC-Adresse (nur Ganglion)")
    p.add_argument("--record", metavar="DATEI.csv",
                   help="Rohdaten beim Beenden als CSV speichern")
    p.add_argument("--headless", type=float, metavar="SEKUNDEN",
                   help="Ohne Fenster laufen, nach N Sekunden ein PNG speichern")
    p.add_argument("--out", default="eeg_snapshot.png",
                   help="Dateiname für das PNG im --headless-Modus")
    return p.parse_args()


class EEGStream:
    """Kapselt das BrainFlow-Board: verbinden, Daten holen, filtern."""

    def __init__(self, board_name, port="", mac=""):
        params = BrainFlowInputParams()
        params.serial_port = port
        params.mac_address = mac
        self.board_id = BOARDS[board_name]
        self.board = BoardShim(self.board_id, params)
        self.fs = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.n_samples = WINDOW_SECONDS * self.fs

    def start(self):
        BoardShim.disable_board_logger()
        self.board.prepare_session()
        self.board.start_stream(45000)

    def stop(self, record_path=None):
        if record_path:
            data = self.board.get_board_data()  # gesamter Puffer
            DataFilter.write_file(data, record_path, "w")
            print(f"Aufnahme gespeichert: {record_path} ({data.shape[1]} Samples)")
        self.board.stop_stream()
        self.board.release_session()

    def window(self):
        """Die letzten WINDOW_SECONDS Sekunden, gefiltert, pro Kanal."""
        data = self.board.get_current_board_data(self.n_samples)
        out = []
        for ch in self.eeg_channels[:MAX_PLOT_CHANNELS]:
            x = np.ascontiguousarray(data[ch], dtype=np.float64)
            if len(x) < self.fs:          # zu wenig Daten am Anfang
                out.append(x)
                continue
            DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(x, self.fs, 1.0, 45.0, 4,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
            # 50 Hz Netzbrummen (Europa) entfernen
            DataFilter.remove_environmental_noise(x, self.fs, NoiseTypes.FIFTY.value)
            out.append(x)
        return out

    def band_powers(self, channels):
        """Relative Bandleistung, gemittelt über alle Kanäle (Summe = 1)."""
        nfft = DataFilter.get_nearest_power_of_two(self.fs)
        totals = np.zeros(len(BANDS))
        used = 0
        for x in channels:
            if len(x) < nfft * 2:
                continue
            psd = DataFilter.get_psd_welch(x, nfft, nfft // 2, self.fs,
                                           WindowOperations.HANNING.value)
            powers = np.array([DataFilter.get_band_power(psd, lo, hi)
                               for _, lo, hi in BANDS])
            if powers.sum() > 0:
                totals += powers / powers.sum()
                used += 1
        return totals / used if used else totals


def build_figure(stream):
    import matplotlib.pyplot as plt

    n = min(len(stream.eeg_channels), MAX_PLOT_CHANNELS)
    fig = plt.figure(figsize=(12, 6.5), facecolor="#0b0f14")
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 0.08, 1], wspace=0.05)
    ax_sig = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 2])

    for ax in (ax_sig, ax_bar):
        ax.set_facecolor("#0b0f14")
        ax.tick_params(colors="#9fb3c8")
        for s in ax.spines.values():
            s.set_color("#25313d")

    t = np.linspace(-WINDOW_SECONDS, 0, stream.n_samples)
    lines = []
    for i in range(n):
        (ln,) = ax_sig.plot(t, np.zeros_like(t) + i, lw=0.8, color="#4fd1c5")
        lines.append(ln)
    ax_sig.set_yticks(range(n))
    ax_sig.set_yticklabels([f"Ch {i + 1}" for i in range(n)])
    ax_sig.set_ylim(-0.8, n - 0.2)
    ax_sig.set_xlim(-WINDOW_SECONDS, 0)
    ax_sig.set_xlabel("Zeit (s)", color="#9fb3c8")
    ax_sig.set_title("EEG, gefiltert 1-45 Hz, 50 Hz Notch", color="#e2e8f0",
                     loc="left", fontsize=11)

    names = [b[0] for b in BANDS]
    bars = ax_bar.bar(names, np.zeros(len(BANDS)),
                      color=["#64748b", "#818cf8", "#4fd1c5", "#f6ad55", "#f56565"])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_title("Relative Bandleistung", color="#e2e8f0", loc="left", fontsize=11)
    ax_bar.tick_params(axis="x", labelsize=8)

    return fig, lines, bars, t


def update_plot(stream, lines, bars, t):
    channels = stream.window()
    for i, (ln, x) in enumerate(zip(lines, channels)):
        if len(x) == 0:
            continue
        # jeden Kanal auf seine eigene Zeile skalieren
        scale = np.percentile(np.abs(x), 98) or 1.0
        y = x / (2.5 * scale) + i
        ln.set_data(t[-len(y):], y)
    powers = stream.band_powers(channels)
    for bar, v in zip(bars, powers):
        bar.set_height(v)
    return powers


def main():
    args = parse_args()
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    stream = EEGStream(args.board, args.port, args.mac)
    try:
        stream.start()
    except Exception as e:  # meist falscher Port oder Dongle nicht gesteckt
        sys.exit(f"Verbindung fehlgeschlagen: {e}\n"
                 f"Tipp: Port prüfen (--port) und Board einschalten.")

    print(f"Board: {args.board} | {stream.fs} Hz | "
          f"{len(stream.eeg_channels)} EEG-Kanäle")

    fig, lines, bars, t = build_figure(stream)
    try:
        if args.headless:
            time.sleep(args.headless)
            powers = update_plot(stream, lines, bars, t)
            fig.savefig(args.out, dpi=110, facecolor=fig.get_facecolor())
            print("Bandleistung: " + ", ".join(
                f"{n} {v:.2f}" for (n, _, _), v in zip(BANDS, powers)))
            print(f"Snapshot gespeichert: {args.out}")
        else:
            anim = FuncAnimation(fig, lambda _: update_plot(stream, lines, bars, t),
                                 interval=100, cache_frame_data=False)
            plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop(args.record)


if __name__ == "__main__":
    main()

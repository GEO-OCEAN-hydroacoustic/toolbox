"""
Association Classifier — Application PyQt5 + pyqtgraph
-------------------------------------------------------
Visualise les formes d'onde et spectrogrammes de chaque station
pour un événement donné et permet de le classer manuellement :
  ✅ Valide   ❓ Incertain   ❌ Invalide

Usage :
    python association_classifier.py

Dépendances :
    pip install pyqtgraph PyQt5 numpy scipy matplotlib
"""

import sys
import pickle
import json
import datetime

import glob2
import numpy as np
from pathlib import Path
from datetime import timedelta

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from matplotlib import pyplot as plt

# ── Adapte ces imports à ton projet ──────────────────────────────────────────
from src.utils.data_reading.sound_data.sound_file_manager import DatFilesManager, WFilesManager
from utils.data_reading.sound_data.station import StationsCatalog
from utils.physics.signal.make_spectrogram import make_spectrogram
from utils.physics.sound_model.ellipsoidal_sound_model import GridEllipsoidalSoundModel
# ─────────────────────────────────────────────────────────────────────────────

# ── Stubs pour permettre de lancer sans les modules métier ───────────────────
# Supprime ces stubs lorsque tu importes les vrais modules.

# def make_spectrogram(data, fs, t_res=0.5, f_res=0.5, return_bins=True,
#                      normalize=False, vmin=40, vmax=120):
#     """Stub : spectrogramme via scipy."""
#     from scipy.signal import spectrogram as sp_spectrogram
#     nperseg = max(64, int(fs / f_res))
#     noverlap = nperseg - max(1, int(fs * t_res))
#     f, t, Sxx = sp_spectrogram(data.astype(float), fs=fs,
#                                 nperseg=nperseg, noverlap=noverlap)
#     Sxx_dB = 10 * np.log10(Sxx + 1e-12)
#     if return_bins:
#         return f, t, Sxx_dB
#     return Sxx_dB


# class DummyManager:
#     """Stub : génère un signal synthétique pour les tests."""
#     sampling_f = 100
#
#     def get_segment(self, start, end):
#         n = int((end - start).total_seconds() * self.sampling_f)
#         t = np.linspace(0, (end - start).total_seconds(), n)
#         sig = (np.sin(2 * np.pi * 5 * t) + 0.3 * np.random.randn(n)).astype(np.float32)
#         return sig

# ─────────────────────────────────────────────────────────────────────────────


def butter_bandpass(lowcut, highcut, fs, order=4):
    from scipy.signal import butter
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs):
    from scipy.signal import filtfilt
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)


def refine_single_detection(DATA_ROOT, DATASET,station_obj, det_time, lowcut=None, highcut=None,time_min=5):
    """
    Refine one pick time by locating the max energy within ±SMOOTH_WINDOW_SEC.
    Also computes a simple SNR ratio.
    Returns a dict or None on failure.
    """
    # load waveform ±3 minutes
    start = det_time - timedelta(minutes=time_min)
    end   = det_time + timedelta(minutes=time_min)
    if station_obj.dataset=="OHASISBIO-2018":
        raw = 'raw'
        mgr = DatFilesManager(f"{DATA_ROOT}/{DATASET}/{station_obj.name}", kwargs=raw)
        data = mgr.get_segment(start, end)
        sampling_f = mgr.sampling_f
    else :
        mgr = WFilesManager(f"/media/rsafran/CORSAIR/CTBT/CTBTO_2018/{station_obj.name}_2018/")
        data = mgr.get_segment(start, end)
        sampling_f = round(mgr.sampling_f)

    n = len(data)
    t = np.arange(n) / sampling_f - (det_time - start).total_seconds()
    if lowcut is not None and highcut is not None:
        data = bandpass_filter(data,lowcut, highcut, round(sampling_f))

    return data,t, sampling_f


def process(event_id, STATIONS, associations, IDX_TO_DET):
    """Extrait les listes (station_obj, det_time) pour un event."""
    mat = associations[event_id][0]          # shape (N, ≥2)
    stations = [STATIONS[j] for j in mat[:, 0]]
    dets     = [IDX_TO_DET[j][0] for j in mat[:, 1]]
    return stations, dets


# ─────────────────────────────────────── Couleurs labels ─────────────────────
LABEL_COLORS = {
    "Valide":    "#27ae60",   # vert
    "Incertain": "#e67e22",   # orange
    "Invalide":  "#c0392b",   # rouge
    None:        "#555555",   # gris
}

LABEL_ICONS = {
    "Valide":    "✅",
    "Incertain": "❓",
    "Invalide":  "❌",
    None:        "·",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Widget : panneau de contrôle filtre + navigation
# ══════════════════════════════════════════════════════════════════════════════
class ControlPanel(QtWidgets.QWidget):
    sigFilterChanged  = QtCore.pyqtSignal(object, object)   # (lowcut, highcut)
    sigViewChanged    = QtCore.pyqtSignal(str)              # 'waveform' | 'spectrogram'
    sigWindowChanged  = QtCore.pyqtSignal(int)              # time_min
    sigHeightChanged  = QtCore.pyqtSignal(int)              # hauteur px par panneau

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Vue ──────────────────────────────────────────────────────────────
        layout.addWidget(QtWidgets.QLabel("Vue :"))
        self.cbo_view = QtWidgets.QComboBox()
        self.cbo_view.addItems(["Spectrogramme", "Forme d'onde"])
        self.cbo_view.currentTextChanged.connect(
            lambda t: self.sigViewChanged.emit(
                "spectrogram" if t == "Spectrogramme" else "waveform"))
        layout.addWidget(self.cbo_view)

        layout.addWidget(QtWidgets.QLabel("  ±"))
        self.spin_window = QtWidgets.QSpinBox()
        self.spin_window.setRange(1, 120)
        self.spin_window.setValue(30)
        self.spin_window.setSuffix(" min")
        self.spin_window.valueChanged.connect(self.sigWindowChanged.emit)
        layout.addWidget(self.spin_window)

        layout.addWidget(QtWidgets.QLabel("  Hauteur (px)"))
        self.spin_height = QtWidgets.QSpinBox()
        self.spin_height.setRange(80, 600)
        self.spin_height.setValue(200)
        self.spin_height.setSingleStep(20)
        self.spin_height.valueChanged.connect(self.sigHeightChanged.emit)
        layout.addWidget(self.spin_height)

        layout.addSpacing(16)
        layout.addWidget(QtWidgets.QLabel("Filtre passe-bande :"))

        self.chk_filter = QtWidgets.QCheckBox("Activer")
        layout.addWidget(self.chk_filter)

        layout.addWidget(QtWidgets.QLabel("  F. basse (Hz)"))
        self.spin_low = QtWidgets.QDoubleSpinBox()
        self.spin_low.setRange(0.01, 200)
        self.spin_low.setValue(1.0)
        self.spin_low.setSingleStep(0.5)
        layout.addWidget(self.spin_low)

        layout.addWidget(QtWidgets.QLabel("  Haut (Hz)"))
        self.spin_high = QtWidgets.QDoubleSpinBox()
        self.spin_high.setRange(0.1, 500)
        self.spin_high.setValue(120.0)
        self.spin_high.setSingleStep(0.5)
        layout.addWidget(self.spin_high)

        btn_apply = QtWidgets.QPushButton("Appliquer")
        btn_apply.clicked.connect(self._emit_filter)
        layout.addWidget(btn_apply)

        layout.addStretch()

    def _emit_filter(self):
        if self.chk_filter.isChecked():
            lo = self.spin_low.value()
            hi = self.spin_high.value()
            # Garantit lowcut < highcut même si l'utilisateur les a saisis à l'envers
            if lo >= hi:
                lo, hi = hi, lo
                self.spin_low.setValue(lo)
                self.spin_high.setValue(hi)
            self.sigFilterChanged.emit(lo, hi)
        else:
            self.sigFilterChanged.emit(None, None)


# ══════════════════════════════════════════════════════════════════════════════
#  Widget : un panneau station (titre + graphique)
# ══════════════════════════════════════════════════════════════════════════════
class StationPlot(QtWidgets.QFrame):
    def __init__(self, station_name: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.station_name = station_name

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.lbl = QtWidgets.QLabel(f"<b>{station_name}</b>")
        self.lbl.setAlignment(QtCore.Qt.AlignLeft)
        layout.addWidget(self.lbl)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#1e1e2e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.plot_widget)

        self.img_item  = pg.ImageItem()
        self.curve     = self.plot_widget.plot(pen=pg.mkPen("#89dceb", width=1))
        self._mode     = "spectrogram"

        # ligne verticale au temps 0 (détection)
        self.vline = pg.InfiniteLine(pos=0, angle=90,
                                     pen=pg.mkPen("#f38ba8", width=1.5, style=QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.vline)

    # ── API publique ──────────────────────────────────────────────────────────
    def set_mode(self, mode: str):
        self._mode = mode

    def plot_waveform(self, t, data):
        self.plot_widget.clear()
        self.plot_widget.setLabel("bottom", "Temps relatif à la détection (s)")
        self.plot_widget.setLabel("left", "Amplitude")
        self.curve = self.plot_widget.plot(t, data,
                                           pen=pg.mkPen("#89dceb", width=1))
        # vline après la courbe pour rester au premier plan
        self.plot_widget.addItem(self.vline)

    def plot_spectrogram(self, t, data, fs):
        self.plot_widget.clear()

        f_arr, t_arr, Sxx = make_spectrogram(data, fs, t_res=0.25, f_res=0.25,
                                             return_bins=True, normalize=True, vmin = 40, vmax = 120)

        img = pg.ImageItem()

        # orientation : x=temps, y=fréquence
        img.setImage(np.flipud(Sxx).T)

        # make_spectrogram renvoie t_arr débutant à 0 (relatif au début du signal),
        # mais t est centré sur la détection (t=0 = det_time).
        # On décale le rect de t[0] pour que la vline à x=0 tombe sur la détection.
        t_offset = float(t[0])          # = -time_min * 60  (négatif)
        img.setRect(pg.QtCore.QRectF(
            t_arr[0] + t_offset,        # x origin décalé
            f_arr[0],
            t_arr[-1] - t_arr[0],       # largeur totale inchangée
            f_arr[-1] - f_arr[0],
        ))
        cmap = pg.colormap.get("inferno", source="matplotlib")
        img.setColorMap(cmap)
        self.plot_widget.addItem(img)

        # vline APRÈS l'image pour ne pas être masquée par elle
        self.plot_widget.addItem(self.vline)
        self.plot_widget.setLabel("bottom", "Temps relatif à la détection (s)")
        self.plot_widget.setLabel("left", "Fréquence (Hz)")

    def update_data(self, t, data, fs):
        if self._mode == "waveform":
            self.plot_waveform(t, data)
        else:
            self.plot_spectrogram(t, data, fs)

    def set_error(self, msg):
        self.plot_widget.clear()
        # Affiche le nom de la station + l'erreur dans le label au-dessus
        self.lbl.setText(f"<b>{self.station_name}</b> — <span style='color:#f38ba8'>{msg}</span>")


# ══════════════════════════════════════════════════════════════════════════════
#  Thread de chargement des données (évite de bloquer l'UI)
# ══════════════════════════════════════════════════════════════════════════════
class LoadWorker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(int, object, object, float)   # idx, t, data, fs
    error        = QtCore.pyqtSignal(int, str)

    def __init__(self, tasks, DATA_ROOT, DATASET, lowcut, highcut, time_min, parent=None):
        super().__init__(parent)
        self.tasks     = tasks          # list of (idx, station_obj, det_time)
        self.DATA_ROOT = DATA_ROOT
        self.DATASET   = DATASET
        self.lowcut    = lowcut
        self.highcut   = highcut
        self.time_min  = time_min
        self._cancel   = False          # flag d'annulation coopérative

    def cancel(self):
        """Demande l'arrêt propre du thread (pas de terminate brutal)."""
        self._cancel = True

    def run(self):
        for idx, station_obj, det_time in self.tasks:
            if self._cancel:
                return                  # sort sans crasher
            try:
                data, t, fs = refine_single_detection(
                    self.DATA_ROOT, self.DATASET, station_obj, det_time,
                    self.lowcut, self.highcut, self.time_min)
                if self._cancel:
                    return
                self.result_ready.emit(idx, t, data, float(fs))
            except Exception as e:
                if not self._cancel:
                    self.error.emit(idx, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  Fenêtre principale
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, associations, STATIONS, IDX_TO_DET, final_results,
                 DATA_ROOT="", DATASET="",
                 labels_path="classifications.json"):
        super().__init__()
        self.setWindowTitle("Association Classifier")
        self.resize(1600, 900)

        # ── Données ───────────────────────────────────────────────────────────
        self.associations  = associations
        self.STATIONS      = STATIONS
        self.IDX_TO_DET    = IDX_TO_DET
        self.event_list    = list(final_results.keys())
        self.DATA_ROOT     = DATA_ROOT
        self.DATASET       = DATASET
        self.labels_path   = labels_path
        self.labels: dict  = {}          # event_id -> label
        self.current_idx   = 0
        self.view_mode     = "spectrogram"
        self.lowcut        = None
        self.highcut       = None
        self.time_min      = 30
        self._worker       = None
        self._load_id      = 0          # incrémenté à chaque chargement pour détecter les résultats périmés
        self._station_plots: list[StationPlot] = []

        # ── Chargement sauvegarde existante ──────────────────────────────────
        if Path(labels_path).exists():
            with open(labels_path) as f:
                self.labels = json.load(f)

        self._build_ui()
        self._load_event(0)

    # ── Construction de l'UI ──────────────────────────────────────────────────
    def _build_ui(self):
        pg.setConfigOptions(antialias=True)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Barre de contrôle ─────────────────────────────────────────────────
        self.ctrl = ControlPanel()
        self.ctrl.sigFilterChanged.connect(self._on_filter_changed)
        self.ctrl.sigViewChanged.connect(self._on_view_changed)
        self.ctrl.sigWindowChanged.connect(self._on_window_changed)
        self.ctrl.sigHeightChanged.connect(self._on_height_changed)
        root.addWidget(self.ctrl)

        # ── Corps : liste événements | graphiques ─────────────────────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        # -- Liste des événements ---------------------------------------------
        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        lbl = QtWidgets.QLabel("<b>Événements</b>")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        left_lay.addWidget(lbl)

        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Filtrer par ID…")
        self.search_box.textChanged.connect(self._filter_event_list)
        left_lay.addWidget(self.search_box)

        self.event_list_widget = QtWidgets.QListWidget()
        self.event_list_widget.setFixedWidth(200)
        self.event_list_widget.currentRowChanged.connect(self._on_list_selection)
        left_lay.addWidget(self.event_list_widget)

        # statistiques
        self.lbl_stats = QtWidgets.QLabel()
        self.lbl_stats.setAlignment(QtCore.Qt.AlignCenter)
        left_lay.addWidget(self.lbl_stats)

        btn_export = QtWidgets.QPushButton("💾  Exporter CSV")
        btn_export.clicked.connect(self._export_csv)
        left_lay.addWidget(btn_export)

        splitter.addWidget(left)

        # -- Zone graphiques --------------------------------------------------
        right = QtWidgets.QWidget()
        self.right_lay = QtWidgets.QVBoxLayout(right)
        self.right_lay.setContentsMargins(4, 4, 4, 4)
        self.right_lay.setSpacing(4)

        # en-tête événement
        hdr = QtWidgets.QHBoxLayout()
        self.lbl_event = QtWidgets.QLabel()
        self.lbl_event.setStyleSheet("font-size:14px; font-weight:bold;")
        hdr.addWidget(self.lbl_event)
        hdr.addStretch()

        btn_prev = QtWidgets.QPushButton("◀  Préc.")
        btn_prev.clicked.connect(lambda: self._navigate(-1))
        btn_next = QtWidgets.QPushButton("Suiv.  ▶")
        btn_next.clicked.connect(lambda: self._navigate(+1))
        hdr.addWidget(btn_prev)
        hdr.addWidget(btn_next)
        self.right_lay.addLayout(hdr)

        # grille de plots (scroll)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(6)
        self.scroll.setWidget(self.grid_container)
        self.right_lay.addWidget(self.scroll, stretch=1)

        # ── Barre de classification ───────────────────────────────────────────
        cls_bar = QtWidgets.QHBoxLayout()
        cls_bar.addStretch()

        for label, color, shortcut in [
            ("✅  Valide",    "#27ae60", "V"),
            ("❓  Incertain", "#e67e22", "U"),
            ("❌  Invalide",  "#c0392b", "I"),
        ]:
            btn = QtWidgets.QPushButton(f"{label}  [{shortcut}]")
            btn.setFixedHeight(42)
            btn.setStyleSheet(
                f"QPushButton {{ background:{color}; color:white; font-size:14px;"
                f"  border-radius:6px; padding:4px 18px; }}"
                f"QPushButton:hover {{ background:{color}cc; }}"
            )
            raw_label = label.split("  ")[1]          # "Valide" / "Incertain" / "Invalide"
            btn.clicked.connect(lambda _, l=raw_label: self._classify(l))
            btn.setShortcut(shortcut)
            cls_bar.addWidget(btn)

        cls_bar.addStretch()
        self.right_lay.addLayout(cls_bar)
        splitter.addWidget(right)
        splitter.setSizes([200, 1400])

        # ── Remplissage de la liste ───────────────────────────────────────────
        self._populate_event_list()
        self._update_stats()

    # ── Liste des événements ──────────────────────────────────────────────────
    def _populate_event_list(self, filter_text=""):
        self.event_list_widget.blockSignals(True)
        self.event_list_widget.clear()
        for eid in self.event_list:
            if filter_text and filter_text.lower() not in str(eid).lower():
                continue
            label = self.labels.get(str(eid))
            icon  = LABEL_ICONS.get(label, "·")
            color = LABEL_COLORS.get(label, "#555555")
            item = QtWidgets.QListWidgetItem(f"{icon} {eid}")
            item.setForeground(QtGui.QColor(color))
            item.setData(QtCore.Qt.UserRole, eid)
            self.event_list_widget.addItem(item)
        self.event_list_widget.blockSignals(False)

    def _filter_event_list(self, text):
        self._populate_event_list(text)

    def _on_list_selection(self, row):
        if row < 0:
            return
        item = self.event_list_widget.item(row)
        if item is None:
            return
        eid = item.data(QtCore.Qt.UserRole)
        idx = self.event_list.index(eid)
        if idx != self.current_idx:
            self.current_idx = idx
            self._load_event(idx)

    # ── Chargement d'un événement ─────────────────────────────────────────────
    def _load_event(self, idx: int):
        if not (0 <= idx < len(self.event_list)):
            return
        self.current_idx = idx
        event_id = self.event_list[idx]

        # synchronise la liste visuelle
        self.event_list_widget.blockSignals(True)
        for i in range(self.event_list_widget.count()):
            item = self.event_list_widget.item(i)
            if item and item.data(QtCore.Qt.UserRole) == event_id:
                self.event_list_widget.setCurrentRow(i)
                break
        self.event_list_widget.blockSignals(False)

        # en-tête
        label = self.labels.get(str(event_id))
        icon  = LABEL_ICONS.get(label, "—")
        color = LABEL_COLORS.get(label, "#aaaaaa")
        self.lbl_event.setText(
            f"Événement <span style='color:#cba6f7'>#{event_id}</span>"
            f"  —  <span style='color:{color}'>{icon} {label or 'Non classé'}</span>"
            f"  ({idx+1}/{len(self.event_list)})"
        )

        # récupère stations + temps de détection
        try:
            stations, dets = process(event_id, self.STATIONS,
                                     self.associations, self.IDX_TO_DET)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erreur", str(e))
            return

        self._rebuild_grid(stations)

        # ── Annulation propre du worker précédent ─────────────────────────────
        # On utilise cancel() + wait() au lieu de terminate() qui peut crasher
        # lorsque le thread est au milieu d'un I/O ou d'un appel C.
        if self._worker is not None:
            self._worker.cancel()
            if self._worker.isRunning():
                self._worker.quit()
                if not self._worker.wait(3000):   # timeout 3 s
                    # Dernier recours seulement si le thread est vraiment bloqué
                    self._worker.terminate()
                    self._worker.wait()

        # Identifiant unique de ce chargement : les signaux périmés seront ignorés
        self._load_id += 1
        current_load_id = self._load_id

        tasks = [(i, stations[i], dets[i]) for i in range(len(stations))]
        self._worker = LoadWorker(tasks, self.DATA_ROOT, self.DATASET,
                                  self.lowcut, self.highcut, self.time_min)
        # Capture current_load_id dans la closure pour filtrer les résultats périmés
        self._worker.result_ready.connect(
            lambda idx, t, data, fs, lid=current_load_id:
                self._on_data_ready(idx, t, data, fs, lid))
        self._worker.error.connect(
            lambda idx, msg, lid=current_load_id:
                self._on_data_error(idx, msg, lid))
        self._worker.start()

    def _rebuild_grid(self, stations):
        """Recrée la grille de StationPlot selon le nombre de stations."""
        # Vide l'ancienne grille
        for sp in self._station_plots:
            sp.setParent(None)
        self._station_plots.clear()

        # Hauteur par panneau : compacte pour voir tous les spectrogrammes d'un coup
        PLOT_HEIGHT = 200

        for i, sta in enumerate(stations):
            name = getattr(sta, 'name', f"Station {i}")
            sp = StationPlot(name)
            sp.set_mode(self.view_mode)
            sp.setFixedHeight(PLOT_HEIGHT)
            # sp.set_loading()
            self._station_plots.append(sp)
            # Colonne unique : toutes les stations s'empilent verticalement
            self.grid_layout.addWidget(sp, i, 0)

    def _on_data_ready(self, idx, t, data, fs, load_id):
        # Ignore les résultats d'un chargement précédent (navigation rapide)
        if load_id != self._load_id:
            return
        if idx >= len(self._station_plots):
            return
        self._station_plots[idx].update_data(t, data, fs)

    def _on_data_error(self, idx, msg, load_id):
        if load_id != self._load_id:
            return
        if idx < len(self._station_plots):
            self._station_plots[idx].set_error(msg)

    # ── Classification ────────────────────────────────────────────────────────
    def _classify(self, label: str):
        event_id = self.event_list[self.current_idx]
        self.labels[str(event_id)] = label
        self._save_labels()
        self._update_stats()
        self._populate_event_list(self.search_box.text())
        # met à jour l'en-tête
        color = LABEL_COLORS.get(label, "#aaaaaa")
        icon  = LABEL_ICONS.get(label, "")
        self.lbl_event.setText(
            f"Événement <span style='color:#cba6f7'>#{event_id}</span>"
            f"  —  <span style='color:{color}'>{icon} {label}</span>"
            f"  ({self.current_idx+1}/{len(self.event_list)})"
        )
        # avance automatiquement au suivant
        self._navigate(+1)

    def _navigate(self, delta: int):
        new_idx = self.current_idx + delta
        if 0 <= new_idx < len(self.event_list):
            self._load_event(new_idx)

    # ── Persistance ───────────────────────────────────────────────────────────
    def _save_labels(self):
        with open(self.labels_path, "w") as f:
            json.dump(self.labels, f, indent=2)

    def _update_stats(self):
        total = len(self.event_list)
        done  = sum(1 for eid in self.event_list if str(eid) in self.labels)
        v  = sum(1 for v in self.labels.values() if v == "Valide")
        u  = sum(1 for v in self.labels.values() if v == "Incertain")
        inv= sum(1 for v in self.labels.values() if v == "Invalide")
        self.lbl_stats.setText(
            f"<small>"
            f"<b>{done}/{total}</b> classés<br>"
            f"<span style='color:{LABEL_COLORS['Valide']}'>✅ {v}</span>  "
            f"<span style='color:{LABEL_COLORS['Incertain']}'>❓ {u}</span>  "
            f"<span style='color:{LABEL_COLORS['Invalide']}'>❌ {inv}</span>"
            f"</small>"
        )

    def _export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter CSV", "classifications.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "label"])
            for eid in self.event_list:
                w.writerow([eid, self.labels.get(str(eid), "")])
        QtWidgets.QMessageBox.information(self, "Export", f"Sauvegardé : {path}")

    # ── Signaux contrôle ──────────────────────────────────────────────────────
    def _on_filter_changed(self, low, high):
        self.lowcut  = low
        self.highcut = high
        self._load_event(self.current_idx)

    def _on_view_changed(self, mode: str):
        self.view_mode = mode
        for sp in self._station_plots:
            sp.set_mode(mode)
        # recharge pour ré-afficher dans le bon mode
        self._load_event(self.current_idx)

    def _on_window_changed(self, minutes: int):
        self.time_min = minutes
        self._load_event(self.current_idx)

    def _on_height_changed(self, px: int):
        """Redimensionne tous les panneaux à la volée sans recharger les données."""
        for sp in self._station_plots:
            sp.setFixedHeight(px)

    # ── Raccourcis clavier ────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        k = event.key()
        if k == QtCore.Qt.Key_Right:
            self._navigate(+1)
        elif k == QtCore.Qt.Key_Left:
            self._navigate(-1)
        elif k == QtCore.Qt.Key_V:
            self._classify("Valide")
        elif k == QtCore.Qt.Key_U:
            self._classify("Incertain")
        elif k == QtCore.Qt.Key_I:
            self._classify("Invalide")
        else:
            super().keyPressEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════
# def make_dummy_data(n_events=50, n_stations=8):
#     """Génère des données factices pour tester l'interface sans les vrais fichiers."""
#     import types
#
#     stations = []
#     for i in range(n_stations):
#         s = types.SimpleNamespace(
#             name=f"OHAS{i+1:02d}",
#             dataset="OHASISBIO-2018",
#             idx=i
#         )
#         stations.append(s)
#
#     # IDX_TO_DET : idx -> (det_time, prob, global_idx)
#     IDX_TO_DET = {}
#     idx_det = 0
#     base = datetime.datetime(2018, 10, 1, tzinfo=datetime.timezone.utc)
#     for e in range(n_events):
#         for s in range(n_stations):
#             t0 = base + datetime.timedelta(hours=e * 2 + s * 0.01)
#             IDX_TO_DET[idx_det] = (t0, 0.9, idx_det)
#             idx_det += 1
#
#     # associations : event_id -> [np.array shape (n_stations, 2)]
#     associations = {}
#     for e in range(n_events):
#         mat = np.array([[s, e * n_stations + s] for s in range(n_stations)])
#         associations[e] = [mat]
#
#     final_results = {e: True for e in range(n_events)}
#     return associations, stations, IDX_TO_DET, final_results


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # ── Palette sombre ────────────────────────────────────────────────────────
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window,          QtGui.QColor("#1e1e2e"))
    pal.setColor(QtGui.QPalette.WindowText,      QtGui.QColor("#cdd6f4"))
    pal.setColor(QtGui.QPalette.Base,            QtGui.QColor("#181825"))
    pal.setColor(QtGui.QPalette.AlternateBase,   QtGui.QColor("#1e1e2e"))
    pal.setColor(QtGui.QPalette.Text,            QtGui.QColor("#cdd6f4"))
    pal.setColor(QtGui.QPalette.Button,          QtGui.QColor("#313244"))
    pal.setColor(QtGui.QPalette.ButtonText,      QtGui.QColor("#cdd6f4"))
    pal.setColor(QtGui.QPalette.Highlight,       QtGui.QColor("#89b4fa"))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#1e1e2e"))
    app.setPalette(pal)

    # ── Chargement réel (décommente et adapte quand tes données sont disponibles)
    CATALOG_PATH   = "/media/rsafran/CORSAIR/OHASISBIO/recensement_stations_OHASISBIO_RS_V2.csv"
    DETECTIONS_DIR = "/media/rsafran/CORSAIR/T-pick_2.2"
    DATA_ROOT      = "/media/rsafran/CORSAIR/OHASISBIO"
    DATASET        = "OHASISBIO-2018"
    file_end       = "_SWIR_crise_10"
    YEAR           = 2018
    ASSOCIATIONS_DIR = f"../../../data/detection/T-pick_2.2/{YEAR}"

    from utils.data_reading.sound_data.station import StationsCatalog
    STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated()
    with open(f"{ASSOCIATIONS_DIR}/cache/detections_0.1_1{file_end}.pkl", "rb") as f:
        DETECTIONS = pickle.load(f)
    # ... (tout le bloc de préparation DETECTIONS / IDX_TO_DET de ton code) ...
    CATALOG_PATH = "/media/rsafran/CORSAIR/OHASISBIO/recensement_stations_OHASISBIO_RS_V2.csv"  # csv catalog files
    DETECTIONS_DIR = "/media/rsafran/CORSAIR/T-pick_2.2"  # where we have detection pickles
    ISAS_PATH = "/media/rsafran/CORSAIR/ISAS/extracted/2018"
    YEAR = 2018
    # Root directory containing subfolders for each station
    DATA_ROOT = "/media/rsafran/CORSAIR/OHASISBIO"
    # Subfolder (e.g. year) within DATA_ROOT
    DATASET = "OHASISBIO-2018"
    # output dir
    ASSOCIATIONS_DIR = f"../../../data/detection/T-pick_2.2/{YEAR}"
    file_end = '_SWIR_crise_10'  # '_all'#'_SWIR_crise_10'
    # ASSOCIATIONS_DIR = f"../../data/detection/H-pick_MAHY/{YEAR}"
    # DETECTIONS_DIR = "/media/rsafran/CORSAIR/H-pick_MAHY"
    # file_end= '_impulsive_MAHY'

    DATE_START = datetime.datetime(YEAR, 9, 26) - datetime.timedelta(hours=2)
    DATE_END = datetime.datetime(YEAR, 10, 26) + datetime.timedelta(hours=2)

    # Detections loading parameters
    MIN_P_TISSNET_PRIMARY = 0.5  # min probability of browsed detections #0.5 / 0.1 crise et 0.6/0.3 pour globale
    MIN_P_TISSNET_SECONDARY = 0.1  # min probability of detections that can be associated with the browsed one
    MERGE_DELTA_S = 1  # threshold below which we consider two events should be merged

    Path(ASSOCIATIONS_DIR).mkdir(exist_ok=True)

    # # delimitation of the detections we keep (this notebook actually associates the detections of 1 year and 4 hours)
    # DATE_START = datetime.datetime(YEAR, 1, 1) - datetime.timedelta(hours=2)
    # DATE_END = datetime.datetime(YEAR+1, 3, 1) + datetime.timedelta(hours=2)
    # #crise
    # # DATE_START = datetime.datetime(YEAR, 7, 10) - datetime.timedelta(hours=2)
    # # DATE_END = datetime.datetime(YEAR+1, 7, 15) + datetime.timedelta(hours=2)
    #
    # # Detections loading parameters
    # MIN_P_TISSNET_PRIMARY = 0.6 # min probability of browsed detections
    # MIN_P_TISSNET_SECONDARY = 0.3  # min probability of detections that can be associated with the browsed one
    # MERGE_DELTA_S = 1# threshold below which we consider two events should be merged

    # The REQ_CLOSEST_STATIONS th closest stations will be required for an association to be valid
    # e.g. if we set it to 6, no association of size <6 will be saved (this is useful to save memory)
    REQ_CLOSEST_STATIONS = 6

    STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated()

    Path(f"{ASSOCIATIONS_DIR}/cache").mkdir(parents=True, exist_ok=True)
    DET_PATH = f"{ASSOCIATIONS_DIR}/cache/detections_{MIN_P_TISSNET_SECONDARY}_{MERGE_DELTA_S}{file_end}.pkl"
    with open(DET_PATH, "rb") as f:
        DETECTIONS = pickle.load(f)

    # do not keep detection entries for which the detection list is empty
    to_del = []
    for s in DETECTIONS.keys():
        if len(DETECTIONS[s]) == 0:
            to_del.append(s)
        if s.name == 'H04S1':  # or s.name == 'H04N1' or s.name == 'H01W1':
            to_del.append(s)
    for s in to_del:
        del DETECTIONS[s]

    # assign an index to each detection
    idx_det = 0
    IDX_TO_DET = {}
    for idx, s in enumerate(DETECTIONS.keys()):
        s.idx = idx  # indexes to store efficiently the associations
        DETECTIONS[s] = list(DETECTIONS[s])
        for i in range(len(DETECTIONS[s])):
            DETECTIONS[s][i] = np.concatenate((DETECTIONS[s][i], [idx_det]))
            IDX_TO_DET[idx_det] = DETECTIONS[s][i]
            idx_det += 1
        DETECTIONS[s] = np.array(DETECTIONS[s])
    DETECTION_IDXS = np.array(list(range(idx_det)))

    # only keep the stations that appear in the kept detections
    STATIONS = [s for s in DETECTIONS.keys()]
    FIRSTS_DETECTIONS = {s: DETECTIONS[s][0, 0] for s in STATIONS}
    LASTS_DETECTIONS = {s: DETECTIONS[s][-1, 0] for s in STATIONS}

    #association loading
    files = glob2.glob(f"{DETECTIONS_DIR}/cache/associations_2018{file_end}.pkl")
    with open(files[0], "rb") as f:
        associations = pickle.load(f)
        print(len(associations))
    #final catalogue loading
    with open(f"{DETECTIONS_DIR}/cache/filtered_results_2018{file_end}.pkl", "rb") as f:
        final_results = pickle.load(f)
    print(list(final_results.keys()))


    # ── Mode démo avec données synthétiques ───────────────────────────────────
    # associations, STATIONS, IDX_TO_DET, final_results = make_dummy_data(
    #     n_events=40, n_stations=6)
    # DATA_ROOT = ""
    # DATASET   = ""
    #
    win = MainWindow(
        associations  = associations,
        STATIONS      = STATIONS,
        IDX_TO_DET    = IDX_TO_DET,
        final_results = final_results,
        DATA_ROOT     = DATA_ROOT,
        DATASET       = DATASET,
        labels_path   = "classifications.json",
    )
    win.show()
    sys.exit(app.exec_())

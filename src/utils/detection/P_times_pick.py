import datetime
import sys
import os
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy import signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QListWidget, QPushButton, QLabel, QFileDialog, QSplitter,
                             QMessageBox, QComboBox, QGroupBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QSettings, QObject, pyqtSignal
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib.patches as patches
from src.utils.data_reading.sound_data.sound_file import DatFile
from src.utils.data_reading.sound_data.sound_file_manager import make_manager


# main_window.py
class PhasePickerApp(QMainWindow):
    """Main application window that connects all components."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Seismic Phase Picker")

        # Create components
        self.settings_manager = SettingsManager()
        self.data_manager = DataManager()
        self.signal_processor = SignalProcessor()

        # Setup UI
        self.setup_ui()
        self.connect_signals()

        # Restore settings
        self.restore_settings()

    def setup_ui(self):
        """Initialize and layout UI components."""
        # Main layout setup
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Create UI components
        self.plot_widget = PlotWidget()
        self.control_panel = ControlPanel()

        # Setup splitter
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.plot_widget)
        self.splitter.addWidget(self.control_panel)
        self.layout.addWidget(self.splitter)

    def connect_signals(self):
        """Connect all component signals and slots."""
        # Connect control panel signals
        self.control_panel.load_button.clicked.connect(self.open_dataset)
        self.control_panel.prev_button.clicked.connect(self.data_manager.previous_file)
        # ... other connections ...

        # Connect data manager signals
        self.data_manager.file_changed.connect(self.update_display)
        # ... other connections ...



# models/seismic_event.py
class SeismicEvent:
    """Class representing a seismic event with associated data."""

    def __init__(self, file_path, file_idx=None):
        self.path = file_path
        self.file_idx = file_idx
        self.phases = []
        self.theoretical_times = []
        self.distance = None
        self.depth = None
        self.event_datetime = None
        self.magnitude = None
        self.picks = {}  # phase -> pick_time

    @property
    def filename(self):
        """Get file basename."""
        return os.path.basename(self.path)

    def add_phase(self, phase_name, theoretical_time, distance=None, depth=None):
        """Add a theoretical phase to this event."""
        self.phases.append(phase_name)
        self.theoretical_times.append(theoretical_time)

        # Store additional metadata if provided
        if len(self.phases) > len(self.theoretical_times):
            idx = len(self.phases) - 1
            if distance is not None:
                if not hasattr(self, 'distances'):
                    self.distances = [None] * len(self.phases)
                self.distances[idx] = distance


# core/data_manager.py
class DataManager(QObject):
    """Manages seismic data files and navigation."""

    # Define signals
    file_changed = pyqtSignal(SeismicEvent)
    loading_progress = pyqtSignal(int, int)  # current, total
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.file_map = []
        self.current_index = 0
        self.files_manager = None

    def load_dataset(self, mat_path, data_dir):
        """Load MAT file and data directory."""
        try:
            # Loading implementation
            # ...

            # Emit progress signals during loading
            self.loading_progress.emit(i, total_files)

        except Exception as e:
            self.error_occurred.emit(f"Failed to load dataset: {str(e)}")

    def next_file(self):
        """Navigate to next file."""
        if not self.file_map:
            return False

        if self.current_index < len(self.file_map) - 1:
            self.current_index += 1
            self.file_changed.emit(self.current_file)
            return True
        return False

    @property
    def current_file(self):
        """Get current file or None."""
        if not self.file_map or self.current_index >= len(self.file_map):
            return None
        return self.file_map[self.current_index]


# core/signal_processor.py
class SignalProcessor:
    """Handles signal filtering and processing."""

    def __init__(self):
        self.sampling_frequency = 240  # Default
        self.setup_filters()

    def setup_filters(self):
        """Initialize filter coefficients."""
        # Filter 1 (0.6-2.1 Hz)
        self.b1_high, self.a1_high = signal.butter(4, 0.6 / (self.sampling_frequency / 2), 'high')
        self.b1_low, self.a1_low = signal.butter(6, 2.1 / (self.sampling_frequency / 2), 'low')

        # Filter 2 (0.06-2.1 Hz)
        self.b2_high, self.a2_high = signal.butter(4, 0.06 / (self.sampling_frequency / 2), 'high')
        self.b2_low, self.a2_low = signal.butter(6, 2.1 / (self.sampling_frequency / 2), 'low')

    def apply_filter(self, data, filter_type=1):
        """Apply the selected filter to data."""
        if filter_type == 0:
            return data  # Raw signal

        if filter_type == 1:
            # Apply Filter 1 (0.6-2.1 Hz)
            filtered = signal.filtfilt(self.b1_high, self.a1_high, data)
            return signal.filtfilt(self.b1_low, self.a1_low, filtered)

        if filter_type == 2:
            # Apply Filter 2 (0.06-2.1 Hz)
            filtered = signal.filtfilt(self.b2_high, self.a2_high, data)
            return signal.filtfilt(self.b2_low, self.a2_low, filtered)

        return data  # Default case


# ui/plot_widget.py
class PlotWidget(QWidget):
    """Widget for displaying seismic plots."""

    # Define signals
    pick_made = pyqtSignal(str, float)  # phase, time

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

        # Plot state
        self.signal_line = None
        self.theoretical_lines = []
        self.marked_arrivals = []
        self.selected_phase = None

    def setup_ui(self):
        """Setup the plot UI."""
        self.layout = QVBoxLayout(self)

        # Figure setup
        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add to layout
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)

        # Connect events
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

        # Setup span selector for interactive zooming
        self.span_selector = SpanSelector(
            self.ax, self.on_span_select, 'horizontal',
            useblit=True, props=dict(alpha=0.3, facecolor='blue'),
            interactive=True, drag_from_anywhere=True
        )

    def plot_signal(self, signal_data, time_data, filter_type=1, filter_processor=None):
        """Plot signal with the specified filter."""
        if self.signal_line:
            self.signal_line.remove()

        if filter_processor and filter_type > 0:
            filtered_data = filter_processor.apply_filter(signal_data, filter_type)
        else:
            filtered_data = signal_data

        self.signal_line, = self.ax.plot(time_data, filtered_data,
                                         color='r' if filter_type == 1 else 'b' if filter_type == 2 else 'k',
                                         label=self.get_filter_label(filter_type),
                                         alpha=0.7)
        self.update_legend()
        self.canvas.draw()

    def plot_theoretical_times(self, times, phases):
        """Plot theoretical arrival times."""
        # Implementation
        # ...

    def on_plot_click(self, event):
        """Handle plot clicks to mark arrival times."""
        if event.inaxes != self.ax or not self.selected_phase:
            return

        picked_time = event.xdata
        self.pick_made.emit(self.selected_phase, picked_time)
        self.mark_arrival(self.selected_phase, picked_time, event.ydata)


# models/settings.py
class SettingsManager:
    """Manages application settings."""

    def __init__(self):
        self.settings = QSettings("SeismicTools", "PhasePicker")

    def save_geometry(self, window):
        """Save window geometry."""
        self.settings.setValue("geometry", window.saveGeometry())

    def restore_geometry(self, window):
        """Restore window geometry."""
        if self.settings.contains("geometry"):
            window.restoreGeometry(self.settings.value("geometry"))

    def save_splitter_sizes(self, splitter):
        """Save splitter sizes."""
        self.settings.setValue("splitter_sizes", splitter.sizes())

    def restore_splitter_sizes(self, splitter):
        """Restore splitter sizes."""
        if self.settings.contains("splitter_sizes"):
            sizes_variant = self.settings.value("splitter_sizes")
            if isinstance(sizes_variant, list):
                sizes = [int(size) for size in sizes_variant]
                splitter.setSizes(sizes)
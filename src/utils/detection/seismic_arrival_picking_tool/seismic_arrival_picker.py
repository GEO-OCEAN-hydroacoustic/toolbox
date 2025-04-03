import sys
import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from datetime import datetime, timedelta
from scipy import signal

# This is a placeholder for the DatFilesManager import
# You'll need to provide the actual path to this module
try:
    from src.utils.data_reading.sound_data.sound_file_manager import DatFilesManager
except ImportError:
    # Placeholder class for testing without the actual module
    class DatFilesManager:
        def __init__(self, path):
            self.path = path
            self.sampling_f = 240  # Default sampling frequency

class SignalProcessing:
    @staticmethod
    def dehaze_audio(data, fs, frame_size=1024, overlap=0.8):
        """Apply spectral subtraction for dehazing"""
        hop_size = int(frame_size * (1 - overlap))
        num_noise_frames = 5
        noise_estimate = np.zeros(frame_size // 2 + 1)
        frames = []
        for i in range(0, len(data) - frame_size, hop_size):
            frame = data[i:i+frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            frames.append(frame)

        # If we have no frames, return the original data
        if not frames:
            return data

        for i in range(min(num_noise_frames, len(frames))):
            noise_frame = frames[i]
            noise_spectrum = np.abs(np.fft.rfft(noise_frame * np.hanning(frame_size)))
            noise_estimate += noise_spectrum / num_noise_frames

        result = np.zeros(len(data))
        window = np.hanning(frame_size)
        for i, frame in enumerate(frames):
            windowed_frame = frame * window
            spectrum = np.fft.rfft(windowed_frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            magnitude = np.maximum(magnitude - noise_estimate * 1.5, 0.01 * magnitude)
            enhanced_spectrum = magnitude * np.exp(1j * phase)
            enhanced_frame = np.fft.irfft(enhanced_spectrum)
            start = i * hop_size
            end = min(start + frame_size, len(result))
            result[start:end] += enhanced_frame[:end-start]

        # Normalize the result
        if np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * np.max(np.abs(data))
        return result

    @staticmethod
    def apply_butter_bandpass(data, fs, lowcut, highcut, order=5):
        """Apply Butterworth bandpass filter"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data

    @staticmethod
    def energy_plot(data, fs, window_size=1.0):
        """Detect potential seismic events based on energy threshold"""
        window_samples = int(window_size * fs)
        if window_samples <= 0:
            return np.array([]), np.array([]), 0

        energy = []
        time_points = []
        for i in range(0, len(data) - window_samples, window_samples // 2):
            window = data[i:i+window_samples]
            window_energy = np.sum(window**2) / len(window)
            energy.append(window_energy)
            time_points.append(i / fs)

        energy = np.array(energy)
        time_points = np.array(time_points)

        if len(energy) == 0:
            return np.array([]), np.array([]), 0

        threshold = np.median(energy) * 2  # Adjustable multiplier
        event_indices = np.where(energy > threshold)[0]

        return event_indices, time_points, energy, threshold


class SeismicArrivalPicker(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up main UI
        self.setWindowTitle("Interactive Seismic Arrival Picking")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.catalogue_df = None
        self.current_event_index = 0
        self.current_event_df = None
        self.waveform_data = None
        self.time_array = None
        self.sampling_frequency = 240  # Default value, will be updated when loading data
        self.observed_arrivals = {}  # Dictionary to store observed arrivals by event and phase
        self.data_path = ""
        self.start_time = None
        self.end_time = None
        self.draggable_lines = {}  # Dictionary to store draggable lines by phase
        self.fixed_lines = {}  # Dictionary to store fixed lines by phase

        # Create central widget and main layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Create a splitter for the two plot windows
        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_layout.addWidget(self.plot_splitter, stretch=1)

        # Create top plot widget for raw data
        self.raw_graph_widget = pg.PlotWidget()
        self.raw_graph_widget.setBackground('k')  # Black background
        self.raw_graph_widget.setLabel('left', 'Amplitude')
        self.raw_graph_widget.setLabel('bottom', 'Time (seconds)')
        self.raw_graph_widget.showGrid(x=True, y=True, alpha=0.3)
        self.raw_graph_widget.setTitle("Raw Waveform Data")

        # Create bottom plot widget for filtered data
        self.filtered_graph_widget = pg.PlotWidget()
        self.filtered_graph_widget.setBackground('k')  # Black background
        self.filtered_graph_widget.setLabel('left', 'Amplitude')
        self.filtered_graph_widget.setLabel('bottom', 'Time (seconds)')
        self.filtered_graph_widget.showGrid(x=True, y=True, alpha=0.3)
        self.filtered_graph_widget.setTitle("Filtered Waveform Data")

        # Add plot widgets to splitter
        self.plot_splitter.addWidget(self.raw_graph_widget)
        self.plot_splitter.addWidget(self.filtered_graph_widget)

        # Set initial sizes (equal division)
        self.plot_splitter.setSizes([400, 400])

        # Create plot items for each graph
        self.raw_plot = self.raw_graph_widget.plot(pen=pg.mkPen('w', width=1))
        self.filtered_plot = self.filtered_graph_widget.plot(pen=pg.mkPen('c', width=1))
        self.energy_plot_item = self.filtered_graph_widget.plot(pen=pg.mkPen('y', width=1))

        # Create controls layout
        controls_layout = QtWidgets.QHBoxLayout()

        # Create navigation controls
        nav_group = QtWidgets.QGroupBox("Navigation")
        nav_layout = QtWidgets.QHBoxLayout()

        self.prev_button = QtWidgets.QPushButton("Previous Event")
        self.next_button = QtWidgets.QPushButton("Next Event")
        self.event_label = QtWidgets.QLabel("Event: 0 / 0")

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.event_label)
        nav_layout.addWidget(self.next_button)

        nav_group.setLayout(nav_layout)
        controls_layout.addWidget(nav_group)

        # Create filter controls
        filter_group = QtWidgets.QGroupBox("Filtering Options")
        filter_layout = QtWidgets.QVBoxLayout()

        self.dehaze_checkbox = QtWidgets.QCheckBox("Apply Dehazing")
        self.bandpass_checkbox = QtWidgets.QCheckBox("Apply Bandpass Filter")
        self.energy_checkbox = QtWidgets.QCheckBox("Show Energy Plot")

        filter_params_layout = QtWidgets.QHBoxLayout()
        self.lowcut_label = QtWidgets.QLabel("Low Cut (Hz):")
        self.lowcut_input = QtWidgets.QLineEdit("1.0")
        self.highcut_label = QtWidgets.QLabel("High Cut (Hz):")
        self.highcut_input = QtWidgets.QLineEdit("20.0")

        filter_params_layout.addWidget(self.lowcut_label)
        filter_params_layout.addWidget(self.lowcut_input)
        filter_params_layout.addWidget(self.highcut_label)
        filter_params_layout.addWidget(self.highcut_input)

        filter_layout.addWidget(self.dehaze_checkbox)
        filter_layout.addWidget(self.bandpass_checkbox)
        filter_layout.addLayout(filter_params_layout)
        filter_layout.addWidget(self.energy_checkbox)

        filter_group.setLayout(filter_layout)
        controls_layout.addWidget(filter_group)

        # Create cursor controls
        cursor_group = QtWidgets.QGroupBox("Cursor Controls")
        cursor_layout = QtWidgets.QVBoxLayout()

        self.reset_button = QtWidgets.QPushButton("Reset All Cursors")
        self.phase_info_label = QtWidgets.QLabel("Current Phases: None")

        cursor_layout.addWidget(self.reset_button)
        cursor_layout.addWidget(self.phase_info_label)

        cursor_group.setLayout(cursor_layout)
        controls_layout.addWidget(cursor_group)

        # Add controls layout to main layout
        main_layout.addLayout(controls_layout)

        # Create status bar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        # Create menu bar
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        load_catalogue_action = QtWidgets.QAction("Load Catalogue", self)
        load_catalogue_action.setShortcut("Ctrl+O")
        file_menu.addAction(load_catalogue_action)

        load_observed_action = QtWidgets.QAction("Load Observed Arrivals", self)
        file_menu.addAction(load_observed_action)

        save_observed_action = QtWidgets.QAction("Save Observed Arrivals", self)
        save_observed_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_observed_action)

        file_menu.addSeparator()

        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)

        # Connect signals and slots
        load_catalogue_action.triggered.connect(self.load_catalogue)
        load_observed_action.triggered.connect(self.load_observed_arrivals)
        save_observed_action.triggered.connect(self.save_observed_arrivals)
        exit_action.triggered.connect(self.close)

        self.prev_button.clicked.connect(self.previous_event)
        self.next_button.clicked.connect(self.next_event)
        self.reset_button.clicked.connect(self.reset_all_cursors)

        self.dehaze_checkbox.stateChanged.connect(self.update_filtered_signal)
        self.bandpass_checkbox.stateChanged.connect(self.update_filtered_signal)
        self.energy_checkbox.stateChanged.connect(self.update_filtered_signal)
        self.lowcut_input.textChanged.connect(self.update_filtered_signal)
        self.highcut_input.textChanged.connect(self.update_filtered_signal)

    def load_catalogue(self):
        """Load catalogue from CSV file"""
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Catalogue File", "", "CSV Files (*.csv)")

        if not file_path:
            return

        try:
            # Load catalogue file
            self.statusBar.showMessage(f"Loading catalogue from {file_path}...")
            self.catalogue_df = pd.read_csv(file_path, parse_dates=['time', 'arrival_time'],infer_datetime_format=True)
            self.catalogue_df['time'] = pd.to_datetime(self.catalogue_df['time'], utc=True, errors='coerce')

            # Add observed_arrival_time column if it doesn't exist
            if 'observed_arrival_time' not in self.catalogue_df.columns:
                self.catalogue_df['observed_arrival_time'] = pd.NaT

            # Filter by candidate flag if it exists
            if 'candidate' in self.catalogue_df.columns:
                self.catalogue_df = self.catalogue_df[self.catalogue_df['candidate'] == True]

            # Get unique event times
            self.event_times = self.catalogue_df['time'].unique()
            self.current_event_index = 0

            # Ask for data directory
            data_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Data Directory")
            if data_dir:
                self.data_path = data_dir

                # Load first event
                self.load_event(self.current_event_index)
                self.event_label.setText(f"Event: {self.current_event_index + 1} / {len(self.event_times)}")
                self.statusBar.showMessage(f"Loaded catalogue with {len(self.event_times)} events.")
            else:
                self.statusBar.showMessage("Data directory not selected.")

        except Exception as e:
            self.statusBar.showMessage(f"Error loading catalogue: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load catalogue: {str(e)}")

    def load_event(self, event_index):
        """Load waveform data for the specified event index"""
        if self.catalogue_df is None or event_index < 0 or event_index >= len(self.event_times):
            return

        event_time = self.event_times[event_index]
        self.current_event_df = self.catalogue_df[self.catalogue_df['time'] == event_time]

        # Extract event details for display
        event_details = self.current_event_df.iloc[0:1]

        # Find the first arrival time for this event
        first_arrival = self.current_event_df['arrival_time'].min()

        # Make sure first_arrival is timezone-aware before operations
        if first_arrival.tzinfo is not None:
            # Define 20-minute window (10 minutes before and after)
            self.start_time = (first_arrival - timedelta(minutes=10)).replace(tzinfo=None)
            self.end_time = (first_arrival + timedelta(minutes=10)).replace(tzinfo=None)
        else:
            # If already timezone-naive, no need to replace tzinfo
            self.start_time = first_arrival - timedelta(minutes=10)
            self.end_time = first_arrival + timedelta(minutes=10)


        try:
            # Load waveform data
            self.statusBar.showMessage(f"Loading waveform data for event at {event_time}...")

            # Create DatFilesManager and get waveform segment
            manager = DatFilesManager(self.data_path, kwargs='raw')

            # Get file number
            file_number = manager.find_file_name(self.start_time) if hasattr(manager, 'find_file_name') else "Unknown"

            self.sampling_frequency = manager.sampling_f
            self.waveform_data = manager.get_segment(self.start_time, self.end_time)

            # Create plot title with event details
            title_text = f"Event: {event_time}"

            # Add magnitude if available
            if 'mag' in event_details.columns:
                title_text += f" | Magnitude: {event_details['mag'].iloc[0]:.2f}"

            # Add location if available
            if 'place' in event_details.columns:
                title_text += f" | Location: {event_details['place'].iloc[0]}"

            # Add depth if available
            if 'depth' in event_details.columns:
                title_text += f" | Depth: {event_details['depth'].iloc[0]:.2f} km"

            # Add errors if available
            if 'horizontalError' in event_details.columns:
                title_text += f" | H.Err: {event_details['horizontalError'].iloc[0]:.2f} km"

            if 'depthError' in event_details.columns:
                title_text += f" | D.Err: {event_details['depthError'].iloc[0]:.2f} km"

            # Add file number
            title_text += f" | DATFile: {file_number}"

            # Set the plot title
            self.raw_graph_widget.setTitle(title_text)

            # Create time array in seconds
            self.time_array = np.arange(len(self.waveform_data)) / self.sampling_frequency

            # Update plots
            self.update_plots()

            # Update UI
            phase_list = ', '.join(self.current_event_df['phase'].unique())
            self.phase_info_label.setText(f"Current Phases: {phase_list}")
            self.statusBar.showMessage(f"Loaded event at {event_time} with {len(self.current_event_df)} arrivals.")

        except Exception as e:
            self.statusBar.showMessage(f"Error loading waveform data: {str(e)}")
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load waveform data: {str(e)}")

    def update_plots(self):
        """Update all plots with current data"""
        if self.waveform_data is None or self.time_array is None:
            return

        # Clear existing plots and lines
        self.raw_graph_widget.clear()
        self.filtered_graph_widget.clear()

        # Recreate plot items
        self.raw_plot = self.raw_graph_widget.plot(pen=pg.mkPen('w', width=1))
        self.filtered_plot = self.filtered_graph_widget.plot(pen=pg.mkPen('c', width=1))
        self.energy_plot_item = self.filtered_graph_widget.plot(pen=pg.mkPen('y', width=1))

        # Plot raw waveform
        self.raw_plot.setData(self.time_array, self.waveform_data)

        # Update filtered signal
        self.update_filtered_signal()

        # Update arrival markers on both plots
        self.update_arrival_markers()

    def update_filtered_signal(self):
        """Apply selected filters to the waveform data"""
        if self.waveform_data is None or self.time_array is None:
            return

        # Clear filtered plots
        self.filtered_plot.clear()
        self.energy_plot_item.clear()

        filtered_data = self.waveform_data.copy()

        # Apply dehazing if selected
        if self.dehaze_checkbox.isChecked():
            filtered_data = SignalProcessing.dehaze_audio(filtered_data, self.sampling_frequency)

        # Apply bandpass filter if selected
        if self.bandpass_checkbox.isChecked():
            try:
                lowcut = float(self.lowcut_input.text())
                highcut = float(self.highcut_input.text())
                filtered_data = SignalProcessing.apply_butter_bandpass(
                    filtered_data, self.sampling_frequency, lowcut, highcut
                )
            except ValueError:
                self.statusBar.showMessage("Invalid filter parameters. Using default values.")
                filtered_data = SignalProcessing.apply_butter_bandpass(
                    filtered_data, self.sampling_frequency, 1.0, 20.0
                )

        # Show filtered signal
        self.filtered_plot.setData(self.time_array, filtered_data)

        # Calculate and show energy plot if selected
        if self.energy_checkbox.isChecked():
            try:
                _, time_points, energy, _ = SignalProcessing.energy_plot(
                    filtered_data, self.sampling_frequency
                )
                if len(time_points) > 0 and len(energy) > 0:
                    # Scale energy to match waveform amplitude
                    scaled_energy = energy / np.max(energy) * np.max(np.abs(filtered_data))
                    self.energy_plot_item.setData(time_points, scaled_energy)
            except Exception as e:
                self.statusBar.showMessage(f"Error calculating energy plot: {str(e)}")

        # Make sure the plots have synchronized ranges
        self.sync_plot_ranges()

    def update_arrival_markers(self):
        """Update arrival markers for the current event"""
        if self.current_event_df is None or self.start_time is None:
            return

        # Clear existing lines from both plots
        for phase, (fixed_line_raw, fixed_line_filtered) in list(self.fixed_lines.items()):
            if fixed_line_raw in self.raw_graph_widget.items():
                self.raw_graph_widget.removeItem(fixed_line_raw)
            if fixed_line_filtered in self.filtered_graph_widget.items():
                self.filtered_graph_widget.removeItem(fixed_line_filtered)

        for phase, (draggable_line_raw, draggable_line_filtered) in list(self.draggable_lines.items()):
            if draggable_line_raw in self.raw_graph_widget.items():
                self.raw_graph_widget.removeItem(draggable_line_raw)
            if draggable_line_filtered in self.filtered_graph_widget.items():
                self.filtered_graph_widget.removeItem(draggable_line_filtered)

        self.fixed_lines.clear()
        self.draggable_lines.clear()

        # Get reference time
        reference_time = self.start_time

        # Get current event time as string for dictionary lookup
        event_time = self.event_times[self.current_event_index]
        event_key = str(event_time)

        # Process unique phases
        processed_phases = set()

        for idx, row in self.current_event_df.iterrows():
            phase = row['phase']

            # Skip duplicate phases
            if phase in processed_phases:
                continue

            processed_phases.add(phase)

            # Ensure timezone consistency
            arrival_time = row['arrival_time']
            if arrival_time.tzinfo is not None:
                arrival_time = arrival_time.replace(tzinfo=None)

            # Calculate predicted time
            predicted_time = (arrival_time - reference_time).total_seconds()

            # Get observed time with fallbacks
            if event_key in self.observed_arrivals and phase in self.observed_arrivals[event_key]:
                observed_time = self.observed_arrivals[event_key][phase]
            elif 'observed_arrival_time' in row and pd.notna(row['observed_arrival_time']):
                observed_time = row['observed_arrival_time']
                if observed_time.tzinfo is not None:
                    observed_time = observed_time.replace(tzinfo=None)
                observed_time = (observed_time - reference_time).total_seconds()

                # Cache this for future use
                if event_key not in self.observed_arrivals:
                    self.observed_arrivals[event_key] = {}
                self.observed_arrivals[event_key][phase] = observed_time
            else:
                observed_time = predicted_time

            # Create fixed (predicted) lines for both plots
            fixed_line_raw = pg.InfiniteLine(
                pos=predicted_time,
                angle=90,
                pen=pg.mkPen('g', width=2, style=QtCore.Qt.DashLine),
                label=f"Predicted {phase}",
                labelOpts={
                    'position': 0.9,
                    'color': (200, 200, 200),
                    'fill': (0, 0, 0, 50),
                    'movable': False
                }
            )

            fixed_line_filtered = pg.InfiniteLine(
                pos=predicted_time,
                angle=90,
                pen=pg.mkPen('g', width=2, style=QtCore.Qt.DashLine),
                label=f"Predicted {phase}",
                labelOpts={
                    'position': 0.9,
                    'color': (200, 200, 200),
                    'fill': (0, 0, 0, 50),
                    'movable': False
                }
            )

            # Create draggable (observed) lines for both plots
            draggable_line_raw = pg.InfiniteLine(
                pos=observed_time,
                angle=90,
                pen=pg.mkPen('r', width=2),
                label=f"Observed {phase}",
                labelOpts={
                    'position': 0.1,
                    'color': (200, 200, 200),
                    'fill': (0, 0, 0, 50),
                    'movable': False
                },
                movable=True  # Make draggable in raw plot
            )

            draggable_line_filtered = pg.InfiniteLine(
                pos=observed_time,
                angle=90,
                pen=pg.mkPen('r', width=2),
                label=f"Observed {phase}",
                labelOpts={
                    'position': 0.1,
                    'color': (200, 200, 200),
                    'fill': (0, 0, 0, 50),
                    'movable': False
                },
                movable=True  # Make draggable in filtered plot
            )

            # Store phase for later reference
            draggable_line_raw.phase = phase
            draggable_line_filtered.phase = phase

            # Store arrival ID for tracking
            arrival_id = idx  # Use row index as ID
            draggable_line_raw.arrival_id = arrival_id
            draggable_line_filtered.arrival_id = arrival_id

            # Connect draggable line movement for both plots
            draggable_line_raw.sigPositionChanged.connect(
                lambda line=draggable_line_raw: self.on_line_moved(line)
            )
            draggable_line_filtered.sigPositionChanged.connect(
                lambda line=draggable_line_filtered: self.on_line_moved(line)
            )

            # Add lines to both plots
            self.raw_graph_widget.addItem(fixed_line_raw)
            self.raw_graph_widget.addItem(draggable_line_raw)
            self.filtered_graph_widget.addItem(fixed_line_filtered)
            self.filtered_graph_widget.addItem(draggable_line_filtered)

            # Store lines in dictionaries
            self.fixed_lines[phase] = (fixed_line_raw, fixed_line_filtered)
            self.draggable_lines[phase] = (draggable_line_raw, draggable_line_filtered)

        # Sync the two plots' X-axis ranges
        self.sync_plot_ranges()


    def sync_plot_ranges(self):
        """Synchronize the view ranges of both plots"""
        # Get the current view range of the raw plot
        raw_view_range = self.raw_graph_widget.viewRange()
        # Apply it to the filtered plot
        self.filtered_graph_widget.setXRange(*raw_view_range[0], padding=0)

        # Connect the X-range change signals for both plots
        self.raw_graph_widget.sigXRangeChanged.connect(self.on_raw_range_changed)
        self.filtered_graph_widget.sigXRangeChanged.connect(self.on_filtered_range_changed)

    def on_raw_range_changed(self, plot_item):
        """Handle range change in raw plot"""
        # Disconnect signal temporarily to avoid recursive calls
        self.filtered_graph_widget.sigXRangeChanged.disconnect(self.on_filtered_range_changed)
        # Set the filtered plot's X range to match
        view_range = plot_item.viewRange()
        self.filtered_graph_widget.setXRange(*view_range[0], padding=0)
        # Reconnect signal
        self.filtered_graph_widget.sigXRangeChanged.connect(self.on_filtered_range_changed)

    def on_filtered_range_changed(self, plot_item):
        """Handle range change in filtered plot"""
        # Disconnect signal temporarily to avoid recursive calls
        self.raw_graph_widget.sigXRangeChanged.disconnect(self.on_raw_range_changed)
        # Set the raw plot's X range to match
        view_range = plot_item.viewRange()
        self.raw_graph_widget.setXRange(*view_range[0], padding=0)
        # Reconnect signal
        self.raw_graph_widget.sigXRangeChanged.connect(self.on_raw_range_changed)

    def on_line_moved(self, line):
        """Handle draggable line movement"""
        new_time = line.value()
        phase = line.phase
        arrival_id = line.arrival_id

        # Update the other line with the same position
        for stored_phase, (line_raw, line_filtered) in self.draggable_lines.items():
            if stored_phase == phase:
                # Update both lines to stay in sync
                line_raw.blockSignals(True)  # Prevent recursive signal firing
                line_filtered.blockSignals(True)

                line_raw.setValue(new_time)
                line_filtered.setValue(new_time)

                line_raw.blockSignals(False)
                line_filtered.blockSignals(False)
                break

        # Store observed time in dictionary
        event_time = self.event_times[self.current_event_index]
        event_key = str(event_time)

        if event_key not in self.observed_arrivals:
            self.observed_arrivals[event_key] = {}

        self.observed_arrivals[event_key][phase] = new_time

        # Update status bar
        self.statusBar.showMessage(f"Observed arrival for phase {phase} updated to: {new_time:.2f} seconds")

    def reset_all_cursors(self):
        """Reset all draggable cursors to their original predicted positions"""
        if not self.fixed_lines or not self.draggable_lines:
            return

        for phase, (fixed_line_raw, fixed_line_filtered) in self.fixed_lines.items():
            if phase in self.draggable_lines:
                draggable_line_raw, draggable_line_filtered = self.draggable_lines[phase]

                # Get predicted position
                predicted_time = fixed_line_raw.value()  # Use either fixed line

                # Block signals temporarily to prevent recursion
                draggable_line_raw.blockSignals(True)
                draggable_line_filtered.blockSignals(True)

                # Update both draggable lines
                draggable_line_raw.setValue(predicted_time)
                draggable_line_filtered.setValue(predicted_time)

                # Restore signals
                draggable_line_raw.blockSignals(False)
                draggable_line_filtered.blockSignals(False)

                # Update observed arrivals dictionary
                event_time = self.event_times[self.current_event_index]
                event_key = str(event_time)

                if event_key in self.observed_arrivals:
                    self.observed_arrivals[event_key][phase] = predicted_time

        self.statusBar.showMessage("All cursors reset to predicted positions.")

    def previous_event(self):
        """Load the previous event"""
        if self.current_event_index > 0:
            self.current_event_index -= 1
            self.load_event(self.current_event_index)
            self.event_label.setText(f"Event: {self.current_event_index + 1} / {len(self.event_times)}")

    def next_event(self):
        """Load the next event"""
        if self.event_times is not None and self.current_event_index < len(self.event_times) - 1:
            self.current_event_index += 1
            self.load_event(self.current_event_index)
            self.event_label.setText(f"Event: {self.current_event_index + 1} / {len(self.event_times)}")

    def load_observed_arrivals(self):
        """Load observed arrival times from CSV file"""
        if self.catalogue_df is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load catalogue file first.")
            return

        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Observed Arrivals File", "", "CSV Files (*.csv)")

        if not file_path:
            return

        try:
            # Load observed arrivals file
            self.statusBar.showMessage(f"Loading observed arrivals from {file_path}...")
            observed_df = pd.read_csv(file_path, parse_dates=['time', 'arrival_time', 'observed_arrival_time'])

            # Clear existing observed arrivals dictionary to load from file
            self.observed_arrivals = {}

            # Process each event in the observed data
            for event_time in observed_df['time'].unique():
                event_key = str(event_time)
                self.observed_arrivals[event_key] = {}

                # Get event data
                event_data = observed_df[observed_df['time'] == event_time]
                first_arrival = event_data['arrival_time'].min()

                # Ensure timezone consistency
                if first_arrival.tzinfo is not None:
                    reference_time = (first_arrival - timedelta(minutes=10)).replace(tzinfo=None)
                else:
                    reference_time = first_arrival - timedelta(minutes=10)

                # Process each phase for this event
                for phase in event_data['phase'].unique():
                    phase_data = event_data[event_data['phase'] == phase]

                    if 'observed_arrival_time' in phase_data.columns and not phase_data[
                        'observed_arrival_time'].isna().all():
                        observed_time = phase_data['observed_arrival_time'].iloc[0]

                        # Convert to relative time in seconds from reference
                        if observed_time.tzinfo is not None:
                            observed_time = observed_time.replace(tzinfo=None)

                        rel_time = (observed_time - reference_time).total_seconds()
                        self.observed_arrivals[event_key][phase] = rel_time

            # Update the catalogue with the observed times
            for event_key, phases in self.observed_arrivals.items():
                try:
                    event_time = pd.to_datetime(event_key)
                    event_mask = self.catalogue_df['time'] == event_time

                    for phase, rel_time in phases.items():
                        phase_mask = (event_mask) & (self.catalogue_df['phase'] == phase)

                        if phase_mask.any():
                            # Find the reference time for this event
                            first_arrival = self.catalogue_df.loc[event_mask, 'arrival_time'].min()

                            # Ensure timezone consistency
                            if first_arrival.tzinfo is not None:
                                reference_time = (first_arrival - timedelta(minutes=10)).replace(tzinfo=None)
                            else:
                                reference_time = first_arrival - timedelta(minutes=10)

                            # Calculate absolute time
                            new_absolute_time = reference_time + timedelta(seconds=rel_time)

                            # If original time had timezone info, add it back
                            if first_arrival.tzinfo is not None:
                                new_absolute_time = new_absolute_time.replace(tzinfo=first_arrival.tzinfo)

                            # Update the DataFrame
                            self.catalogue_df.loc[phase_mask, 'observed_arrival_time'] = new_absolute_time
                except Exception as e:
                    self.statusBar.showMessage(f"Warning: Could not process event {event_key}: {str(e)}")

            # Update current event's markers
            self.update_plots()
            self.statusBar.showMessage(f"Loaded observed arrivals from {file_path}.")

        except Exception as e:
            self.statusBar.showMessage(f"Error loading observed arrivals: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load observed arrivals: {str(e)}")

    def save_observed_arrivals(self):
        """Save observed arrival times to CSV file"""
        if self.catalogue_df is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No data to save.")
            return

        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save Observed Arrivals", "", "CSV Files (*.csv)")

        if not file_path:
            return

        try:
            # Create a copy of the catalogue to avoid modifying the original during the save process
            save_df = self.catalogue_df.copy()

            # Ensure observed_arrival_time column exists
            if 'observed_arrival_time' not in save_df.columns:
                save_df['observed_arrival_time'] = save_df['arrival_time']

            # Update catalogue with observed arrival times, but ONLY for phases that were manually adjusted
            for event_key, phases in self.observed_arrivals.items():
                try:
                    # Convert string key back to datetime
                    event_time = pd.to_datetime(event_key)

                    # Find all rows for this event
                    event_mask = save_df['time'] == event_time

                    for phase, rel_time in phases.items():
                        # Find rows for this phase in this event
                        phase_mask = (event_mask) & (save_df['phase'] == phase)

                        if not phase_mask.any():
                            continue

                        # Get the predicted time for this phase
                        phase_arrival_time = save_df.loc[phase_mask, 'arrival_time'].iloc[0]

                        # Find the reference time for this event
                        first_arrival = save_df.loc[event_mask, 'arrival_time'].min()

                        # Ensure timezone consistency
                        if first_arrival.tzinfo is not None:
                            reference_time = (first_arrival - timedelta(minutes=10)).replace(tzinfo=None)
                        else:
                            reference_time = first_arrival - timedelta(minutes=10)

                        # Calculate the predicted time in seconds from reference
                        if phase_arrival_time.tzinfo is not None:
                            phase_arrival_time = phase_arrival_time.replace(tzinfo=None)
                        predicted_time = (phase_arrival_time - reference_time).total_seconds()

                        # Only update if the observed time differs from the predicted time
                        # This means the cursor was actually moved by the user
                        if abs(rel_time - predicted_time) > 0.01:  # Small threshold to account for floating point errors
                            # Calculate absolute time
                            new_absolute_time = reference_time + timedelta(seconds=rel_time)

                            # If original time had timezone info, add it back
                            if first_arrival.tzinfo is not None:
                                new_absolute_time = new_absolute_time.replace(tzinfo=first_arrival.tzinfo)

                            # Update the DataFrame only for this specific phase
                            save_df.loc[phase_mask, 'observed_arrival_time'] = new_absolute_time
                except Exception as e:
                    self.statusBar.showMessage(f"Warning: Could not process event {event_key}: {str(e)}")

            # Save to CSV
            save_df.to_csv(file_path, index=False)
            self.statusBar.showMessage(f"Saved observed arrivals to {file_path}.")

        except Exception as e:
            self.statusBar.showMessage(f"Error saving observed arrivals: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save observed arrivals: {str(e)}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = SeismicArrivalPicker()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

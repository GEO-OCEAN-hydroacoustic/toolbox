import datetime
import sys
import os
import pandas as pd
import numpy as np
import scipy.io as sio
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QListWidget, QPushButton, QLabel, QFileDialog, QSplitter,
                             QMessageBox, QComboBox, QGroupBox)
from PyQt5.QtCore import Qt, QSettings
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib.patches as patches
from src.utils.data_reading.sound_data.sound_file import DatFile
from src.utils.data_reading.sound_data.sound_file_manager import make_manager


class PhasePickerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Seismic Phase Picker")
        self.setGeometry(100, 100, 1400, 900)

        # Data state
        self.file_map = []
        self.current_file_idx = 0
        self.current_picks = {}
        self.selected_phase = None
        self.selected_theoretical_idx = None
        self.zoom_span = None
        self.marked_arrivals = []  # Store visual markers
        self.signal_line = None
        self.theoretical_lines = []

        # Settings
        self.settings = QSettings("SeismicTools", "PhasePicker")

        # Create UI
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Create splitter for adjustable panels
        self.splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(self.splitter)

        # Plot area
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.plot_layout.addWidget(self.canvas)

        # Navigation toolbar with zoom and pan
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout.addWidget(self.toolbar)

        # Control panel
        self.control_widget = QWidget()
        self.control_layout = QVBoxLayout(self.control_widget)

        # File navigation section
        self.file_group = QGroupBox("File Navigation")
        self.file_layout = QVBoxLayout()
        self.file_label = QLabel("Current File: None")
        self.file_layout.addWidget(self.file_label)

        # File navigation buttons
        self.nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.skip_button = QPushButton("Skip")
        self.next_button = QPushButton("Next")
        self.nav_layout.addWidget(self.prev_button)
        self.nav_layout.addWidget(self.skip_button)
        self.nav_layout.addWidget(self.next_button)
        self.file_layout.addLayout(self.nav_layout)

        # Load data button
        self.load_button = QPushButton("Load Dataset")
        self.file_layout.addWidget(self.load_button)
        self.file_group.setLayout(self.file_layout)
        self.control_layout.addWidget(self.file_group)

        # Phase selection section
        self.phase_group = QGroupBox("Phase Selection")
        self.phase_layout = QVBoxLayout()
        self.phase_label = QLabel("Theoretical Phases:")
        self.phase_list = QListWidget()
        self.phase_layout.addWidget(self.phase_label)
        self.phase_layout.addWidget(self.phase_list)
        self.phase_group.setLayout(self.phase_layout)
        self.control_layout.addWidget(self.phase_group)

        # Display options section
        self.display_group = QGroupBox("Display Options")
        self.display_layout = QVBoxLayout()
        self.amplitude_label = QLabel("Amplitude Scale:")
        self.amplitude_combo = QComboBox()
        self.amplitude_combo.addItems(["1x", "2x", "5x", "10x"])
        self.display_layout.addWidget(self.amplitude_label)
        self.display_layout.addWidget(self.amplitude_combo)
        self.display_group.setLayout(self.display_layout)
        self.control_layout.addWidget(self.display_group)

        # Actions section
        self.actions_group = QGroupBox("Actions")
        self.actions_layout = QVBoxLayout()
        self.clear_button = QPushButton("Clear Current Pick")
        self.save_button = QPushButton("Save & Next")
        self.actions_layout.addWidget(self.clear_button)
        self.actions_layout.addWidget(self.save_button)
        self.actions_group.setLayout(self.actions_layout)
        self.control_layout.addWidget(self.actions_group)

        # Status section
        self.status_label = QLabel("Ready")
        self.control_layout.addWidget(self.status_label)

        # Add stretch to push everything up
        self.control_layout.addStretch(1)

        # Add widgets to splitter
        self.splitter.addWidget(self.plot_widget)
        self.splitter.addWidget(self.control_widget)
        self.splitter.setSizes([800, 400])  # Initial sizes

        # Connect signals
        self.connect_signals()

        # Setup interactive elements
        self.setup_interactive_elements()

        # Restore settings
        self.restore_settings()

    def connect_signals(self):
        """Connect all UI signals"""
        self.prev_button.clicked.connect(self.previous_file)
        self.next_button.clicked.connect(self.next_file)
        self.skip_button.clicked.connect(self.next_file)
        self.save_button.clicked.connect(self.save_picks)
        self.load_button.clicked.connect(self.open_dataset)
        self.clear_button.clicked.connect(self.clear_current_pick)
        self.phase_list.itemClicked.connect(self.select_phase)
        self.amplitude_combo.currentIndexChanged.connect(self.update_amplitude_scale)
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)

    def setup_interactive_elements(self):
        """Setup interactive matplotlib elements"""
        # Zoom span selector
        self.span_selector = SpanSelector(
            self.ax, self.on_span_select, 'horizontal',
            useblit=True, props=dict(alpha=0.3, facecolor='blue'),
            interactive=True, drag_from_anywhere=True
        )

    def restore_settings(self):
        """Restore application settings"""
        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))

        # Fixed: Convert QSettings string values to integers
        if self.settings.contains("splitter_sizes"):
            sizes_variant = self.settings.value("splitter_sizes")
            # Convert to list of integers regardless of stored type
            if isinstance(sizes_variant, list):
                sizes = [int(size) for size in sizes_variant]
                self.splitter.setSizes(sizes)

    def closeEvent(self, event):
        """Save settings when closing"""
        self.settings.setValue("geometry", self.saveGeometry())
        # Store splitter sizes as a list of integers
        self.settings.setValue("splitter_sizes", self.splitter.sizes())
        super().closeEvent(event)

    def open_dataset(self):
        """Open dialog to select MAT file and data directory"""
        mat_path, _ = QFileDialog.getOpenFileName(
            self, "Select MAT File", "", "MAT Files (*.mat)"
        )
        if not mat_path:
            return

        data_dir = QFileDialog.getExistingDirectory(
            self, "Select Data Directory", ""
        )
        if not data_dir:
            return

        self.load_data(mat_path, data_dir)

    def load_data(self, mat_path, data_dir):
        """Load MAT file and find matching files using SoundFilesManager"""
        try:
            # Load the MAT file
            mat_data = sio.loadmat(mat_path)

            # Create a DataFrame from the events list with appropriate column names
            columns = ['distance', 'depth', 'phase', 'travel_time', 'year', 'month',
                       'day', 'h', 'm', 's', 'mag', 'e1', 'e2', 'e3', 'associated_file']

            # Parse the data into a structured DataFrame
            data_rows = []
            for line in mat_data['events_list'].flatten():
                # Convert from numpy string to regular string and split
                if hasattr(line, 'item'):
                    parts = line.item().split()
                else:
                    parts = str(line).split()

                if len(parts) >= len(columns):  # Ensure we have enough data
                    data_rows.append(parts[:len(columns)])

            # Create DataFrame with proper columns
            df = pd.DataFrame(data_rows, columns=columns)

            # Convert columns to appropriate data types
            numeric_cols = ['distance', 'depth', 'travel_time', 'year', 'month',
                            'day', 'h', 'm', 's', 'mag', 'e1', 'e2', 'e3']

            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add datetime column for better time handling
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'h', 'm', 's']], errors='coerce')

            # Initialize SoundFilesManager for the data directory

            try:
                # Try to create the appropriate manager for the data directory
                self.files_manager = make_manager(data_dir, kwargs={"sensitivity": -163.5})
                if not self.files_manager:
                    raise Exception(f"No compatible files found in {data_dir}")

                self.status_label.setText(f"Using {self.files_manager.__class__.__name__} for {data_dir}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize FileManager: {str(e)}")
                return

            # Initialize the file map
            self.file_map = []

            # Group events by datetime to handle multiple phases per event
            for event_time, group in df.groupby('datetime'):
                try:
                    # Skip if datetime conversion failed
                    if pd.isna(event_time):
                        continue

                    # Find the correct file for this event using SoundFilesManager
                    # Check if the event time is within our dataset bounds
                    if event_time < self.files_manager.dataset_start or event_time > self.files_manager.dataset_end:
                        print(
                            f"Warning: Event time {event_time} outside dataset bounds ({self.files_manager.dataset_start} to {self.files_manager.dataset_end})")
                        continue

                    # Find the file containing this event time
                    file_idx = self.files_manager._find_file(event_time)
                    if file_idx is None:
                        print(f"Warning: No file found for event at {event_time}")
                        continue

                    file = self.files_manager.files[file_idx]
                    file_path = file.path

                    # Extract phase information
                    phases = group['phase'].tolist()
                    theoretical_times = group['travel_time'].tolist()

                    # Calculate seconds from file start to event
                    event_offset = (event_time - file.header["start_date"]).total_seconds()

                    self.file_map.append({
                        'path': file_path,
                        'theoretical_times': [event_offset + tt for tt in theoretical_times],  # Adjust to file start
                        'phases': phases,
                        'event_offset': event_offset,  # Store offset for reference
                        'metadata': {
                            'distances': group['distance'].tolist(),
                            'depths': group['depth'].tolist(),
                            'magnitudes': group['mag'].tolist(),
                            'datetimes': group['datetime'].tolist(),
                            'original_file': group['associated_file'].tolist()  # Keep original for reference
                        }
                    })
                except Exception as e:
                    print(f"Error processing event at {event_time}: {e}")

            self.status_label.setText(f"Loaded {len(self.file_map)} events")
            self.current_file_idx = 0
            if self.file_map:
                self.load_current_file()
            else:
                QMessageBox.warning(self, "Warning", "No valid events found in the dataset")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_current_file(self):
        """Load and display current file data with properly aligned theoretical times"""
        if not self.file_map:
            self.status_label.setText("No files loaded")
            return

        if self.current_file_idx >= len(self.file_map):
            self.show_final_message()
            return

        file_info = self.file_map[self.current_file_idx]
        self.current_picks = {}
        self.marked_arrivals = []
        self.theoretical_lines = []  # Reset theoretical lines list

        try:
            # Load data using DatFile
            df = DatFile(file_info['path'])

            # Get recording start time from header if available
            recording_start_time = None
            if hasattr(df, 'header') and 'start_date' in df.header:
                recording_start_time = df.header['start_date']

            # Create time array for the seismogram
            time = np.arange(len(df.data)) / df.header['sampling_frequency']
            signal = df.data

            self.ax.clear()
            self.signal_line, = self.ax.plot(time, signal, 'k', label='Signal', alpha=0.7)

            # Calculate and plot theoretical arrival times relative to recording start
            adjusted_theoretical_times = []

            # If we have recording start time and event origin times, calculate relative arrivals
            if 'metadata' in file_info and 'datetimes' in file_info['metadata'] and recording_start_time:
                for i, (phase_time, phase, event_datetime) in enumerate(
                        zip(file_info['theoretical_times'], file_info['phases'], file_info['metadata']['datetimes'])):

                    # Calculate absolute arrival time: event origin time + travel time
                    if isinstance(event_datetime, pd.Timestamp):
                        # Convert recording_start_time to pandas Timestamp if it's a datetime object
                        if isinstance(recording_start_time, datetime.datetime):
                            recording_start_time = pd.Timestamp(recording_start_time)

                        # Calculate absolute theoretical arrival time (event time + travel time)
                        arrival_time = event_datetime + pd.Timedelta(seconds=phase_time)

                        # Calculate seconds since recording start
                        if isinstance(recording_start_time, pd.Timestamp):
                            seconds_since_start = (arrival_time - recording_start_time).total_seconds()
                            adjusted_theoretical_times.append(seconds_since_start)
                        else:
                            # Fallback if recording_start_time isn't valid
                            adjusted_theoretical_times.append(phase_time)
                    else:
                        # Fallback to using travel time directly if datetime conversion failed
                        adjusted_theoretical_times.append(phase_time)

            # Plot the adjusted theoretical times
            for i, (t, phase) in enumerate(zip(adjusted_theoretical_times, file_info['phases'])):
                # Skip if the theoretical time is outside the signal timeframe
                if t < 0 or t > time[-1]:
                    continue

                # Get metadata for this phase if available
                metadata = {}
                if 'metadata' in file_info:
                    for key, values in file_info['metadata'].items():
                        if i < len(values):
                            metadata[key] = values[i]

                # Format enhanced label
                distance = metadata.get('distances', 'N/A')
                magnitude = metadata.get('magnitudes', 'N/A')
                label_text = f"{phase}"

                # Save the adjusted theoretical time back to file_info for later use
                file_info['theoretical_times'][i] = t

                # Draw line for theoretical time
                line = self.ax.axvline(t, color='r', linestyle='--', alpha=1,
                                       label=f'Theoretical ({phase})' if i == 0 else '')
                self.theoretical_lines.append(line)

                # Add text label with phase name
                self.ax.text(t, self.ax.get_ylim()[1] * 0.95, label_text,
                             rotation=90, verticalalignment='top', color='g')

            # Set initial view to focus on theoretical arrivals
            if adjusted_theoretical_times:
                valid_times = [t for t in adjusted_theoretical_times if 0 <= t <= time[-1]]
                if valid_times:
                    min_time = max(0, min(valid_times) - 90)
                    max_time = min(time[-1], max(valid_times) + 90)
                    self.ax.set_xlim(min_time, max_time)

            # Format title with additional metadata if available
            title = os.path.basename(file_info['path'])
            if 'metadata' in file_info and 'datetimes' in file_info['metadata'] and file_info['metadata']['datetimes']:
                first_event = file_info['metadata']['datetimes'][0]
                if isinstance(first_event, pd.Timestamp):
                    date_str = first_event.strftime("%Y-%m-%d %H:%M:%S")
                    title = f"{title} - Event: {date_str}"

                    # Add recording start time if available
                    if recording_start_time and isinstance(recording_start_time, pd.Timestamp):
                        rec_str = recording_start_time.strftime("%Y-%m-%d %H:%M:%S")
                        title += f" - Rec: {rec_str}"

            self.ax.set_title(title)
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude (V)")
            self.ax.legend()
            self.ax.grid(True)

            # Update phase list with enhanced information
            self.phase_list.clear()
            for i, phase in enumerate(file_info['phases']):
                # Add metadata to list items if available
                if 'metadata' in file_info and i < len(file_info['phases']):
                    distance = file_info['metadata']['distances'][i] if i < len(
                        file_info['metadata']['distances']) else 'N/A'
                    # Show adjusted time in the list
                    time_display = f"{adjusted_theoretical_times[i]:.2f}s" if i < len(
                        adjusted_theoretical_times) else 'N/A'
                    self.phase_list.addItem(f"{phase} ({time_display}, Dist: {distance}°)")
                else:
                    self.phase_list.addItem(phase)

            # Update file info display
            self.file_label.setText(
                f"File {self.current_file_idx + 1}/{len(self.file_map)}: {os.path.basename(file_info['path'])}")

            # Apply current amplitude scale
            self.update_amplitude_scale()

            self.canvas.draw()
            self.status_label.setText("Ready to pick phases")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.warning(self, "Error", f"Could not load file: {str(e)}")
            # Print full traceback for debugging
            import traceback
            traceback.print_exc()
            self.next_file()

    def select_phase(self, item):
        """Handle phase selection from list"""
        self.selected_phase = item.text()
        phase_idx = self.phase_list.currentRow()

        # Highlight the selected theoretical time
        self.selected_theoretical_idx = phase_idx

        # Safeguard against empty theoretical_lines
        if not self.theoretical_lines:
            return

        # Update line colors
        for i, line in enumerate(self.theoretical_lines):
            if i == phase_idx:
                line.set_color('r')
                line.set_alpha(1.0)
                line.set_linewidth(2)
            else:
                line.set_color('g')
                line.set_alpha(0.7)
                line.set_linewidth(1)

        # Center view on selected phase if available
        file_info = self.file_map[self.current_file_idx]
        if 0 <= phase_idx < len(file_info['theoretical_times']):
            t_time = file_info['theoretical_times'][phase_idx]
            current_xlim = self.ax.get_xlim()
            window_width = current_xlim[1] - current_xlim[0]
            self.ax.set_xlim(t_time - window_width / 2, t_time + window_width / 2)

        self.canvas.draw()
        self.status_label.setText(f"Selected phase: {self.selected_phase}")

    def on_plot_click(self, event):
        """Handle plot clicks to mark arrival times"""
        if event.inaxes != self.ax or not self.selected_phase:
            return

        picked_time = event.xdata

        # Store the pick
        self.current_picks[self.selected_phase] = picked_time

        # Clear previous markers for this phase
        self.clear_phase_markers(self.selected_phase)

        # Add visual marker
        marker = self.ax.axvline(picked_time, color='r', linestyle='-', linewidth=1.5)
        marker_text = self.ax.text(picked_time, event.ydata, f"{self.selected_phase} Pick",
                                   rotation=90, color='r', verticalalignment='bottom')

        self.marked_arrivals.append((self.selected_phase, marker, marker_text))

        self.canvas.draw()
        self.status_label.setText(f"Marked {self.selected_phase} at {picked_time:.3f}s")

    def clear_phase_markers(self, phase):
        """Clear markers for a specific phase"""
        # Find and remove existing markers for this phase
        to_remove = []
        for i, (p, marker, text) in enumerate(self.marked_arrivals):
            if p == phase:
                marker.remove()
                text.remove()
                to_remove.append(i)

        # Update the list without the removed items
        self.marked_arrivals = [m for i, m in enumerate(self.marked_arrivals) if i not in to_remove]

    def clear_current_pick(self):
        """Clear current phase pick"""
        if not self.selected_phase:
            return

        if self.selected_phase in self.current_picks:
            del self.current_picks[self.selected_phase]

        self.clear_phase_markers(self.selected_phase)
        self.canvas.draw()
        self.status_label.setText(f"Cleared {self.selected_phase} pick")

    def on_span_select(self, xmin, xmax):
        """Handle span selection for zooming"""
        self.ax.set_xlim(xmin, xmax)
        self.canvas.draw()

    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'n':
            self.next_file()
        elif event.key == 'p':
            self.previous_file()
        elif event.key == 's':
            self.save_picks()
        elif event.key == 'escape':
            self.clear_current_pick()
        elif event.key == 'c':
            self.toolbar.home()  # Reset zoom
            self.canvas.draw()

    def update_amplitude_scale(self):
        """Update amplitude scaling"""
        if not hasattr(self, 'signal_line') or self.signal_line is None:
            return

        try:
            scale_text = self.amplitude_combo.currentText()
            scale = float(scale_text.replace('x', ''))

            # Get current data
            ydata = self.signal_line.get_ydata()

            # Check for valid data
            if ydata is None or len(ydata) == 0:
                return

            # Calculate current amplitude for centering
            current_ylim = self.ax.get_ylim()
            center = (current_ylim[0] + current_ylim[1]) / 2

            # Calculate new limits based on data range
            data_range = np.max(ydata) - np.min(ydata)
            if data_range == 0:  # Avoid division by zero
                data_range = 1e-10

            half_range = data_range / 2 / scale

            # Set new limits centered on the same point
            self.ax.set_ylim(center - half_range, center + half_range)
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating amplitude scale: {e}")

    def save_picks(self):
        """Save results with enhanced metadata and move to next file"""
        if not self.file_map or self.current_file_idx >= len(self.file_map):
            return

        file_info = self.file_map[self.current_file_idx]

        # Prepare data for saving with enhanced metadata
        results = {
            'filename': os.path.basename(file_info['path']),
            'path': file_info['path'],
            'pick_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add metadata if available
        if 'metadata' in file_info:
            # Add first event's metadata
            for key in file_info['metadata']:
                if file_info['metadata'][key] and len(file_info['metadata'][key]) > 0:
                    if key == 'datetimes' and isinstance(file_info['metadata'][key][0], pd.Timestamp):
                        results['event_datetime'] = file_info['metadata'][key][0].strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        results[f'event_{key[:-1]}'] = file_info['metadata'][key][0]  # Remove 's' from plural key

        # Add theoretical times
        for i, (phase, time) in enumerate(zip(file_info['phases'], file_info['theoretical_times'])):
            results[f'theoretical_{phase}'] = time

        # Add picked times
        for phase, time in self.current_picks.items():
            results[f'picked_{phase}'] = time

            # Calculate residual if theoretical time exists
            if phase in file_info['phases']:
                phase_idx = file_info['phases'].index(phase)
                if phase_idx < len(file_info['theoretical_times']):
                    theoretical = file_info['theoretical_times'][phase_idx]
                    residual = time - theoretical
                    results[f'residual_{phase}'] = residual

        # Create pandas DataFrame
        df = pd.DataFrame([results])

        try:
            # Check if file exists to determine whether to write header
            output_file = 'phase_picks.csv'
            file_exists = os.path.isfile(output_file)

            # Append to CSV
            df.to_csv(output_file, mode='a', header=not file_exists, index=False)

            self.status_label.setText(f"Saved picks for {os.path.basename(file_info['path'])}")

            # Also show how many files remain
            remaining = len(self.file_map) - (self.current_file_idx + 1)
            if remaining > 0:
                self.status_label.setText(f"Saved picks. {remaining} files remaining.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save picks: {str(e)}")
            import traceback
            traceback.print_exc()
            return

        # Move to next file
        self.next_file()

    def previous_file(self):
        """Go to previous file"""
        if not self.file_map:
            return

        self.current_file_idx -= 1
        if self.current_file_idx < 0:
            self.current_file_idx = 0
            self.status_label.setText("Already at first file")
        else:
            self.load_current_file()

    def next_file(self):
        """Go to next file"""
        if not self.file_map:
            return

        self.current_file_idx += 1
        if self.current_file_idx < len(self.file_map):
            self.load_current_file()
        else:
            self.show_final_message()

    def show_final_message(self):
        """Show completion message"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Processing Complete!\nResults saved to phase_picks.csv",
                     ha='center', va='center', fontsize=14)
        self.canvas.draw()

        # Disable controls
        self.control_widget.setEnabled(False)

        # Show message box
        QMessageBox.information(self, "Complete", "All files processed. Results saved to phase_picks.csv")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhasePickerApp()
    window.show()
    sys.exit(app.exec_())
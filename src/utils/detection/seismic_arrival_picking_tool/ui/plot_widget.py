import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal
from src.utils.detection.seismic_arrival_picking_tool.signal_processing.audio_processing import energy_plot

class SeismicPlotWidget(QWidget):
    arrival_picked = pyqtSignal(float, str, float, float)
    
    def __init__(self, data, sampling_freq, event_details):
        super().__init__()
        
        self.data = data
        self.sampling_freq = sampling_freq
        self.event_details = event_details
        
        layout = QVBoxLayout()

        # Main plot widget for seismic signal
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget, 4)  # 4/5 of the space

        # Energy plot widget
        self.energy_plot_widget = pg.PlotWidget()
        layout.addWidget(self.energy_plot_widget, 1)  # 1/5 of the space

        self.setLayout(layout)
        
        self.fixed_markers = {}
        self.movable_markers = {}

        self._setup_plots()

    def _setup_plots(self):
        # Time array (convert samples to seconds)
        time_array = np.arange(len(self.data)) / self.sampling_freq

        # Plot seismic signal
        self.plot_widget.plot(time_array, self.data)

        # First arrival for time reference
        first_arrival = self.event_details['predicted_arrival_time'].min()

        # Add predicted and existing observed arrival markers
        for _, row in self.event_details.iterrows():
            # Calculate relative time for plotting
            predicted_time = row['predicted_arrival_time']
            relative_predicted_time = (predicted_time - first_arrival).total_seconds()+600
            
            # Fixed marker (predicted time)
            fixed_marker = pg.InfiniteLine(
                pos=relative_predicted_time,
                angle=90,
                movable=False,
                pen=pg.mkPen('r', width=2),
                label=f"Predicted {row['phase']}"
            )
            self.plot_widget.addItem(fixed_marker)
            self.fixed_markers[row['phase']] = fixed_marker

            # Movable marker (initially at predicted time)
            movable_marker = pg.InfiniteLine(
                pos=relative_predicted_time,
                angle=90, 
                movable=True,
                label=f"Observed {row['phase']}"
            )
            self.plot_widget.addItem(movable_marker)
            self.movable_markers[row['phase']] = movable_marker

            # Connect movement signal
            movable_marker.sigPositionChanged.connect(
                lambda _, phase=row['phase']: self._on_marker_moved(_, phase)
            )

        # Configure main plot labels
        self.plot_widget.setLabel('bottom', 'Time', 'seconds')
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setTitle('Seismic Signal with Arrival Markers')

        # Calculate and plot energy
        event_indices, energy, threshold = energy_plot(
            self.data,
            self.sampling_freq
        )

        # Time array for energy
        energy_time_array = np.arange(len(energy)) * (self.sampling_freq / 2) / self.sampling_freq

        # Plot energy
        self.energy_plot_widget.plot(energy_time_array, energy, pen='b')
        self.energy_plot_widget.plot(
            [0, energy_time_array[-1]],
            [threshold, threshold],
            pen='r',
            label='Threshold'
        )



        # Configure energy plot labels
        self.energy_plot_widget.setLabel('bottom', 'Time', 'seconds')
        self.energy_plot_widget.setLabel('left', 'Energy')
        self.energy_plot_widget.setTitle('Signal Energy')

        # Link x-axes of both plots
        self.plot_widget.setXLink(self.energy_plot_widget)

    def _on_marker_moved(self, line, phase):
        # Get first arrival time as reference
        first_arrival = self.event_details['predicted_arrival_time'].min()

        # Get current marker positions
        predicted_marker = self.fixed_markers[phase]
        movable_marker = self.movable_markers[phase]

        # Emit the new picked arrival time
        self.arrival_picked.emit(
            movable_marker.pos()[0],
            phase,
            predicted_marker.pos()[0],
            movable_marker.pos()[0]
        )
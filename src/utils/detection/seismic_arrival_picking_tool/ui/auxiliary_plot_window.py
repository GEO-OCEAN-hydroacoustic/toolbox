import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt

from src.utils.detection.seismic_arrival_picking_tool.signal_processing.audio_processing import (
    dehaze_audio, 
    apply_butter_bandpass, 
    energy_plot
)

class AdditionalProcessingWidget(QWidget):
    def __init__(self, data, sampling_freq, main_plot_widget=None):
        super().__init__()

        self.original_data = data
        self.sampling_freq = sampling_freq
        self.main_plot_widget = main_plot_widget
        
        # Prepare layout
        main_layout = QVBoxLayout()
        
        # Processing controls
        processing_layout = QHBoxLayout()

        # Bandpass filter controls
        self.lowcut_combo = QComboBox()
        self.lowcut_combo.addItems(['1', '2', '5', '10'])
        self.lowcut_combo.setCurrentText('1')
        processing_layout.addWidget(QLabel("Low Cut (Hz):"))
        processing_layout.addWidget(self.lowcut_combo)

        self.highcut_combo = QComboBox()
        self.highcut_combo.addItems(['10', '20', '50', '100'])
        self.highcut_combo.setCurrentText('50')
        processing_layout.addWidget(QLabel("High Cut (Hz):"))
        processing_layout.addWidget(self.highcut_combo)

        # Dehaze checkbox
        self.dehaze_check = QCheckBox("Dehaze")
        processing_layout.addWidget(self.dehaze_check)

        # Process button
        process_button = QPushButton("Process")
        process_button.clicked.connect(self.process_data)
        processing_layout.addWidget(process_button)
        
        main_layout.addLayout(processing_layout)

        # Plot widgets
        plot_layout = QHBoxLayout()
        
        # Raw signal plot
        self.raw_plot_widget = pg.PlotWidget()
        self.raw_plot_widget.setLabel('bottom', 'Time', 'seconds')
        self.raw_plot_widget.setLabel('left', 'Amplitude')
        self.raw_plot_widget.setTitle('Raw Signal')
        plot_layout.addWidget(self.raw_plot_widget)
        
        # Processed signal plot
        self.processed_plot_widget = pg.PlotWidget()
        self.processed_plot_widget.setLabel('bottom', 'Time', 'seconds')
        self.processed_plot_widget.setLabel('left', 'Amplitude')
        self.processed_plot_widget.setTitle('Processed Signal')
        plot_layout.addWidget(self.processed_plot_widget)
        
        main_layout.addLayout(plot_layout)
        
        # Energy plot
        self.energy_plot_widget = pg.PlotWidget()
        self.energy_plot_widget.setLabel('bottom', 'Time', 'seconds')
        self.energy_plot_widget.setLabel('left', 'Energy')
        self.energy_plot_widget.setTitle('Signal Energy')
        main_layout.addWidget(self.energy_plot_widget)
        
        self.setLayout(main_layout)
        
        # Initial plot of raw data
        self._plot_raw_data()
        
        # Link x-axes if main plot widget is provided
        if main_plot_widget:
            self._link_x_axes()

    def _plot_raw_data(self):
        # Time array
        time_array = np.arange(len(self.original_data)) / self.sampling_freq
        
        # Plot raw signal
        self.raw_plot_widget.plot(time_array, self.original_data)
    
    def _link_x_axes(self):
        # Link x-axes of plot widgets
        self.raw_plot_widget.setXLink(self.main_plot_widget.plot_widget)
        self.processed_plot_widget.setXLink(self.main_plot_widget.plot_widget)
        self.energy_plot_widget.setXLink(self.main_plot_widget.plot_widget)

    def process_data(self):
        # Get filter parameters
        lowcut = float(self.lowcut_combo.currentText())
        highcut = float(self.highcut_combo.currentText())
        
        # Copy of original data
        processed_data = self.original_data.copy()
        
        # Apply bandpass filter
        processed_data = apply_butter_bandpass(
            processed_data,
            self.sampling_freq,
            lowcut,
            highcut
        )

        # Optional dehaze
        if self.dehaze_check.isChecked():
            processed_data = dehaze_audio(
                processed_data,
                self.sampling_freq
            )

        # Time array
        time_array = np.arange(len(processed_data)) / self.sampling_freq

        # Clear previous plots
        self.processed_plot_widget.clear()
        self.energy_plot_widget.clear()
        
        # Plot processed signal
        self.processed_plot_widget.plot(time_array, processed_data)
        
        # Calculate and plot energy
        event_indices, energy, threshold = energy_plot(
            processed_data,
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

        # Highlight event regions
        for idx in event_indices:
            start_time = energy_time_array[idx]
            self.energy_plot_widget.addItem(
                pg.InfiniteLine(
                    pos=start_time,
                    angle=90,
                    pen=pg.mkPen('g', style=pg.QtCore.Qt.DashLine)
                )
            )
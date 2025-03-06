import os
import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QWidget, QMessageBox, QFileDialog, QComboBox, QTabWidget
)
from PyQt5.QtCore import Qt

from src.utils.detection.seismic_arrival_picking_tool.data_manager import SeismicDataManager
from src.utils.detection.seismic_arrival_picking_tool.ui.plot_widget import SeismicPlotWidget
from src.utils.detection.seismic_arrival_picking_tool.ui.auxiliary_plot_window import AdditionalProcessingWidget

from src.utils.data_reading.sound_data.sound_file_manager import DatFilesManager

class SeismicArrivalPickingTool(QMainWindow):
    def __init__(self, catalogue_path, dat_file_path):
        super().__init__()
        
        self.setWindowTitle("Seismic Arrival Picking Tool")
        self.resize(1600, 1000)
        
        # Initialize data managers
        self.data_manager = SeismicDataManager(catalogue_path)
        self.dat_file_manager = DatFilesManager(dat_file_path)
        
        # Get unique events
        self.unique_events = self.data_manager.get_unique_events()
        self.current_event_index = 0

        # Setup main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Tabbed interface
        self.tab_widget = QTabWidget()

        # Arrival Picking Tab
        arrival_picking_widget = QWidget()
        arrival_picking_layout = QVBoxLayout()

        # Event navigation and info section
        nav_layout = QHBoxLayout()
        
        # Event info label
        self.event_info_label = QLabel("Event: Loading...")
        nav_layout.addWidget(self.event_info_label)

        # Phase selection dropdown
        self.phase_combo = QComboBox()
        nav_layout.addWidget(QLabel("Select Phase:"))
        nav_layout.addWidget(self.phase_combo)

        # Navigation buttons
        self.prev_button = QPushButton("Previous Event")
        self.prev_button.clicked.connect(self.load_previous_event)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next Event")
        self.next_button.clicked.connect(self.load_next_event)
        nav_layout.addWidget(self.next_button)
        
        arrival_picking_layout.addLayout(nav_layout)

        # Plot area
        self.plot_area = QWidget()
        plot_layout = QVBoxLayout()
        self.plot_area.setLayout(plot_layout)
        arrival_picking_layout.addWidget(self.plot_area)

        # Save progress button
        save_button = QPushButton("Save Progress")
        save_button.clicked.connect(self.save_progress)
        arrival_picking_layout.addWidget(save_button)

        arrival_picking_widget.setLayout(arrival_picking_layout)

        # Additional Processing Tab
        self.additional_processing_widget = None
        
        # Add tabs
        self.tab_widget.addTab(arrival_picking_widget, "Arrival Picking")

        main_layout.addWidget(self.tab_widget)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Load first event
        self.load_event(self.current_event_index)

    def load_event(self, index):
        # Clear previous plot
        for i in reversed(range(self.plot_area.layout().count())): 
            self.plot_area.layout().itemAt(i).widget().setParent(None)
        
        # Get event time
        event_time = self.unique_events[index]
        
        # Get event details and seismic data
        event_details = self.data_manager.get_event_details(event_time)
        data, sampling_freq, _, _, file_number = self.data_manager.get_seismic_data(event_time, self.dat_file_manager)

        # Update event info
        self.event_info_label.setText(
            f"Event: {event_time} | Magnitude: {event_details['mag'].iloc[0]:.2f} | "
            f"Location: {event_details['place'].iloc[0]} | DATFile: {file_number}"
        )

        # Populate phase dropdown
        self.phase_combo.clear()
        phases = event_details['phase'].unique()
        self.phase_combo.addItems(phases)

        # Create plot widget
        plot_widget = SeismicPlotWidget(data, sampling_freq, event_details)
        plot_widget.arrival_picked.connect(self.update_arrival_time)
        
        self.plot_area.layout().addWidget(plot_widget)

        # Update navigation buttons
        self.prev_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.unique_events) - 1)

        # Create or update additional processing tab
        if self.additional_processing_widget:
            # Remove existing tab
            self.tab_widget.removeTab(1)

        # Create new additional processing widget
        self.additional_processing_widget = AdditionalProcessingWidget(
            data,
            sampling_freq,
            plot_widget
        )
        self.tab_widget.addTab(self.additional_processing_widget, "Signal Processing")

    def load_previous_event(self):
        if self.current_event_index > 0:
            self.current_event_index -= 1
            self.load_event(self.current_event_index)

    def load_next_event(self):
        if self.current_event_index < len(self.unique_events) - 1:
            self.current_event_index += 1
            self.load_event(self.current_event_index)

    def update_arrival_time(self, observed_time, phase, predicted_time, final_observed_time):
        # Get current event time
        event_time = self.unique_events[self.current_event_index]
        
        # Get first arrival time as reference
        event_details = self.data_manager.get_event_details(event_time)
        first_arrival = event_details['predicted_arrival_time'].min()

        # Convert relative times to absolute times
        predicted_absolute_time = first_arrival + pd.Timedelta(seconds=predicted_time)
        observed_absolute_time = first_arrival + pd.Timedelta(seconds=final_observed_time)

        # Update arrival times
        self.data_manager.update_arrival_times(
            event_time,
            phase,
            predicted_absolute_time,
            observed_absolute_time
        )
    
    def save_progress(self):
        output_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Updated Catalogue", 
            "", 
            "CSV Files (*.csv)"
        )
        
        if output_path:
            self.data_manager.save_updated_catalogue(output_path)
            QMessageBox.information(
                self, 
                "Save Successful", 
                f"Catalogue saved to {output_path}"
            )

def main():
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Paths for the seismic data
    CATALOGUE_PATH = r'../../../../../data/ELAN_2018_M6.csv'
    DAT_FILE_PATH = r'F:\OHASISBIO\2018\2018_ELAN_raw'

    window = SeismicArrivalPickingTool(CATALOGUE_PATH, DAT_FILE_PATH)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
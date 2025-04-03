import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from GUI.windows.spectral_viewer import SpectralViewerWindow

if __name__ == "__main__":
    # main to call in order to launch the dataset exploration tool.
    app = QApplication(sys.argv)
    app.setStyleSheet(
        "QLabel{font-size: 16pt;} QDateTimeEdit{font-size: 16pt;} QPushButton{font-size: 20pt;} QDoubleSpinBox{font-size: 16pt;}")

    # dataset yaml to get more information about the stations ; the file can be empty
    datasets_yaml = "/home/rsafran/PycharmProjects/toolbox/data/recensement_stations_PY.yaml"

    tissnet_checkpoint = "/home/rsafran/PycharmProjects/toolbox/data/models/TiSSNet/torch_save"  # TiSSNet weights for demonstration purpose ; optional
    if not Path(tissnet_checkpoint).exists():
        tissnet_checkpoint = None

    window = SpectralViewerWindow(datasets_yaml, tissnet_checkpoint=tissnet_checkpoint, events_path="../../data/GUI/events.yaml")
    window.show()

    sys.exit(app.exec())
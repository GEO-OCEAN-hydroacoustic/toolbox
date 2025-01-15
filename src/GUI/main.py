import sys
from pathlib import Path
import matplotlib
from PySide6.QtWidgets import QApplication

from GUI.windows.spectral_viewer import SpectralViewerWindow

if __name__ == "__main__":
    # main to call in order to launch the dataset exploration tool.
    app = QApplication(sys.argv)
    app.setStyleSheet(
        "QLabel{font-size: 10pt;} QDateTimeEdit{font-size: 10pt;} QPushButton{font-size: 10pt;} QDoubleSpinBox{font-size: 10pt;}")
    matplotlib.rcParams.update({'font.size': 10})

    # dataset yaml to get more information about the stations ; the file can be empty
    datasets_yaml = "../../data/demo/dataset.yaml"
    datasets_yaml = "/home/plerolland/Bureau/dataset.yaml"

    tissnet_checkpoint = "../../data/models/TiSSNet/torch_save"  # TiSSNet weights for demonstration purpose ; optional
    if not Path(tissnet_checkpoint).exists():
        tissnet_checkpoint = None

    window = SpectralViewerWindow(datasets_yaml, tissnet_checkpoint=tissnet_checkpoint,
                                  events_path="../../data/GUI/events.yaml",
                                  loc_res_path="../../data/GUI/localizations.csv")
    window.show()

    sys.exit(app.exec())
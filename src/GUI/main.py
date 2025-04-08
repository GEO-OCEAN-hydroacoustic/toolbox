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
    datasets_yaml = "/home/rsafran/PycharmProjects/toolbox/data/recensement_stations_PY.yaml"

    tissnet_checkpoint = "../../data/models/TiSSNet/torch_save"  # TiSSNet weights for demonstration purpose ; optional
    tissnet_checkpoint = "/home/rsafran/PycharmProjects/toolbox/data/models/TiSSNet/torch_save"  # TiSSNet weights for demonstration purpose ; optional
    if not Path(tissnet_checkpoint).exists():
        tissnet_checkpoint = None

    # path of catalog of events to visualize, leave None if it is not needed
    events_path = "../data/GUI/events.yaml"

    # when picking events with the software, they will be saved here
    output_path = "../data/GUI/localizations.csv"

    # now define the sound model that the GUI will use
    velocity_grid_paths = [f"../data/sound_model/min-velocities_month-{i:02d}.nc" for i in range(1, 13)]
    if np.all([Path(f).is_file() for f in velocity_grid_paths]):
        print("Velocity grids found, using a velocity grid sound model.")
        sound_model = GridSoundModel(velocity_grid_paths)
    else:
        sound_model = SoundModel(sound_speed=1480)
        print(f"No velocity grid found, using a homogeneous sound model (sound speed={sound_model})")

    window = SpectralViewerWindow(datasets_csv=datasets_csv, sound_model=sound_model,
                                  tissnet_checkpoint=tissnet_checkpoint, events_path=events_path,
                                  loc_res_path=output_path)
    window.show()

    sys.exit(app.exec())
import sys
from pathlib import Path
import matplotlib
import numpy as np
from PySide6.QtWidgets import QApplication

from GUI.windows.spectral_viewer import SpectralViewerWindow
from utils.physics.sound_model.spherical_sound_model import HomogeneousSphericalSoundModel as SoundModel
from utils.physics.sound_model.spherical_sound_model import GridSphericalSoundModel as GridSoundModel

if __name__ == "__main__":
    # main to call in order to launch the dataset exploration tool.
    app = QApplication(sys.argv)
    app.setStyleSheet(
        "QLabel{font-size: 10pt;} QDateTimeEdit{font-size: 10pt;} QPushButton{font-size: 10pt;} QDoubleSpinBox{font-size: 10pt;}")
    matplotlib.rcParams.update({'font.size': 10})

    # dataset csv to get more information about the stations ; the file can be empty or the variable can be set to None
    # note : if this points to a directory, it will look at its content
    datasets_csv = "../data/demo/sound_data"

    # TiSSNet weights for demonstration purpose, leave at None if unavailable
    tissnet_checkpoint = "../data/models/TiSSNet/torch_save"
    if tissnet_checkpoint and not Path(tissnet_checkpoint).exists():
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
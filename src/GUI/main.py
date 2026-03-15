import sys
from pathlib import Path

import matplotlib
import numpy as np
from PySide6.QtWidgets import QApplication

from GUI.windows.spectral_viewer import SpectralViewerWindow
from utils.physics.sound_model.spherical_sound_model import HomogeneousSphericalSoundModel as SoundModel
from utils.physics.sound_model.spherical_sound_model import GridSphericalSoundModel as GridSoundModel

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

# Data sources
DATASETS_CSV = "../data/demo/sound_data"           # station metadata CSV, directory, or None
EVENTS_PATH = "../data/GUI/events.yaml"             # event catalog to visualize, or None
OUTPUT_PATH = "../data/GUI/localizations.csv"       # where picked events are saved

# Sound model
VELOCITY_GRID_DIR = None           # directory containing monthly velocity grids. If None, homogeneous speed is used.
DEFAULT_SOUND_SPEED = 1480         # m/s, used if no velocity grids found

# Detection model (TiSSNet)
TISSNET_CHECKPOINT = None  # path to weights, leave to None if unused

# Display
VMIN_SPECTRO = 60
VMAX_SPECTRO = 120
FONT_SIZE = 10

# ──────────────────────────────────────────────


def build_sound_model():
    """Load the velocity-grid model if available, otherwise fall back to a homogeneous model."""
    if VELOCITY_GRID_DIR:
        grid_paths = [f"{VELOCITY_GRID_DIR}/min-velocities_month-{i:02d}.nc" for i in range(1, 13)]
        if all(Path(f).is_file() for f in grid_paths):
            print("Velocity grids found, using a velocity grid sound model.")
            return GridSoundModel(grid_paths)
    print(f"No velocity grid found, using a homogeneous sound model (speed={DEFAULT_SOUND_SPEED} m/s)")
    return SoundModel(sound_speed=DEFAULT_SOUND_SPEED)


def resolve_checkpoint():
    """Return the checkpoint path if it exists, None otherwise."""
    if TISSNET_CHECKPOINT and Path(TISSNET_CHECKPOINT).exists():
        return TISSNET_CHECKPOINT
    return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(
        f"QLabel{{font-size: {FONT_SIZE}pt;}} "
        f"QDateTimeEdit{{font-size: {FONT_SIZE}pt;}} "
        f"QPushButton{{font-size: {FONT_SIZE}pt;}} "
        f"QDoubleSpinBox{{font-size: {FONT_SIZE}pt;}}"
    )
    matplotlib.rcParams.update({"font.size": FONT_SIZE})

    window = SpectralViewerWindow(
        datasets_csv=DATASETS_CSV,
        sound_model=build_sound_model(),
        tissnet_checkpoint=resolve_checkpoint(),
        events_path=EVENTS_PATH,
        loc_res_path=OUTPUT_PATH,
        vmin_spectro=VMIN_SPECTRO,
        vmax_spectro=VMAX_SPECTRO,
    )
    window.show()
    sys.exit(app.exec())
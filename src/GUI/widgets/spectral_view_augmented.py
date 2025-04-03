import numpy as np
import torch
from torchvision.transforms import Resize
from scipy.signal import find_peaks
from skimage.transform import resize

from GUI.widgets.mpl_canvas import MplCanvas
from GUI.widgets.spectral_view import SpectralView, MIN_SEGMENT_DURATION_S
from utils.physics.signal.make_spectrogram import make_spectrogram


class SpectralViewTissnet(SpectralView):
    """ Spectral view enabling, with shift+enter, to apply TiSSNet on the current spectrogram (visualization purpose).
    """
    def __init__(self, SpectralViewer, station, date=None, delta_view_s=MIN_SEGMENT_DURATION_S, vmin_spectro=60, vmax_spectro=120, *args, **kwargs):
        """ Initialize the SpectralView. See SpectralView documentation.
        """
        super().__init__(SpectralViewer, station, date, delta_view_s, vmin_spectro, vmax_spectro, *args, **kwargs)
        self.init_mpl()
        self.mpl_layout.addWidget(self.mpl_models)
        self.last_date_processed = None  # enable to know if the user moved since the last results visualization

    def on_key(self, key):
        """ Checks if the TiSSNet shortcut has been pressed.
        :param key: The key pressed.
        :return: None.
        """
        if key.key == "shift+enter":
            self.process_tissnet()
        else:
            super().on_key(key)

    def init_mpl(self):
        """ Initialize the Matplotlib widget.
        :return: None
        """
        self.mpl_models = MplCanvas(self)
        self.mpl_models.axes.figure.subplots_adjust(left=0.05, right=0.97, bottom=0, top=0.94)
        self.mpl_models.axes.axis('off')
        self.mpl_models.setFixedHeight(40)
        self.mpl_models.setVisible(False)

    def process_tissnet(self):
        """ Apply TiSSNet on the current spectrogram and adds the result below it with a colormap.
        :return: None.
        """
        if self.spectralViewer.detection_model is None:
            print("Trying to use TiSSNet but it has not been loaded")
            return

        start, end = self.get_time_bounds()
        if self.last_date_processed == (start, end):
            # the user did not move and likely wants to remove the result visualization
            self.mpl_layout.removeWidget(self.mpl_models)
            self.mpl_models.setVisible(False)
            return

        self.last_date_processed = (start, end)
        data = self.manager.get_segment(start, end)
        spectrogram = make_spectrogram(data, self.manager.sampling_f, t_res=0.5342, f_res=0.9375, return_bins=False,
                                       normalize=True, vmin=-35, vmax=140).astype(np.float32)
        spectrogram = spectrogram[np.newaxis, :, :]
        input_data = Resize((128, spectrogram.shape[-1]))(torch.from_numpy(spectrogram))  # resize data
        with torch.no_grad():
            res = self.spectralViewer.detection_model(input_data).numpy()

        self.showModelResult(res)

    def showModelResult(self, to_show):
        """ Shows a 1D time series as a small heat map.
        :param to_show: The 1D data to show.
        :return: None.
        """
        self.mpl_layout.removeWidget(self.mpl_models)
        self.init_mpl()
        self.mpl_layout.addWidget(self.mpl_models)

        self.mpl_models.axes.imshow(to_show.reshape((1, -1)), aspect="auto", vmin=0, vmax=1)
        self.mpl_models.setVisible(True)
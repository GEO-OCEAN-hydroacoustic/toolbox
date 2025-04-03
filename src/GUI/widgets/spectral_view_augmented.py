import numpy as np
import torch
from torchvision.transforms import Resize
from scipy.signal import find_peaks
from skimage.transform import resize

from GUI.widgets.mpl_canvas import MplCanvas
from GUI.widgets.spectral_view import SpectralView, MIN_SEGMENT_DURATION_S
from utils.physics.signal.make_spectrogram import make_spectrogram


class SpectralViewTissnet(SpectralView):
    """ Spectral view enabling, with shift+enter, to apply TiSSNet on the current spectrogram.
    """
    def __init__(self, SpectralViewer, station, date=None, delta_view_s=MIN_SEGMENT_DURATION_S, *args, **kwargs):
        """ Initialize the SpectralView.
        :param SpectralViewer: The parent SpectralViews window.
        :param station: A Station instance.
        :param date: Initial datetime on which to focus the widget. If None, the start of the data will be chosen.
        :param delta_view_s: Initial half duration of the shown spectrogram, in s.
        :param args: Supplementary arguments for the widget as a PySide widget.
        :param kwargs: Supplementary key arguments for the widget as a PySide widget.
        """
        super().__init__(SpectralViewer, station, date, delta_view_s, *args, **kwargs)
        self.init_mpl()
        self.mpl_layout.addWidget(self.mpl_models)

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
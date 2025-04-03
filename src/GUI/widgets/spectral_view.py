import datetime
import os
import sounddevice
import numpy as np
import scipy
from PySide6 import QtWidgets, QtGui
from PySide6.QtWidgets import QLabel, QSlider, \
    QDateTimeEdit, QDoubleSpinBox, QHBoxLayout, QWidget
from PySide6.QtCore import (QDate, QDateTime, QTime, Qt, QTimer)
from scipy import signal

from GUI.widgets.mpl_canvas import MplCanvas
from utils.physics.signal.make_spectrogram import make_spectrogram
# minimal duration of the spectrograms shown, in s
MIN_SEGMENT_DURATION_S = 30
# maximal duration of the spectrograms shown, in s
MAX_SEGMENT_DURATION_S = 36_000

class SpectralView(QtWidgets.QWidget):
    """ Widget to explore a .wav folder by viewing spectrograms
    """

    def __init__(self, SpectralViewer, station, date=None, delta_view_s=MIN_SEGMENT_DURATION_S*8, vmin_spectro=60, vmax_spectro=120, *args, **kwargs):
        """ Constructor initializing various parameters and setting up the interface.
        :param SpectralViewer: The parent SpectralViews window.
        :param station: A Station instance.
        :param date: Initial datetime on which to focus the widget. If None, the start of the data will be chosen.
        :param delta_view_s: Initial half duration of the shown spectrogram, in s.
        :param vmin_spectro: Minimum colorbar bound used for spectrogram visualization.
        :param vmax_spectro: Maximum colorbar bound used for spectrogram visualization.
        :param args: Supplementary arguments for the widget as a PySide widget.
        :param kwargs: Supplementary key arguments for the widget as a PySide widget.
        """
        super(SpectralView, self).__init__(*args, **kwargs)
        self.spectralViewer = SpectralViewer
        self.station = station
        station.load_data()
        self.manager = station.manager
        self.waveform = False  # if true, displays the waveform instead of the spectrogram
        self.focused = True  # if true, this spectral_view is used for computations such as source location
        self.vmin_spectro, self.vmax_spectro = vmin_spectro, vmax_spectro

        # min and max shown frequencies
        self.freq_range = [0, int(self.manager.sampling_f / 2)]

        self.data, self.start, self.end = [], None, None

        # main layout that will contain the spectro, dataset name and options
        self.layout = QtWidgets.QHBoxLayout(self)
        # secondary layout (at left) that will contain dataset name and options
        self.control_widget = QWidget(self)
        self.control_widget.setFixedWidth(250)
        self.control_layout = QtWidgets.QVBoxLayout(self.control_widget)
        self.control_layout.setContentsMargins(4, 4, 4, 4)
        self.layout.addWidget(self.control_widget)

        # control buttons
        self.control_buttons_layout = QHBoxLayout()
        self.control_layout.addLayout(self.control_buttons_layout)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.setStyleSheet("background-color: red")
        self.close_button.clicked.connect(self.close_button_click)
        self.control_buttons_layout.addWidget(self.close_button)
        self.unfocus_button = QtWidgets.QPushButton("Unfocus")
        self.unfocus_button.setCheckable(True)
        self.unfocus_button.setChecked(False)
        self.unfocus_button.setStyleSheet("""
            QPushButton {
                border: 2px solid #e8a838;
                padding: 4px 8px;
                background-color: #e8a838;
                color: #5c3f0e;
            }
            QPushButton:checked {
                background-color: #fdf3e0;
                color: #e8a838;
            }
        """)
        self.unfocus_button.clicked.connect(self.unfocus_button_click)
        self.control_buttons_layout.addWidget(self.unfocus_button)
        self.waveform_button = QtWidgets.QPushButton("Waveform")
        self.waveform_button.setCheckable(True)
        self.waveform_button.setChecked(False)
        self.waveform_button.setStyleSheet("""
            QPushButton {
                border: 2px solid #3daee9;
                padding: 4px 8px;
                background-color: #3daee9;
                color: #0e3a5c;
            }
            QPushButton:checked {
                background-color: #e0f0fd;
                color: #3daee9;
            }
        """)
        self.waveform_button.clicked.connect(self.toggle_waveform)
        self.control_buttons_layout.addWidget(self.waveform_button)

        # name of the dataset
        self.label = QLabel(self)
        self.label.setFixedWidth(240)
        dataset = self.station.dataset or self.manager.path.split("/")[-2]
        label_text = f'{dataset} - {self.station.name}'
        if self.station.lat:
            label_text += f"\n({self.station.lat:.1f}, {self.station.lon:.1f})"
        self.label.setText(label_text)  # dataset - station \n coordinates (if available)
        font = QtGui.QFont()
        font.setBold(True)
        self.label.setFont(font)
        self.control_layout.addWidget(self.label)

        # controller of the segment length, made of a spin box (text editor) and a slider, both interconnected
        # segment length: label + spin box on same line, slider below
        self.segment_length_layout = QtWidgets.QVBoxLayout()
        self.segment_length_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.control_layout.addLayout(self.segment_length_layout)

        self.segment_length_row = QHBoxLayout()
        self.segment_length_label = QLabel("Segment duration (s)", self)
        self.segment_length_row.addWidget(self.segment_length_label)
        self.segment_length_doubleSpinBox = QDoubleSpinBox(self)
        self.segment_length_doubleSpinBox.setMinimum(MIN_SEGMENT_DURATION_S)
        self.segment_length_doubleSpinBox.setMaximum(MAX_SEGMENT_DURATION_S)
        self.segment_length_doubleSpinBox.setValue(2 * delta_view_s)
        self.segment_length_row.addWidget(self.segment_length_doubleSpinBox)
        self.segment_length_layout.addLayout(self.segment_length_row)

        self.segment_length_slider = QSlider(Qt.Horizontal, self)
        self.segment_length_slider.setMinimum(MIN_SEGMENT_DURATION_S)
        self.segment_length_slider.setMaximum(MAX_SEGMENT_DURATION_S)
        self.segment_length_layout.addWidget(self.segment_length_slider)
        # slider trigger to update the spin box when released (and recursively the plot)
        self.segment_length_slider.sliderReleased.connect(self.update_segment_length_editor)
        self.segment_length_doubleSpinBox.valueChanged.connect(self.update_date_bounds)
        self.segment_length_doubleSpinBox.valueChanged.connect(self.update_segment_length_slider)
        self.segment_length_doubleSpinBox.valueChanged.connect(self.update_plot)

        # controller of the spectrogram color bounds (vmin / vmax)
        self.spectro_bounds_layout = QtWidgets.QGridLayout()
        self.spectro_bounds_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.control_layout.addLayout(self.spectro_bounds_layout)

        self.spectro_bounds_label = QLabel(self)
        self.spectro_bounds_label.setText("Spectrogram bounds (dB)")
        self.spectro_bounds_layout.addWidget(self.spectro_bounds_label, 0, 0, 1, 2)

        # vmin
        self.vmin_label = QLabel("min", self)
        self.spectro_bounds_layout.addWidget(self.vmin_label, 1, 0)
        self.vmin_slider = QSlider(Qt.Horizontal, self)
        self.vmin_slider.setMinimum(0)
        self.vmin_slider.setMaximum(200)
        self.vmin_slider.setValue(self.vmin_spectro)
        self.spectro_bounds_layout.addWidget(self.vmin_slider, 1, 1)

        # vmax
        self.vmax_label = QLabel("max", self)
        self.spectro_bounds_layout.addWidget(self.vmax_label, 2, 0)
        self.vmax_slider = QSlider(Qt.Horizontal, self)
        self.vmax_slider.setMinimum(0)
        self.vmax_slider.setMaximum(200)
        self.vmax_slider.setValue(self.vmax_spectro)
        self.spectro_bounds_layout.addWidget(self.vmax_slider, 2, 1)

        # connect sliders
        self.vmin_slider.valueChanged.connect(self._update_vmin_tooltip)
        self.vmax_slider.valueChanged.connect(self._update_vmax_tooltip)
        self.vmin_slider.sliderReleased.connect(self.update_spectro_bounds)
        self.vmax_slider.sliderReleased.connect(self.update_spectro_bounds)

        # initialize tooltips
        self._update_vmin_tooltip()
        self._update_vmax_tooltip()

        # controller of the segment date, made of a date time editor and a slider, both interconnected
        # segment date: label + datetime edit on same line, slider below
        self.segment_date_layout = QtWidgets.QVBoxLayout()
        self.segment_date_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.control_layout.addLayout(self.segment_date_layout)

        self.segment_date_row = QHBoxLayout()
        self.segment_date_label = QLabel("Segment date", self)
        self.segment_date_row.addWidget(self.segment_date_label)
        self.segment_date_dateTimeEdit = QDateTimeEdit(self)
        self.segment_date_dateTimeEdit.setInputMethodHints(Qt.ImhPreferNumbers)
        self.segment_date_dateTimeEdit.setDisplayFormat("yyyy/MM/dd hh:mm:ss")
        self.segment_date_dateTimeEdit.setDateTime(datetime_to_Qdatetime(self.manager.dataset_start))
        self.segment_date_row.addWidget(self.segment_date_dateTimeEdit)
        self.segment_date_layout.addLayout(self.segment_date_row)

        self.segment_date_slider = QSlider(Qt.Horizontal, self)
        self.segment_date_layout.addWidget(self.segment_date_slider)
        # slider trigger to update the date time edit when released (and recursively the plot)
        self.segment_date_slider.sliderReleased.connect(self.update_segment_date_editor)
        if date is not None:
            self.segment_date_dateTimeEdit.setDateTime(datetime_to_Qdatetime(date))
        # date time edit triggers to update the slider and the plot when changed
        self.segment_date_dateTimeEdit.dateTimeChanged.connect(self.update_segment_date_slider)
        self.segment_date_dateTimeEdit.dateTimeChanged.connect(self.update_plot)

        # mpl canvas containing the spectrogram
        self.mpl_layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.mpl_layout)
        self.mpl = MplCanvas(self)
        self.mpl.axes.figure.subplots_adjust(left=0.05, right=0.97, bottom=0.15, top=0.94)
        self.mpl_layout.addWidget(self.mpl)

        # initialize the different widgets
        self.layout.setStretch(1, 5)
        self.update_date_bounds()
        self.update_segment_length_slider()
        self.update_segment_date_slider()
        self.update_plot()

    def close_button_click(self):
        """ Close the current spectral_view.
        :return: None.
        """
        self.spectralViewer.close_spectral_view(self)

    def unfocus_button_click(self):
        """ Remove the current spectral_view from the focus, s.t. it is not used for computations like source location.
        :return: None.
        """
        self.spectralViewer.unfocus_spectral_view(self)
        self.unfocus_button.setText("Unfocus" if self.focused else "Focus")

    def update_spectro_bounds(self):
        """Update vmin/vmax from sliders, ensuring vmin < vmax, then refresh the plot."""
        vmin = self.vmin_slider.value()
        vmax = self.vmax_slider.value()
        if vmin >= vmax:
            # impossible case, the vmax or vmin are modified accordingly
            if self.sender() == self.vmin_slider:
                vmax = vmin + 1
                self.vmax_slider.setValue(vmax)
            else:
                vmin = vmax - 1
                self.vmin_slider.setValue(vmin)
        self.vmin_spectro = vmin
        self.vmax_spectro = vmax
        self.start = None
        self.update_plot()

    def _update_vmin_tooltip(self):
        self.vmin_slider.setToolTip(f"{self.vmin_slider.value()} dB")

    def _update_vmax_tooltip(self):
        self.vmax_slider.setToolTip(f"{self.vmax_slider.value()} dB")

    def toggle_waveform(self):
        """ Change the current visualization to waveform
        :return: None
        """
        self.waveform = self.waveform_button.isChecked()
        self.waveform_button.setText("Spectrogram" if self.waveform else "Waveform")
        self.start = None
        self.update_plot()

    def update_segment_length_editor(self):
        """ Update the segment length spin box after the slider is changed.
        :return: None.
        """
        self.segment_length_doubleSpinBox.setValue(self.segment_length_slider.value())

    def update_segment_length_slider(self):
        """ Update the segment length slider after the spin box is changed.
        :return: None.
        """
        self.segment_length_slider.setValue(self.segment_length_doubleSpinBox.value())

    def update_segment_date_slider(self):
        """ Update the segment date slider after the editor is changed.
        :return: None.
        """
        self.segment_date_slider.sliderReleased.disconnect(self.update_segment_date_editor)

        new_date = Qdatetime_to_datetime(self.segment_date_dateTimeEdit.date(),
                                         self.segment_date_dateTimeEdit.time())
        old_date = self.manager.dataset_start + datetime.timedelta(seconds=self.segment_date_slider.value())
        self.spectralViewer.notify_delta(new_date - old_date, self)
        self.segment_date_slider.setValue((Qdatetime_to_datetime(self.segment_date_dateTimeEdit.date(),
                                                                 self.segment_date_dateTimeEdit.time()) - self.manager.dataset_start).total_seconds())

        self.segment_date_slider.sliderReleased.connect(self.update_segment_date_editor)

    def update_segment_date_editor(self, secondary_trigger=False):
        """ Update the segment date editor after the slider is changed.
        :return: None.
        """
        self.segment_date_dateTimeEdit.dateTimeChanged.disconnect(self.update_segment_date_slider)
        self.segment_date_dateTimeEdit.dateTimeChanged.disconnect(self.update_plot)

        delta = self.segment_length_doubleSpinBox.value() / 2
        delta = datetime.timedelta(seconds=delta)
        old_date = Qdatetime_to_datetime(self.segment_date_dateTimeEdit.date(),
                                         self.segment_date_dateTimeEdit.time())
        new_date = self.manager.dataset_start + delta + datetime.timedelta(seconds=self.segment_date_slider.value())
        self.spectralViewer.notify_delta(new_date - old_date, self)
        self.segment_date_dateTimeEdit.setDateTime(datetime_to_Qdatetime(new_date))

        self.segment_date_dateTimeEdit.dateTimeChanged.connect(self.update_segment_date_slider)
        self.segment_date_dateTimeEdit.dateTimeChanged.connect(self.update_plot)

        self.update_plot()

    def update_date_bounds(self):
        """ Update the current min and max allowed segment date given current segment length (to avoid getting out of
        the dataset).
        :return: None.
        """
        delta = self.segment_length_doubleSpinBox.value() / 2 + 1  # 1s of margin to avoid border conflicts
        delta = datetime.timedelta(seconds=delta)
        start = self.manager.dataset_start + delta
        end = self.manager.dataset_end - delta
        self.segment_date_dateTimeEdit.setMinimumDateTime(datetime_to_Qdatetime(start))
        self.segment_date_dateTimeEdit.setMaximumDateTime(datetime_to_Qdatetime(end))
        self.segment_date_slider.setMinimum(0)
        self.segment_date_slider.setMaximum((end - start).total_seconds())

    def _draw_picks(self, picks_dict, delta, color, alpha):
        """Draw vertical lines for picks of the current station within the visible window."""
        if self.station not in picks_dict:
            return
        l = picks_dict[self.station]
        idx = max(np.searchsorted(l, self.start, side="left") - 1, 0)
        while idx < len(l) and l[idx] < self.end:
            if l[idx] > self.start:
                t = (l[idx] - (self.start + delta)).total_seconds()
                self.mpl.axes.axvline(t, color=color, alpha=alpha, linestyle="--")
            idx += 1

    def _update_data(self):
        """ Update the spectrogram value.
        :return: None.
        """
        delta = datetime.timedelta(seconds=self.segment_length_doubleSpinBox.value() / 2)
        start, end = self.get_time_bounds()
        # get the spectro from the extractor to be able to show it
        if self.start != start or self.end != end:
            self.start, self.end = start, end
            self.data = self.manager.get_segment(start, end)
            self.mpl.axes.cla()
            extent = [-delta.total_seconds(), delta.total_seconds(), 0, self.manager.sampling_f/2]
            if self.waveform:
                self.mpl.axes.plot(np.linspace(extent[0], extent[1], len(self.data)), self.data)
                self.mpl.axes.set_xlim(extent[0], extent[1])
                self.mpl.axes.set_ylim(-np.max(np.abs(self.data)), np.max(np.abs(self.data)))
            else:
                (f, t, spectro) = make_spectrogram(self.data, self.manager.sampling_f, t_res=0.25, f_res=0.9375,
                                                   return_bins=True, normalize=True, vmin=self.vmin_spectro,
                                                   vmax=self.vmax_spectro)
                extent = [-delta.total_seconds(), delta.total_seconds(), 0, self.manager.sampling_f / 2]
                self.mpl.axes.imshow(spectro, aspect="auto", extent=extent, vmin=0, vmax=1, cmap="inferno")
                self.mpl.axes.set_ylim(self.freq_range[0], self.freq_range[1])
                self.mpl.axes.set_ylabel('f (Hz)')

            self.mpl.axes.axvline(0, color="w", alpha=0.25, linestyle="--")
            self.mpl.axes.set_xlabel('t (s)')

            # get previously picked events
            pick_sources = [
                (self.spectralViewer.loc_res, "bisque", 0.5),
                (self.spectralViewer.loc_res_single, "yellow", 0.25),
                (self.spectralViewer.loc_res_prev, "deepskyblue", 0.5),
                (self.spectralViewer.loc_res_single_prev, "lime", 0.25),
            ]
            for picks_dict, color, alpha in pick_sources:
                self._draw_picks(picks_dict, delta, color, alpha)

    def _draw(self):
        """ Update the plot widget.
        :return: None.
        """
        self.mpl.draw()

    def update_plot(self):
        """ Update the spectrogram and then the plot widget.
        :return: None.
        """
        self._update_data()
        self._draw()

    def on_click(self, click):
        """ Handles click operation to enable point and click move.
        :param click: Object giving details about the click.
        :return: None.
        """
        if click.xdata is not None:
            if click.button.name == "LEFT":
                segment_center = Qdatetime_to_datetime(self.segment_date_dateTimeEdit.date(),
                                                       self.segment_date_dateTimeEdit.time())
                segment_center += datetime.timedelta(seconds=click.xdata)  # move to the click location
                self.segment_date_dateTimeEdit.setDateTime(datetime_to_Qdatetime(segment_center))
            elif click.button.name == "RIGHT":  # Pn / T annotation
                path = self.spectralViewer.loc_res_path[:-4] + "_single.csv"
                if path is not None:
                    with open(path, "a+") as f:
                        name = self.station.name
                        center_time = Qdatetime_to_datetime(self.segment_date_dateTimeEdit.date(),
                                                               self.segment_date_dateTimeEdit.time())
                        click_time = center_time + datetime.timedelta(seconds=click.xdata)
                        if "ShiftModifier" in str(click.guiEvent):
                            f.write(f'{name},{click_time.strftime("%Y%m%d_%H%M%S")},{center_time.strftime("%Y%m%d_%H%M%S")}')
                        else:
                            f.write(f'{name},{click_time.strftime("%Y%m%d_%H%M%S")}')
                        self.spectralViewer.loc_res_single.setdefault(self.station, []).append(click_time)
                        self.spectralViewer.loc_res_single[self.station] = sorted(self.spectralViewer.loc_res_single[self.station])
                        self.mpl.axes.axvline(click.xdata, color="yellow", alpha=0.25, linestyle="--")
                        self.mpl.draw()
                        f.write("\n")

    def on_key(self, key):
        """ Handles key press operations by passing the event to the parent spectral_viewer.
        :param key: Object giving details about the key press.
        :return: None.
        """
        self.spectralViewer.on_key(self, key)

    def on_key_local(self, key):
        """ Handles key press operations to enable various shortcuts, only acting on the current spectral_view.
        :param key: Object giving details about the key press.
        :return: None.
        """
        segment_center = Qdatetime_to_datetime(self.segment_date_dateTimeEdit.date(),
                                               self.segment_date_dateTimeEdit.time())
        delta = datetime.timedelta(seconds=self.segment_length_doubleSpinBox.value() / 2)

        if key.key == 'right':
            # go to right
            segment_center += delta
        elif key.key == 'left':
            # go to left
            segment_center -= delta
        elif key.key == '+':
            # zoom
            delta /= 2
        elif key.key == '-':
            # dezoom
            delta *= 2
        elif key.key == 'enter':
            # play the sound with 20 times increased frequency
            data = self.manager.get_segment(segment_center - delta, segment_center + delta)
            data = data / np.max(np.abs(data))

            # write a file
            to_write = np.int16(32767 * data / np.max(np.abs(data)))
            # name : station_YYYYMMDD_hhmmss_durations
            station_str = f"{self.station.dataset}_{self.station.name}" if self.station.dataset else self.station.name
            name = f'{station_str}_{(segment_center-delta).strftime("%Y%m%d_%H%M%S")}_{int((delta*2).total_seconds())}s.wav'
            scipy.io.wavfile.write(f"../../data/GUI/{name}", int(self.manager.sampling_f * 20), to_write)
            # play the sound 20x faster
            sounddevice.play(to_write, int(self.manager.sampling_f * 20))

        elif key.key == "up":
            # increase min and max allowed frequency, respecting the limit of the Nyquist-Shannon frequency
            d = min(self.manager.sampling_f / 2 - self.freq_range[1],
                    0.1 * self.manager.sampling_f / 2)
            self.freq_range[0] += d
            self.freq_range[1] += d

        elif key.key == "down":
            # decrease min and max allowed frequency, respecting the limit of 0
            d = min(self.freq_range[0],
                    0.1 * self.manager.sampling_f / 2)
            self.freq_range[0] -= d
            self.freq_range[1] -= d

        elif key.key == "*":
            # decrease the maximal allowed frequency
            d = 0.05 * (self.freq_range[1] - self.freq_range[0])
            self.freq_range[1] -= d

        elif key.key == "/":
            # increase the minimal and maximal allowed frequency, respecting the limit of 0 and the
            # Nyquist-Shannon frequency
            delta_top = min(0.05 * (self.freq_range[1] - self.freq_range[0]),
                            self.manager.sampling_f / 2 - self.freq_range[1])
            delta_bot = min(0.05 * (self.freq_range[1] - self.freq_range[0]),
                            self.freq_range[0])
            self.freq_range[0] -= delta_bot
            self.freq_range[1] += delta_top

        elif key.key == "ctrl+enter":
            # put all the SpectralView widgets of the parent window to the same configuration
            segment_center = Qdatetime_to_datetime(self.segment_date_dateTimeEdit.date(),
                                                   self.segment_date_dateTimeEdit.time())
            for spectralview in self.spectralViewer.SpectralViews:
                spectralview.freq_range = self.freq_range.copy()
                spectralview.change_date(segment_center)
                spectralview.change_segment_length(self.segment_length_doubleSpinBox.value())
        else:
            return

        # update widgets
        self.start = None
        self.segment_date_dateTimeEdit.setDateTime(datetime_to_Qdatetime(segment_center))
        self.segment_length_doubleSpinBox.setValue(delta.total_seconds() * 2)
        self.update_plot()

    def change_date(self, new_date):
        """ Change the date the widget focuses on to a new one.
        :param new_date: The new datetime to focus on.
        :return: None.
        """
        # round the new date to the closest second to avoid having microseconds
        new_date = new_date + datetime.timedelta(seconds=round(new_date.microsecond / 10 ** 6))
        new_date = new_date.replace(microsecond=0)
        # simply update the segment date editor which will cascade the change
        self.segment_date_dateTimeEdit.setDateTime(datetime_to_Qdatetime(new_date))

    def change_segment_length(self, new_length):
        """ Change the current segment length.
        :param new_length: The new length, in seconds.
        :return: None.
        """
        # simply update the segment length editor which will cascade the change
        self.segment_length_doubleSpinBox.setValue(new_length)

    def get_time_bounds(self):
        """ Get the current time bounds of the tool and return them as a tuple.
        :return: A tuple (start, end) of datetimes.
        """
        segment_center = Qdatetime_to_datetime(self.segment_date_dateTimeEdit.date(),
                                               self.segment_date_dateTimeEdit.time())
        delta = datetime.timedelta(seconds=self.segment_length_doubleSpinBox.value() / 2)

        return segment_center - delta, segment_center + delta


def Qdatetime_to_datetime(qdate, qtime):
    """ Utility function to convert PySide dates and times to Python datetimes.
    :param qdate: A PySide date (Y,M,D).
    :param qtime: A Pyside time (h,m,s).
    :return: A Python datetime.
    """
    return datetime.datetime(qdate.year(), qdate.month(), qdate.day(),
                             qtime.hour(), qtime.minute(), qtime.second())


def datetime_to_Qdatetime(dt, day_offset=0):
    """ Utility function to convert Python datetimes to PySide datetimes.
    :param dt: A Python datetime.
    :param day_offset: Optional offset in days.
    :return: A PySide datetime.
    """
    dt = dt + datetime.timedelta(days=day_offset)
    return QDateTime(
        QDate(dt.year, dt.month, dt.day),
        QTime(dt.hour, dt.minute, dt.second))

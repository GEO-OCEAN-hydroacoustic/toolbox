import datetime
import wave

import numpy as np
import math
import re
import locale
import soundfile as sf
import warnings
from src.utils.time.GPS import get_leap_second
from src.utils.transformation.signal import butter_bandpass_filter


class SoundFile:
    """ Class representing a single sound file.
    """
    EXTENSION = ""  # extension of the files like ".wav"

    def __init__(self, path, skip_data=False, identifier=None):
        """ Constructor reading file metadata and content if required.
        :param path: The path of the file.
        :param skip_data: If True, we only read the metadata of the file. Else, we also read its content.
        :param identifier: The ID of this file, that will be used to compare it with another file. If unspecified,
        path is used.
        """
        self.header = {}  # dict that contains the metadata of the file
        self.data = []  # data of the file
        self.path = path
        self.identifier = self.path if identifier is None else identifier
        self._read_header()  # read the metadata of the file
        if not skip_data:
            self.read_data()  # read the data of the file

    def _read_header(self):
        """ Read the metadata of the file and update self.header.
        :return: None.
        """
        pass  # abstract method

    def read_data(self):
        """ Read the data of the file and update self.data. Public method because in case the file was previously loaded
        but without reading its data, it may be necessary to ask to read the data.
        :return: None.
        """
        self.data = self.read_data_subpart(None, None)

    def read_data_subpart(self, offset_points_start, points_to_keep):
        """ Read the specified part of the data of the file and return it.
        In case the file data was already read we use it, otherwise we just read the relevant data of the file.
        :param offset_points_start: Number of points to skip before the data part to keep. None if from the start.
        :param points_to_keep: Number of points to keep. None in case we keep everything after the start.
        :return: The required data.
        """
        if len(self.data) > 0:
            return self._read_data_subpart_cached(offset_points_start, points_to_keep)
        return self._read_data_subpart_uncached(offset_points_start, points_to_keep)

    def _read_data_subpart_uncached(self, offset_points_start, points_to_keep):
        """ Read the specified part of the data from yet uncached file data and return it.
        :param offset_points_start: Number of points to skip before the data part to keep. None if from the start.
        :param points_to_keep: Number of points to keep. None in case we keep everything after the start.
        :return: The required data.
        """
        return None  # abstract method

    def _read_data_subpart_cached(self, offset_points_start, points_to_keep):
        """ Read the specified part of the data from cached file data and return it.
        :param offset_points_start: Number of points to skip before the data part to keep. None if from the start.
        :param points_to_keep: Number of points to keep. None in case we keep everything after the start.
        :return: The required data.
        """
        data = self.data
        if offset_points_start is not None:
            data = data[offset_points_start:]
        if points_to_keep is not None:
            data = data[:points_to_keep]
        return data

    def get_data(self, start=None, end=None):
        """ Given a start datetime and an end datetime, read the data. In case the bounds are outside of the file, the
        method does not throw an exception and simply return the data from its start, and/or to its end.
        :param start: Start datetime of the segment.
        :param end: End datetime of the segment.
        :return: The required data.
        """
        if start is not None and end is not None:
            if start >= end:
                return []
        offset_points_start, points_to_keep = None, None

        if start is not None:
            # select the data starting after start
            offset_start = start - self.header["start_date"]
            offset_points_start = int(offset_start.total_seconds() * self.header["sampling_frequency"])
            offset_points_start = None if offset_points_start < 0 else offset_points_start

        if end is not None:
            # select the data ending before end
            keep = end - (max(start, self.header["start_date"]) if start else self.header["start_date"])
            points_to_keep = int(keep.total_seconds() * self.header["sampling_frequency"])
            max_possible_length = self.header["samples"] - offset_points_start if offset_points_start else self.header[
                "samples"]
            points_to_keep = None if points_to_keep > max_possible_length else points_to_keep

        if points_to_keep is not None and points_to_keep < 0:
            return []

        data = self.read_data_subpart(offset_points_start, points_to_keep)
        return data

    def __eq__(self, other):
        """ Test if another file has a similar identifier, or if an item is the file identifier.
        :param other: Another SoundFile or an object of the class of self.identifier.
        :return: True if other is a SoundFile with the same identifier or if other is our identifier, else False.
        """
        if type(other) == type(self.identifier):  # the other object may be the identifier
            return self.identifier == other
        if type(other) == type(self):  # the other object is a SoundFile that may have the same identifier
            return self.identifier == other.identifier
        return False


class WavFile(SoundFile):
    """ Class representing .wav files. We expect wav files to be named with their start time as YYYYMMDD_hhmmss.
    """
    EXTENSION = "wav"
    TO_VOLT = 5.0 / 2 ** 32 # we consider a fixed dynamic range of 5V on 32 bits

    def __init__(self, path, sensitivity=-163.5, skip_data=False, identifier=None):
        """ Constructor reading file metadata and content if required.
        :param path: The path of the file.
        :param sensitivity: Sensibility of the sensor.
        :param skip_data: If True, we only read the metadata of the file. Else, we also read its content.
        :param identifier: The ID of this file, that will be used to compare it with another file. If unspecified,
        path is used.
        """
        self.sensitivity = sensitivity
        super().__init__(path, skip_data, identifier)

    def _read_header(self):
        """ Read the metadata of the file using its name and header and update self.header.
        :return: None.
        """
        with wave.open(self.path, 'rb') as file:
            # get information from the file header
            self.header["sampling_frequency"] = file.getframerate()
            self.header["samples"] = file.getnframes()

        duration_micro = 10 ** 6 * self.header["samples"] / self.header["sampling_frequency"]
        self.header["duration"] = datetime.timedelta(microseconds=duration_micro)
        file_name = self.path.split("/")[-1][:-4]  # get the name of the file and get rid of extension

        # example : 24-02-22_000011.128000_acq.wav
        self.header["start_date"] = datetime.datetime.strptime("20"+file_name, "%Y-%m-%d_%H%M%S.%f_acq")
        self.header["end_date"] = self.header["start_date"] + self.header["duration"]

    def _read_data_subpart_uncached(self, offset_points_start, points_to_keep):
        """ Read and return the required data.
        :param offset_points_start: Number of points to skip before the data part to keep. None if from the start.
        :param points_to_keep: Number of points to keep. None in case we keep everything after the start.
        :return: The required data.
        """
        file = sf.SoundFile(self.path)
        file.seek(offset_points_start if offset_points_start else 0)
        data = file.read(points_to_keep if points_to_keep else -1, dtype='int32')
        data = data * (self.TO_VOLT / 10 ** (self.sensitivity / 20))
        return data


class DatFile(SoundFile):
    """ Class representing .dat files specific of GEO-OCEAN lab. Relevant metadata are in the files headers.
    """
    EXTENSION = "DAT"
    TO_VOLT = 5.0 / 2 ** 24  # we consider a fixed dynamic range of 5V on 24 bits

    def __init__(self, path, sensitivity=-163.5, skip_data=False, identifier=None, raw=False):
        """ Constructor reading file metadata and content if required.
        :param path: The path of the file.
        :param sensitivity: Sensibility of the sensor.
        :param skip_data: If True, we only read the metadata of the file. Else, we also read its content.
        :param identifier: The ID of this file, that will be used to compare it with another file. If unspecified,
        path is used.
        """
        self.sensitivity = sensitivity
        self.raw = raw
        # if "raw" in path:
        #     self.raw = True
        super().__init__(path, skip_data, identifier)

    def _read_header(self):
        """ Read the metadata of the file using its header and update self.header.
        :return: None.
        """
        with open(self.path, 'rb') as file:
            file_header = file.read(400)

        file_header = file_header.decode('ascii').split("\n")

        self.header["site"] = file_header[3].split()[1]
        self.header["bytes_per_sample"] = int(re.findall(r'(\d*)\s*[bB]', file_header[6])[0])
        self.header["samples"] = int(int(file_header[7].split()[1]))
        self.header["sampling_frequency"] = float(file_header[5].split()[1])
        self._original_samples = int(file_header[7].split()[1])
        duration_micro = 10 ** 6 * float(file_header[7].split()[1]) / float(file_header[5].split()[1])
        self.header["duration"] = datetime.timedelta(microseconds=duration_micro)

        if self.raw:
            #str start date is not correct in raw data since it comes from user computer used to start recording
            cycle = 10 ** 6 * float(file_header[11].split()[1]) / float(file_header[5].split()[1])
            try :
                offset = int(file_header[11].split()[-1][:-1])*200 #cylcle reset every 200days
            except :
                # warnings.warn('Data not corrected from offset')
                offset = int(file_header[11].split()[-1])*200
            #forced raw mode header :int(file_header[11].split()[-1])
            gps_zero = ' '.join(file_header[8].split()[-4:])
            locale.setlocale(locale.LC_TIME, "C")  # ensure we use english months names
            if "." in gps_zero:
                gps_zero = datetime.datetime.strptime(gps_zero, "%b %d %H:%M:%S.%f %Y")
            else:
                gps_zero = datetime.datetime.strptime(gps_zero, "%b %d %H:%M:%S %Y")
            self.header["start_date"] = gps_zero + datetime.timedelta(microseconds=cycle)+datetime.timedelta(days=offset)
        else:
            # 9 is 1st sample, 10 is start date
            date = file_header[10].split()
            date = ' '.join(date[-4:])
            locale.setlocale(locale.LC_TIME, "C")  # ensure we use english months names
            if "." in date:
                self.header["start_date"] = datetime.datetime.strptime(date, "%b %d %H:%M:%S.%f %Y")
            else:
                # handle the case where no decimal is present
                self.header["start_date"] = datetime.datetime.strptime(date, "%b %d %H:%M:%S %Y")

        leap = int(get_leap_second(self.header["start_date"]))  # get the current GPS / TAI shift
        self.header["start_date"] -= datetime.timedelta(seconds=leap)
        self.header["end_date"] = self.header["start_date"] + self.header["duration"]

    def _read_data_subpart_uncached(self, offset_points_start, points_to_keep):
        """ Read the specified data of the file using numpy, put it from 24 to 32 bits.
        :param offset_points_start: Number of points to skip before the data part to keep. None if from the start.
        :param points_to_keep: Number of points to keep. None in case we keep everything after the start.
        :return: The required data.
        """
        sampsize = self.header["bytes_per_sample"]
        with open(self.path, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8,
                               offset=400 + ((sampsize * offset_points_start) if offset_points_start else 0),
                               count=(sampsize * points_to_keep) if points_to_keep else -1)

        data = data[:-(len(data) % sampsize)] if (len(data) % sampsize) != 0 else data
        data = data.reshape((-1, sampsize))  # original array with custom nb of bytes

        next_pow_2 = 2 ** math.ceil(math.log2(sampsize))  # size at which the data will be stored

        # prepare array of next_pow_2 bytes
        valid_data = np.zeros((data.shape[0], next_pow_2), dtype=np.uint8)
        neg = (data[:, 0] >= 2 ** 7)
        # in case the nb if negative, add several 1 before, else 0 (see ISO signed integer representation)
        valid_data[:, :next_pow_2 - sampsize] = \
            np.swapaxes(np.full((next_pow_2 - sampsize, neg.shape[0]), (2 ** 8 - 1) * neg), 0, 1)
        valid_data[:, 1:] = data  # copy data content
        # concatenate these bytes to form an int32
        data = np.frombuffer(valid_data[:, ::-1].tobytes(), dtype=np.int32)

        data = data * (self.TO_VOLT / 10 ** (self.sensitivity / 20))
        data = butter_bandpass_filter(data, 1, 119, self.header["sampling_frequency"])

        # now convert data to meaningful data
        return data


class WFile(SoundFile):
    """ Class representing a record from a .w file specific of CTBTO's IMS.
    Virtually, we represent 1 record (1 line of a .wfdisc) by 1 instance of this class.
    """
    EXTENSION = "w"

    def _read_header(self):
        """ The metadata should have been passed in the path attribute, we distribute it accordingly
        :return: None.
        """
        (self.path, self.header["start_date"], self.header["end_date"], self.header["start_index"],
         self.header["end_index"], self.header["sampling_frequency"], self.header["cnt_to_upa"], self.header["site"]) \
            = self.path
        self.header["bytes_per_sample"] = 4  # fix value
        self.header["duration"] = self.header["end_date"] - self.header["start_date"]
        self.header["samples"] = int(self.header["duration"].total_seconds() * self.header["sampling_frequency"])

    def _read_data_subpart_uncached(self, offset_points_start, points_to_keep):
        """ Read the specified data of the file using numpy.
        :param offset_points_start: Number of points to skip before the data part to keep. None if from the start.
        :param points_to_keep: Number of points to keep. None in case we keep everything after the start.
        :return: The required data.
        """
        with (open(self.path, 'rb') as file):
            offset_points_start = 0 if offset_points_start is None else offset_points_start
            if offset_points_start > self.header["samples"]:
                return []  # the required time is after the file because it is in a no-data section
            # the first point is the first from the file + the specified offset
            start_idx = self.header["start_index"] + offset_points_start
            # the nb of samples to read is the one specified if it exists or the nb of samples remaining before the end
            # if it exists or the whole remaining file (-1)
            to_read = points_to_keep if points_to_keep else \
                self.header["end_index"] - start_idx if self.header["end_index"] else -1
            file.seek(start_idx * self.header["bytes_per_sample"])
            data = file.read(-1 if to_read<0 else to_read * self.header["bytes_per_sample"])
        data = np.frombuffer(data, dtype=">f4")

        # now convert data to meaningful data
        data = butter_bandpass_filter(data, 1, 119, self.header["sampling_frequency"])
        data = data * self.header["cnt_to_upa"]
        return data

    def read_data(self):
        """ Method to read the whole file, disabled for .w which are generally very big.
        :return: None
        """
        return None
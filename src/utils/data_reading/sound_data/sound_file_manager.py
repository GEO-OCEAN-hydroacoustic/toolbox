import datetime
import glob
import math
import os
from collections import deque

import numpy as np
import scipy
from tqdm import tqdm

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from src.utils.data_reading.sound_data.sound_file import SoundFile, WavFile, DatFile, WFile

# epsilon to compare two close datetimes
TIMEDELTA_EPSILON = datetime.timedelta(microseconds=10**4)

class SoundFilesManager:
    """ Class embodying an abstract layer between the sound files and the user such that she can call methods to get the
    data on some periods without worrying about the local organization of the files (e.g. duration of files, what if the
    required period is split on several files...).
    We expect the audio data to be organized as a directory (which is named as its station) of individual sound files.
    Each sound file can give the information of its starting time, its sampling frequency and its length
    (this can be encoded in its header, or simply its name). Their names must be such that when sorted by them, the
    files have to be chronologically sorted.
    This way, the manager will be able to determine which file is responsible for which datetime.
    """
    FILE_CLASS = SoundFile  # type of the individual sound files

    def __init__(self, path, cache_size=4, fill_empty_with_zeroes=True, kwargs=None):
        """ Constructor of the class that reads the headers of the files of the folder to get its global organization.
        :param path: Path of the directory containing the files that we want to manage.
        :param cache_size: Number of files to keep loaded in memory, so that if one is used again by the user
        :param fill_empty_with_zeroes: If True, any missing data point between the dataset bounds is replaced by a 0
        :param kwargs: Other arguments as a dict, that can be used by particular FilesManager (e.g. sensitivity...).
        it is fast to load (FIFO fashion).
        """
        self._process_kwargs(kwargs)
        self.path = path

        # cache that keeps most recent files in mem, s.t. they can be used again quicker
        self.cache_size = cache_size
        self.cache = deque()

        self.fill_empty_with_zeroes = fill_empty_with_zeroes

        # we consider the name of the directory as a station name
        self.path = self.path.rstrip("/")
        self.name = self.path.split("/")[-1]

        # get the files in the folder
        self._initialize_files()
        self.dataset_start = self.files[0].header["start_date"]
        self.dataset_end = self.files[-1].header["end_date"]
        # we assume a similar sf for the whole dataset
        self.sampling_f = self.files[0].header["sampling_frequency"]

    def _process_kwargs(self, kwargs):
        """ Do something with the available kwargs. (e.g. initialize sensitivity)
        :param kwargs: Dict of additional arguments.
        :return: None.
        """
        pass

    def _initialize_files(self):
        """ Get the list of the files in the directory and initialize the corresponding SoundFiles.
        :return: None.
        """
        paths = glob.glob(self.path + "/*." + self.FILE_CLASS.EXTENSION)
        if len(paths) == 0:
            raise Exception(f"No files found in {self.path}")
        file_paths = paths

        self.files = []

        # parallelize file headers parsing
        cpus = cpu_count()
        threads = ThreadPool(cpus - 1)
        threads.map(lambda path: self.files.append(self._initialize_sound_file(path)), file_paths)
        threads.close()  # tell the pool to stop threads when they finish
        threads.join()  # wait all threads to be closed

        self.files.sort(key=lambda x: x.header["start_date"])
        for i in range(len(self.files)):
            self.files[i].identifier = i

    def _initialize_sound_file(self, path, skip_data=True, file_number=None):
        """ Read a file and return the corresponding SoundFile instance.
        :param path: Path of the file to load.
        :param skip_data: If True, only read metadata.
        :param file_number: Number of the file to use as ID.
        :return: The SoundFile instance.
        """
        return self.FILE_CLASS(path, skip_data=skip_data, identifier=file_number)

    def _find_file(self, target_datetime):
        """ Find a file containing a given datetime and return its index.
        :param target_datetime: The datetime we look for.
        :return: The index of the matching file or None.
        """
        file_starts = [f.header["start_date"] for f in self.files]
        target_idx = np.searchsorted(file_starts, target_datetime, side="right")-1
        target_idx = max(int(target_idx), 0)
        return target_idx

    def find_file_name(self, target_datetime):
        """ Find a file containing a given datetime and return its index.
        :param target_datetime: The datetime we look for.
        :return: The index of the matching file or None.
        """
        file_starts = [f.header["start_date"] for f in self.files]
        target_idx = np.searchsorted(file_starts, target_datetime, side="right")-1
        target_idx = max(int(target_idx), 0)
        return target_idx


    def _find_files_segment(self, start, end):
        """ Given a start and end datetime, find the first and last file indexes.
        :param start: Start datetime of the segment.
        :param end: End datetime of the segment.
        :return: First and last file indexes.
        """
        assert start >= self.dataset_start - TIMEDELTA_EPSILON, "start is before the first file"
        assert end <= self.dataset_end + TIMEDELTA_EPSILON, "end is after the last file"

        first_file, last_file = self._find_file(start), self._find_file(end)
        return first_file, last_file

    def get_segment(self, start, end, pad_with_zeros=True):
        """ Given a start date and an end date, return an array containing all the data points between them.
        :param start: Start datetime of the segment.
        :param end: End datetime of the segment.
        :return: A numpy array containing the data points in the segment, where unavailable points are replaced by 0s.
        """
        end -= TIMEDELTA_EPSILON  # small epsilon to exclude the last point of the interval
        first_file, last_file = self._find_files_segment(start, end)

        file_numbers = range(first_file, last_file + 1)

        data = []
        # pad start if needed
        if self.files[file_numbers[0]].header["start_date"] > start and pad_with_zeros:
            diff_s = ( self.files[file_numbers[0]].header["start_date"] - start).total_seconds()
            data = [0] * round(diff_s*self.sampling_f)

        for file_number in file_numbers:
            file = self.files[file_number]

            if len(self.cache) > 0:
                file.read_data()  # if we have a cache, read all data of the file

            # check the file indeed includes wanted data (rounding errors may lead to load a useless file or we may be
            # in a data leap)
            if file.header["end_date"] > start and file.header["start_date"] < end:
                file_data = file.get_data(start=start, end=end)
                data.extend(file_data)

            # pad end if needed
            next_start = end if file_number == file_numbers[-1] else self.files[file_number+1].header["start_date"]
            if file.header["end_date"] < next_start and self.fill_empty_with_zeroes:
                diff_s = (next_start - file.header["end_date"]).total_seconds()
                data.extend([0] * round(diff_s * self.sampling_f))

        if len(data) == 0:
            print(f"0-length data fetched for files {file_numbers} from date {start} to {end} (station {self.name})")

        return np.array(data)

    def flush_cache(self):
        """ Clear the cache.
        :return: None.
        """
        self.cache.clear()

    def to_wav(self, start, end, file_duration, path):
        """ Load the required segment and save it as a wav file.
        :param start: Start of the segment to save. Start of the dataset if None.
        :param end: End of the segment to save. End of the dataset if None.
        :param file_duration: Duration, as a datetime.timedelta, of a single output file.
        :param path: Path of the directory where to save the wav file.
        :return: None
        """
        start, end = start or self.dataset_start, end or self.dataset_end
        for i in tqdm(range(math.ceil((end-start)/file_duration)), desc="Converting dataset"):
            seg_start = start + i*file_duration
            seg_end = min(seg_start + file_duration, end)

            data = np.array(self.get_segment(seg_start, seg_end), dtype=np.int32)
            scipy.io.wavfile.write(f'{path}/{seg_start.strftime("%Y%m%d_%H%M%S")}.wav',
                                   int(self.sampling_f), data)


    def __eq__(self, other):
        """ Test if another manager works with the same path.
        :param other: Another manager.
        :return: True if other is a manager with the same directory, else False.
        """
        if type(self) == type(other) and other.path == self.path:
            return True
        return False

    def __str__(self):
        """ Basic display of sound file manager.
        :return: A string representation of the manager.
        """
        res = f"File manager of station {self.name} of type {self.__class__.__name__} with cache size {self.cache_size}"
        return res

class WavFilesManager(SoundFilesManager):
    """ Class accounting for .wav files.
    """
    FILE_CLASS = WavFile

class DatFilesManager(SoundFilesManager):
    """ Class accounting for .dat files specific of GEO-OCEAN lab.
    """
    FILE_CLASS = DatFile

    def _process_kwargs(self, kwargs):
        """ Initilize sensitivity.
        :param kwargs: Dict of additional arguments.
        :return: None.
        """
        self.raw = True if kwargs is not None and "raw" in kwargs else False  # default
        self.sensitivity = float(kwargs["sensitivity"]) \
            if kwargs is not None and "sensitivity" in kwargs else -163.5  # default

    def _initialize_sound_file(self, path, skip_data=True, file_number=None):
        """ Read a file and return the corresponding SoundFile instance.
        :param path: Path of the file to load.
        :param skip_data: If True, only read metadata.
        :param file_number: Number of the file to use as ID.
        :return: The SoundFile instance.
        """
        return self.FILE_CLASS(path, self.sensitivity, skip_data=skip_data, identifier=file_number, raw=self.raw)

class WFilesManager(SoundFilesManager):
    """ Class accounting for .w files specific of CTBTO.
    Virtually, we represent 1 record (1 line of .wfdisc) by 1 file (1 instance of WFile).
    """
    FILE_CLASS = WFile

    def _initialize_files(self):
        """ Get the list of the virtual files (nb or records) in the directory.
        :return: None.
        """
        files = glob.glob(self.path + "/wfdisc/*.wfdisc")
        files.sort()  # we assume alphanumerically sorted files are also chronologically sorted
        if len(files) == 0:
            raise Exception(f"No files found in {self.path}/wfdisc")
        vfiles = []
        for file in files:
            with open(file, "r") as f:
                lines = f.readlines()
            lines = [line.split() for line in lines]
            paths = ["/".join(file.split("/")[:-2])+"/"+line[16] for line in lines]
            starts = [datetime.datetime.utcfromtimestamp(float(line[2])) for line in lines]
            ends = [datetime.datetime.utcfromtimestamp(float(line[6])) for line in lines]
            indexes_start = [int(line[17])//4 for line in lines]
            indexes_end = indexes_start[1:] + [None]
            sfs = [float(line[8]) for line in lines]
            cnt_to_upa = [float(line[9]) for line in lines]
            name = [str(line[0]) for line in lines]
            vfiles.extend(zip(paths, starts, ends, indexes_start, indexes_end, sfs, cnt_to_upa, name))
        self.start_dates = np.array([f[1] for f in vfiles])
        # sort files by date
        argsort = np.argsort(self.start_dates)
        self.start_dates = self.start_dates[argsort]
        self.files = np.array(vfiles)[argsort]

    def _getPath(self, file_number):
        """ Given an index of a vfile, return its path.
        :param file_number: The index of the vfile to get.
        :return: The path of the vfile, its dates of start and end, indexes of start and end, sampling f, count to upa
        parameter and name
        """
        return self.files[file_number]

    def _findFirstAndLastFileNumber(self):
        """ Find the indexes of the first and last vfiles of the dataset.
        :return: The indexes (in the files list) of the first and last vfiles of the dataset.
        """
        return 0, len(self.files) - 1

    def _locateFile(self, target_datetime, ref=(None, None), history=None, round_down=True):
        closest = np.argmin(np.abs(target_datetime - self.start_dates))
        if round_down:
            closest = closest - 1 if target_datetime < self.files[closest][1] else closest
        else:
            closest = closest - 1 if target_datetime < self.files[closest][1] and target_datetime < \
                                     self.files[closest - 1][2] else closest
        return closest

def make_manager(path, kwargs):
    """ Looks for the extension of the files in the given directory and returns the correct files manager.
    :param path: The path of the directory we want to load.
    :param kwargs: Other arguments as a dict, that can be used by particular FilesManager (e.g. sensitivity...).
    :return: A FilesManager instance able to load the files of the given directory, or None.
    """
    files = [f.split('.')[-1] for f in os.listdir(path)]
    if WavFile.EXTENSION in files:
        return WavFilesManager(path, kwargs=kwargs)
    if DatFile.EXTENSION in files:
        return DatFilesManager(path, kwargs=kwargs)
    if WFile.EXTENSION in files:
        return WFilesManager(path, kwargs=kwargs)
    print(f"No matching manager found for path {path}")
    return None
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import random
from collections import defaultdict
import re


class Reader(object):
    def __init__(self, dataset_dir, notes_dir=None, listfile=None):
        self._dataset_dir = dataset_dir
        self._notes_dir = notes_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = dataset_dir.join("/listfile.csv")
        else:

            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()


    def read_note(self, nt_filename):
        text = defaultdict(list)
        doc_info = dict()
        (id_patient, _) = nt_filename.split('_')
        with open( self._notes_dir + "/" + id_patient + "/" + nt_filename, "r", encoding='utf8') as ntfile:
        #with open(os.path.join(self._notes_dir, id_patient+'/'+nt_filename), "r") as ntfile:
            tmp_sent = []
            head = True
            for line in ntfile:
                line = line.strip()
                if line:
                    if head:
                        (id_doc, doc_cat, chart_date, chart_time, h, icu_id, h_id) = line.split(',')
                        # read only the notes in the same episode
                        if self.stay_id == icu_id:
                
                            id_doc = int(id_doc)
                            doc_info[id_doc] = (doc_cat, chart_date, chart_time, h, icu_id, h_id)
                            head = False
                        else:
                            head = False
                    else:
                        if self.stay_id == icu_id:
                            tmp_sent.append(line)
                else:
                    if self.stay_id == icu_id:
                        text[id_doc].append(tmp_sent)
                        tmp_sent = []
                        head = True
                    else:
                        head = True

        return (doc_info, text)



    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)



class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, notes_dir=None, listfile=None, period_length=48.0):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, notes_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(n, x, int(y)) for (n, x, y) in self._data]
        self._period_length = period_length
        self.stay_id = None

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)


    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")
        note = self._data[index][0]
        name = self._data[index][1]
        file_name = os.path.splitext(name)[0]
        (subject_id, episode_id, type_episode, stay_id) = file_name.split('_')
        self.stay_id = stay_id
        t = self._period_length
        y = self._data[index][2]
        (doc_info, text) = self.read_note(note)
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "text": text,
                "text_info": doc_info,
                "y": y,
                "header": header,
                "name": name}




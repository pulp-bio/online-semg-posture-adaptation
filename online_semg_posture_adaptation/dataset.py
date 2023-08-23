"""
    Author(s):
    Marcello Zanghieri <marcello.zanghieri2@unibo.it>
    
    Copyright (C) 2023 University of Bologna and ETH Zurich
    
    Licensed under the GNU Lesser General Public License (LGPL), Version 2.1
    (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        https://www.gnu.org/licenses/lgpl-2.1.txt
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from __future__ import annotations

import scipy.io as sio
import numpy as np


NUM_SUBJECTS = 7
NUM_DAYS = 8
NUM_POSTURES = 4

NUM_CLASSES = 6  # rest + 5 gestures
# The number of repetitions is not fixed. File-wise (i.e., for each
# subject-day-posture dataset), the maximum number of repetitions over all
# classes is at least 10 and at most 15.

NUM_CHANNELS = 4
FS_HZ = 500.0  # sampling rate in Hz

# in home filesystem: /home/zanghieri/work/UniboINAIL_BioCAS2019/
# in scratch memory:  /scratch/zanghieri/unibo_inail/data/downloaded/
LOCAL_DATASET_FOLDER = '/scratch/zanghieri/unibo_inail/data/downloaded/'
FILENAME_TEMPLATE = 'DATA_EMG_U%d.mat'  # one file per subject


def subject2filename(idx_subject: int) -> str:
    """
    idx_subject is zero-based, hence {0, ..., 7}
    """
    filename = FILENAME_TEMPLATE % (idx_subject + 1)
    return filename


def load_session(
    idx_subject: int,
    idx_day: int,
    idx_posture: int,
) -> dict:

    """
    Here, "session" refers to a subject-day-posture dataset, even if postures
    were actually acquired in the same physical session, i.e., without
    doffing-donning of the electrodes.
    """

    subject_filename = subject2filename(idx_subject)
    subject_file_full_path = LOCAL_DATASET_FOLDER + subject_filename

    # Each subject's MATLAB workspace only contains one variable, DATA_USER,
    # which is a 1x1 struct (size of the order of 300 MB).
    # The following accesses go to what in MATLAB is stored in:
    #     DATA_USER.days{1, idx_day}.position{1, idx_posture}.emg
    #     DATA_USER.days{1, idx_day}.position{1, idx_posture}.label
    #     DATA_USER.days{1, idx_day}.position{1, idx_posture}.relabel
    #     DATA_USER.days{1, idx_day}.position{1, idx_posture}.gestureCounter

    matlab_workspace = sio.loadmat(subject_file_full_path)
    data_subject = matlab_workspace['DATA_USER']
    del matlab_workspace
    data_day = data_subject['days'].item()[0, idx_day]
    del data_subject
    data_posture = data_day['position'].item()[0, idx_posture]
    del data_day

    # emg
    emg = data_posture['emg'].item()
    emg = np.float32(emg)  # from np.dtype('<f8')
    emg = emg.T  # from (num_samples, num_chnl) to (num_chnl, num_samples)
    # no changes on the values

    # label
    label = data_posture['label'].item()
    label = np.uint8(label)  # should be already np.uint8
    label = label.flatten()  # from (num_samples, 1)
    # change to the values:
    label -= 1  # from {1, ..., 6} to {0, ..., 5}

    # relabel
    relabel = data_posture['relabel'].item()
    relabel = np.uint8(relabel)  # should be already np.uint8
    relabel = relabel.flatten()  # from (num_samples, 1)
    # change to the values:
    relabel -= 1  # from {1, ..., 6} to {0, ..., 5}

    # gesture counter
    gesture_counter = data_posture['gestureCounter'].item()
    gesture_counter = np.uint8(gesture_counter)  # should be already np.uint8
    gesture_counter = gesture_counter.flatten()  # from (num_samples, 1)
    # change to the values: keep 0 as 0, make the other ones zero-based
    # NB: the initial rest, before the first movements, is the only one counted
    # as 0th; giving it to the first movement will assign two rests (i.e.,
    # before and after) to the first repetition of the first performed gesture
    # (which is not always the class 1, i.e. not always the same).
    gesture_counter[gesture_counter > 0] -= 1

    # delete original data container for clarity
    del data_posture

    # checks

    # emg
    assert emg.shape[0] == NUM_CHANNELS

    # label
    assert len(label.shape) == 1
    assert label.min() == 0 and label.max() == 5

    # relabel
    assert len(relabel.shape) == 1
    assert relabel.min() == 0 and relabel.max() == 5

    # gesture_counter
    assert len(gesture_counter.shape) == 1
    assert gesture_counter.min() == 0  # max is not constant over gestures

    # coherence in sample number
    assert emg.shape[1] == len(label) == len(relabel)

    # create dictioary, convenient as return format

    session_data_dict = {
        'emg': emg,
        'label': label,
        'relabel': relabel,
        'gesture_counter': gesture_counter,
    }

    return session_data_dict


def split_into_calib_and_valid(
    emg: np.ndarray[np.float32],
    relabel: np.ndarray[np.uint8],
    gesture_counter: np.ndarray[np.uint8],
    num_calib_repetitions=int,
) -> tuple[
    np.ndarray[np.float32], np.ndarray[np.uint8],
    np.ndarray[np.float32], np.ndarray[np.uint8],
]:

    mask_calib = gesture_counter < num_calib_repetitions
    mask_valid = ~ mask_calib

    # boolean indexing automatically concatenates repetitions
    emg_calib = emg[:, mask_calib]
    emg_valid = emg[:, mask_valid]
    relabel_calib = relabel[mask_calib]
    relabel_valid = relabel[mask_valid]

    return emg_calib, relabel_calib, emg_valid, relabel_valid


def main() -> None:
    pass


if __name__ == '__main__':
    main()

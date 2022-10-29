import pickle as pkl
import numpy as np
import os
from .dataset import Dataset


class flag3d(Dataset):
    dataname = "flag3d"

    def __init__(self, datapath="data/HumanAct12Poses", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        pkldatafilepath = os.path.join(datapath, "flag3d.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))

        self._pose = [x for x in data["poses"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x for x in data["joints3D"]]

        self._actions = [x for x in data["y"]]

        total_num_actions = 60
        self.num_classes = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanact12_coarse_action_enumerator
        self._tr_subjects = [1, 2, 3, 4, 5]
        self._test_subjects = [6, 7, 8, 9, 10]
        self._train = []
        self._test = []
        for index, x in enumerate(data['p']):
            if x in self._tr_subjects:
                self._train.append(index)
            else:
                self._test.append(index)
    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 24, 3)
        return pose

humanact12_coarse_action_enumerator = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "11",
    12: "12",
    13: "13",
    14: "14",
    15: "15",
    16: "16",
    17: "17",
    18: "18",
    19: "19",
    20: "20",
    21: "21",
    22: "22",
    23: "23",
    24: "24",
    25: "25",
    26: "26",
    27: "27",
    28: "28",
    29: "29",
    30: "30",
    31: "31",
    32: "32",
    33: "33",
    34: "34",
    35: "35",
    36: "36",
    37: "37",
    38: "38",
    39: "39",
    40: "40",
    41: "41",
    42: "42",
    43: "43",
    44: "44",
    45: "45",
    46: "46",
    47: "47",
    48: "48",
    49: "49",
    50: "50",
    51: "51",
    52: "52",
    53: "53",
    54: "54",
    55: "55",
    56: "56",
    57: "57",
    58: "58",
    59: "59",
}

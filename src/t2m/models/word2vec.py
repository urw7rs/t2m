import pickle
from os import PathLike
from pathlib import Path
from typing import Optional

import gdown
import numpy as np


POS_ENUM = {
    "VERB": 0,
    "NOUN": 1,
    "DET": 2,
    "ADP": 3,
    "NUM": 4,
    "AUX": 5,
    "PRON": 6,
    "ADJ": 7,
    "ADV": 8,
    "Loc_VIP": 9,
    "Body_VIP": 10,
    "Obj_VIP": 11,
    "Act_VIP": 12,
    "Desc_VIP": 13,
    "OTHER": 14,
}

Loc_list = (
    "left",
    "right",
    "clockwise",
    "counterclockwise",
    "anticlockwise",
    "forward",
    "back",
    "backward",
    "up",
    "down",
    "straight",
    "curve",
)

Body_list = (
    "arm",
    "chin",
    "foot",
    "feet",
    "face",
    "hand",
    "mouth",
    "leg",
    "waist",
    "eye",
    "knee",
    "shoulder",
    "thigh",
)

Obj_List = (
    "stair",
    "dumbbell",
    "chair",
    "window",
    "floor",
    "car",
    "ball",
    "handrail",
    "baseball",
    "basketball",
)

Act_list = (
    "walk",
    "run",
    "swing",
    "pick",
    "bring",
    "kick",
    "put",
    "squat",
    "throw",
    "hop",
    "dance",
    "jump",
    "turn",
    "stumble",
    "dance",
    "stop",
    "sit",
    "lift",
    "lower",
    "raise",
    "wash",
    "stand",
    "kneel",
    "stroll",
    "rub",
    "bend",
    "balance",
    "flap",
    "jog",
    "shuffle",
    "lean",
    "rotate",
    "spin",
    "spread",
    "climb",
)

Desc_list = (
    "slowly",
    "carefully",
    "fast",
    "careful",
    "slow",
    "quickly",
    "happy",
    "angry",
    "sad",
    "happily",
    "angrily",
    "sadly",
)

VIP_dict = {
    "Loc_VIP": Loc_list,
    "Body_VIP": Body_list,
    "Obj_VIP": Obj_List,
    "Act_VIP": Act_list,
    "Desc_VIP": Desc_list,
}


class Word2Vec(object):
    """Glove word embeddings with external dictionary

    returns word embeddings and part-of-speech (POS) tags of words
    """

    def __init__(self, root: PathLike):
        root = Path(root)

        vectors = np.load(root / "our_vab_data.npy")

        with (root / "our_vab_words.pkl").open(mode="rb") as f:
            words = pickle.load(f)

        with (root / "our_vab_idx.pkl").open(mode="rb") as f:
            self.word2idx = pickle.load(f)

        self.word2vec = {w: vectors[self.word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_ENUM))
        if pos in POS_ENUM:
            pos_vec[POS_ENUM[pos]] = 1
        else:
            pos_vec[POS_ENUM["OTHER"]] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split("/")
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec["unk"]
            pos_vec = self._get_pos_ohot("OTHER")
        return word_vec, pos_vec

    @classmethod
    def from_pretrained(cls, path: Optional[PathLike] = None):
        url = "https://drive.google.com/uc?id=1bCeS6Sh_mLVTebxIgiUHgdPrroW06mb6"
        md5 = "c9365026c8c8fc7179794ddd9777fe19"

        if path is None:
            path = Path.cwd()

        output = Path(path).expanduser() / "glove.zip"

        gdown.cached_download(url, str(output), md5=md5, postprocess=gdown.extractall)

        root = output.parent / "glove"
        return cls(root)

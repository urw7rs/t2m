from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Optional

import gdown
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .word2vec import POS_ENUM


@dataclass
class EvaluatorHyperParams:
    dim_pose: int
    dim_word: int = 300
    max_motion_length: int = 196
    dim_pos_ohot: int = len(POS_ENUM)
    dim_motion_hidden: int = 1024
    dim_movement_enc_hidden: int = 512
    dim_movement_latent: int = 512
    max_text_len: int = 20
    dim_text_hidden: int = 512
    dim_coemb_hidden: int = 512
    device: str = "cpu"


def init_weight(m):
    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.ConvTranspose1d)
    ):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=True)
        )

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(
            input_embs, cap_lens, batch_first=True, enforce_sorted=False
        )

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=True)
        )

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(
            input_embs, cap_lens, batch_first=True, enforce_sorted=False
        )

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


def build_models(opt):
    movement_enc = MovementConvEncoder(
        opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent
    )
    text_enc = TextEncoderBiGRUCo(
        word_size=opt.dim_word,
        pos_size=opt.dim_pos_ohot,
        hidden_size=opt.dim_text_hidden,
        output_size=opt.dim_coemb_hidden,
        device=opt.device,
    )

    motion_enc = MotionEncoderBiGRUCo(
        input_size=opt.dim_movement_latent,
        hidden_size=opt.dim_motion_hidden,
        output_size=opt.dim_coemb_hidden,
        device=opt.device,
    )

    return text_enc, motion_enc, movement_enc


class Text2Motion(nn.Module):
    """Pretrained T2M model

    Arguments:
        path (Path): path to extracted checkpoint file
        unit_length (int): unit length, defaults to 4
    """

    def __init__(self, path: Path, unit_length: int = 4):
        super().__init__()

        if isinstance(path, str):
            path = Path(path)

        mean = np.load(path / "VQVAEV3_CB1024_CMT_H1024_NRES3/meta" / "mean.npy")
        mean = mean[None, :]
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))

        std = np.load(path / "VQVAEV3_CB1024_CMT_H1024_NRES3/meta" / "std.npy")
        std = std[None, :]
        self.register_buffer("std", torch.tensor(std, dtype=torch.float))

        dim_pose = mean.shape[-1]

        opt = EvaluatorHyperParams(dim_pose=dim_pose)

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(
            opt
        )

        ckpt_path = path / "text_mot_match" / "model" / "finest.tar"
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))

        self.movement_encoder.load_state_dict(ckpt["movement_encoder"])
        self.text_encoder.load_state_dict(ckpt["text_encoder"])
        self.motion_encoder.load_state_dict(ckpt["motion_encoder"])

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

        self.unit_length = unit_length

    def normalize(self, motion):
        return (motion - self.mean) / self.std

    def denormlize(self, normalized):
        return normalized * self.std + self.mean

    def forward_movement(self, motion):
        movements = self.movement_encoder(motion[:, :, :-4])
        return movements

    def forward_motion(self, movements, lengths):
        m_lens = lengths // self.unit_length
        motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

    def forward_text(self, word_embs, pos_ohot, cap_lens):
        text_embedding = self.text_encoder(
            word_embs, pos_ohot, cap_lens
        )
        return text_embedding

    @classmethod
    def from_pretrained(cls, name:str, path: Optional[PathLike] = None, unit_length: int = 4):
        if name == "HumanML3D":
            url = "https://drive.google.com/uc?id=1o7RTDQcToJjTm9_mNWTyzvZvjTWpZfug"
            md5 = "acee26596c49600983e5fc738028cc5a"
            file = "t2m.zip"
        elif name == "KIT-ML":
            url = "https://drive.google.com/uc?id=1KNU8CsMAnxFrwopKBBkC8jEULGLPBHQp"
            md5 = "bd172c9529ebbdb9074029f15a5430ec"
            file = "kit.zip"
        else:
            raise NotImplementedError()

        if path is None:
            path = Path.cwd()

        output = Path(path).expanduser() / file
        gdown.cached_download(url, str(output), md5=md5, postprocess=gdown.extractall)

        return cls(path=output.with_suffix(""), unit_length=unit_length)

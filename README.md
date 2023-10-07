# t2m: helper library for text-to-motion generation

This repository is a refactored fork of the original [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git) code. It is refactored into a python package so it can be easily integrated into any project.

## Building HumanML3D dataset

download SMPL+H G data for the following datasets from [AMASS](https://amass.is.tue.mpg.de/download.php)

* ACCD (ACCD)
* HDM05 (MPI_HDM05)
* TCDHands (TCD_handMocap)
* SFU (SFU)
* BMLmovi (BMLmovi)
* CMU (CMU)
* Mosh (MPI_mosh)
* EKUT (EKUT)
* KIT  (KIT)
* Eyes_Janpan_Dataset (Eyes_Janpan_Dataset)
* BMLhandball (BMLhandball)
* Transitions (Transitions_mocap)
* PosePrior (MPI_Limits)
* HumanEva (HumanEva)
* SSM (SSM_synced)
* DFaust (DFaust_67)
* TotalCapture (TotalCapture)
* BMLrub (BioMotionLab_NTroje)


download Extended SMPL+H model from [MANO](https://mano.is.tue.mpg.de/login.php) and DMPLs compatible with SMPL from [SMPL](https://smpl.is.tue.mpg.de/download.php)

then run the code bleow
```python
from t2m.datasets import HumanML3D

# search for amass dtasets, smpl+h model, and dmpls from src and build dataset at dst
HumanML3D.build(src="~/Downloads", dst="HumanML3D")
```

## Downloading KIT-ML dataset

```python
from t2m.datasets import KITML

# downloads kit-ml dataset to dst
KITML.download(dst="KIT-ML")
```


## evaluation

```python
from t2m.models import Text2Motion


class EvaluatorMDMWrapper(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()

        if "humanml" in dataset_name or "t2m" in dataset_name:
            name = "HumanML3D"
        elif "kit" in dataset_name:
            name = "KIT-ML"
        else:
            raise NotImplementedError()

        self.t2m = Text2Motion.from_pretrained(name)

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().float()
            pos_ohot = pos_ohot.detach().float()
            motions = motions.detach().float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            """Movement Encoding"""
            movements = self.t2m.forward_movement(motions)
            motion_embedding = self.t2m.forward_motion(movements, m_lens)

            """Text Encoding"""
            text_embedding = self.t2m.forward_text(word_embs.float(), pos_ohot.float(), cap_lens.float())
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            """Movement Encoding"""
            movements = self.t2m.forward_movement(motions)
            motion_embedding = self.t2m.forward_motion(movements, m_lens)
        return motion_embedding
```

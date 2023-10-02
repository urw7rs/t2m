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

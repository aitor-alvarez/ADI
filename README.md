# ADI
Arabic Dialect Identification using intonation patterns and Hybrid BLSTM Resnets networks

## Requirements

The following libraries are required to run the model:

1. Pytorch
2. Torchaudio
3. Numpy


To run the model just execute python main.py


Data from intonation patterns extracted from the VarDial and MGB-3 test set is available at https://github.com/aitor-mir/adi-patterns 

If you intend to use this software for research purposes, please cite the following paper:
```
@inproceedings{Alvarez2020,
  author={Aitor Arronte Alvarez and Elsayed Sabry Abdelaal Issa},
  title={{Learning Intonation Pattern Embeddings for Arabic Dialect Identification}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={472--476},
  doi={10.21437/Interspeech.2020-2906},
  url={http://dx.doi.org/10.21437/Interspeech.2020-2906}
}
```

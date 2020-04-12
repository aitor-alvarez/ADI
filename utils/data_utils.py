import os
import torch
from torchaudio.transforms import MelSpectrogram
import torchaudio

def padding_tensor(sequences):
    """
    input=list of tensors
    """
    num = len(sequences)
    max_len = max([s.size(1) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(1)
        out_tensor[i, :length] = tensor
    return out_tensor


def create_mel_tensors(path):
    audio_dir = [f for f in os.listdir(path) if '.wav' in f]
    direct = './data/'
    for aud in audio_dir:
        pat = aud[0]
        path2 = direct + pat.replace('_', '') + '/'
        waveform, sample_rate = torchaudio.load(os.path.join(path, aud))

        if not os.path.exists(path2):
            os.mkdir(path2)
        mel_spec = MelSpectrogram(sample_rate)(waveform)
        torch.save(mel_spec, path2+ aud.replace('.wav', '.pt'))


def tensor_loader(path):
    sample = torch.load(path)
    return sample

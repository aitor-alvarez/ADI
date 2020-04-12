import torch
import librosa
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils.data_utils import padding_tensor
from torchvision.datasets import DatasetFolder
from torch.utils.data.sampler import SubsetRandomSampler




def load_audio(item):
    wav, sr = librosa.load(item, mono=True)
    return wav


def get_data():
    dataset = DatasetFolder(
        root='./patterns/',
        loader=load_audio,
        extensions='.wav'
    )

    data = [torch.as_tensor(d[0]) for d in dataset]
    data = padding_tensor(data)
    labels = [d[1] for d in dataset]
    targets = torch.as_tensor(labels)
    tensor_dataset = TensorDataset(targets, data)
    valid_size = 0.2
    dataset_size = len(tensor_dataset)
    indices = list(range(dataset_size))
    split = int(valid_size * dataset_size)
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = DataLoader(tensor_dataset,
                             sampler=train_sampler, batch_size=40)
    testloader = DataLoader(tensor_dataset,
                            sampler=test_sampler, batch_size=40)

    return trainloader, testloader
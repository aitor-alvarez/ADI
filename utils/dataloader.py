import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils.data_utils import padding_tensor
from torchvision.datasets import DatasetFolder
from torch.utils.data import SubsetRandomSampler


def load_audio(item):
    wav, sr = torchaudio.load(item)
    return wav


def get_data():
    dataset_test = DatasetFolder(
        root='./patterns_test/',
        loader=load_audio,
        extensions='.wav'
    )

    dataset = DatasetFolder(
        root='./patterns/',
        loader=load_audio,
        extensions='.wav'
    )

    data = [torch.as_tensor(d[0]) for d in dataset]
    data = padding_tensor(data)
    targets = torch.as_tensor(dataset.targets)
    tensor_dataset = TensorDataset(targets, data)
    dataset_size = int(len(tensor_dataset)*0.55)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    #Test
    data_test = [torch.as_tensor(d[0]) for d in dataset_test]
    data_test = padding_tensor(data_test)
    targets_test = torch.as_tensor(dataset_test.targets)
    tensor_dataset_test = TensorDataset(targets_test, data_test)
    dataset_size_test = int(len(tensor_dataset_test))
    indices_test = list(range(dataset_size_test))
    np.random.shuffle(indices_test)
    ####
    train_sampler = SubsetRandomSampler(indices)
    test_sampler = SubsetRandomSampler(indices_test)
    trainloader = DataLoader(tensor_dataset,
                             sampler=train_sampler, batch_size=128)
    testloader = DataLoader(tensor_dataset,
                            sampler=test_sampler, batch_size=128)

    return trainloader, testloader
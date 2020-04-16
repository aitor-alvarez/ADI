from utils.dataloader import get_data
from models.crnn import CRNN
from models import train
import torch

train_data, test_data = get_data()
model = CRNN()
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
model.to(device)
train.train_model(model, train_data)
model.save()
train.test_model(model, test_data)
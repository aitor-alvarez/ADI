from utils.dataloader import get_data
from models.crnn import CRNN
from models.residual import ResidualLSTM, Resblock
from models import train
import torch

train_data, test_data = get_data()
model = CRNN()
#model = ResidualLSTM(Resblock, [2])
if torch.cuda.is_available():
    model = model.cuda()

model_gpu = model
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model_gpu = torch.nn.DataParallel(model).cuda()

train.train_model(model_gpu, train_data)
train.test_model(model_gpu, test_data)
from utils.dataloader import get_data
from models.residual import ResidualLSTM, Resblock
from models import train

train_data, test_data = get_data()
model = ResidualLSTM(Resblock, [2])
train.train_model(model, train_data)
train.test_model(model, test_data)
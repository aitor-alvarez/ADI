from utils.dataloader import get_data
from models.crnn import CRNN
from models import train

train_data, test_data = get_data()
model = CRNN()
train.train_model(model, train_data)

train.test_model(model, test_data)
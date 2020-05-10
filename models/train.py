import torch
from torch.optim import Adam
from utils.data_utils import write_file
import numpy as np



def train_model(model, trainloader):
    learning_rate = 0.001
    num_epochs = 40
    epochs_stop = 5
    min_loss = np.Inf
    epoch_min_loss = np.Inf
    no_improve = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    total_step = len(trainloader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (labels, sounds) in enumerate(trainloader):
            if torch.cuda.is_available():
                labels = labels.cuda()
                sounds = sounds.cuda()
            # Run the forward pass
            outputs = model(sounds)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            if loss<min_loss:
                min_loss=loss
            # Backprop and perform Adam optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 10 in [0, 10]:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
        if min_loss< epoch_min_loss:
            epoch_min_loss=min_loss
            no_improve=0
        else:
            no_improve+=1
        if no_improve==epochs_stop:
            break
        else:
            continue
    loss_list.append('-----'+str(epoch)+'-----')
    write_file('accuracy.txt', acc_list)
    write_file('loss.txt', loss_list)


def test_model(model, testloader):
    correct = 0
    total = 0
    y=[]
    y_predicted=[]
    with torch.no_grad():
        for labels, sounds in testloader:
            if torch.cuda.is_available():
                labels = labels.cuda()
                sounds = sounds.cuda()
            outputs = model(sounds)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_predicted.append(outputs.numpy())
            y.append(labels.numpy())

    print('Accuracy: %d %%' % (100 * correct / total))
    write_file('accuracy_test.txt', [(100 * correct / total)])
    torch.save(model.state_dict(), '.')
    targets = np.asarray(y)
    pred = np.asarray(y_predicted)
    np.save('targets.npy', targets)
    np.save('predicted.npy', pred)
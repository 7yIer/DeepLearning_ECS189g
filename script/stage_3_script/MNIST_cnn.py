import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, f1_score, recall_score



class MyDataset(Dataset):
    def __init__(self, data):
        self.images = []
        self.labels = []
        for item in data:
            self.images.append(item['image'])
            self.labels.append(item['label'])

    def __len__(self):
        return len(self.images)
    def __repr__(self):
        return str(tuple((self.images, self.labels)))

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
f1 = []
precision = []
recall = []
y_predArr = []
y_trueArr = []
with open('../../data/stage_3_data/MNIST', 'rb') as f:
    data = pickle.load(f)

train_dataset = MyDataset(data['train'])
test_dataset = MyDataset(data['test'])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train(self):
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = cnn(inputs.unsqueeze(1).float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                predicted = cnn(inputs.unsqueeze(1).float())
                # calculate accuracy
                _, predicted_labels = torch.max(predicted, 1)
                correct = (predicted_labels == labels).sum().item()
                total = labels.size(0)
                acc = correct / total
                y_true = torch.LongTensor(correct)
                y_pred = self.forward(torch.FloatTensor(predicted))
                f1.append(f1_score(y_true, predicted.max(1)[1], average="weighted"))
                precision.append(precision_score(y_true, y_pred.max(1)[1], average="weighted", zero_division=0))
                recall.append(recall_score(y_true, y_pred.max(1)[1], average="weighted", zero_division=0))


                running_loss += loss.item()
                if i % 100 == 99:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, 10, i + 1, len(train_dataloader), loss.item(), acc * 100))
                    running_loss = 0.0
        print('Finished training')# print results
        print(np.average(f1))
        print(np.average(precision))
        print(np.average(recall))


    cnn.eval()
    # Initialize variables to keep track of predictions and true labels
    all_predictions = []
    all_labels = []

    d = 0
    # Iterate over the test dataloader
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            # Get predictions from the model
            outputs = cnn(inputs.unsqueeze(1).float())
            predictions = torch.argmax(outputs, axis=1)
            d= +1
            # Append predictions and labels to lists
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate accuracy
    accuracy = np.mean(all_predictions == all_labels) * 100
    print("Test accuracy: {:.2f}%".format(accuracy))
    print(d);

cnn = Net()
cnn.forward()
#self.forward()
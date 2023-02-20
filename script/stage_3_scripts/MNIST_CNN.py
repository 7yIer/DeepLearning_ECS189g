import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MyDataset(Dataset):
    def __init__(self, data):
        self.images = []
        self.labels = []
        for item in data:
            self.images.append(item['image'])
            self.labels.append(item['label'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
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
    def train(self, train_dataloader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(50):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.forward(inputs.unsqueeze(1).float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                predicted = self.forward(inputs.unsqueeze(1).float())

                # calculate accuracy
                _, predicted_labels = torch.max(predicted, 1)
                correct = (predicted_labels == labels).sum().item()
                total = labels.size(0)
                acc = correct / total

                running_loss += loss.item()
                if i % 100 == 99:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, 10, i + 1, len(train_dataloader), loss.item(), acc * 100))
                    running_loss = 0.0

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def train(self, train_dataloader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(50):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.forward(inputs.unsqueeze(1).float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                predicted = self.forward(inputs.unsqueeze(1).float())

                # calculate accuracy
                _, predicted_labels = torch.max(predicted, 1)
                correct = (predicted_labels == labels).sum().item()
                total = labels.size(0)
                acc = correct / total

                running_loss += loss.item()
                if i % 100 == 99:
                    print('Epoch: {}, Accuracy: {:.2f}%'
                          .format(epoch + 1, 10, i + 1, len(train_dataloader), loss.item(), acc * 100))
                    running_loss = 0.0

with open('../../data/stage_3_data/MNIST', 'rb') as f:
    data = pickle.load(f)

train_dataset = MyDataset(data['train'])
test_dataset = MyDataset(data['test'])

train_dataloader = DataLoader(train_dataset, batch_size=60000, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False)


CNN = CNN1()
CNN.train(train_dataloader)

# Set model to evaluation mode

# Initialize variables to keep track of predictions and true labels
all_predictions = []
all_labels = []

# Iterate over the test dataloader
with torch.no_grad():
    for inputs, labels in test_dataloader:
        # Get predictions from the model
        outputs = CNN(inputs.unsqueeze(1).float())
        predictions = torch.argmax(outputs, axis=1)

        # Append predictions and labels to lists
        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

# Convert lists to numpy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_predictions) * 100

# Calculate precision, recall, and F1-score
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1_score = f1_score(all_labels, all_predictions, average='weighted')

print("Test accuracy: {:.2f}%".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1_score))

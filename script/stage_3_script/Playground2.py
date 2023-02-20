import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data):
        self.images = []
        self.labels = []
        for item in data:
            img = Image.fromarray(item['image']).convert('L')  # convert to grayscale
            img_tensor = torch.Tensor(np.array(img)).unsqueeze(0)
            self.images.append(img_tensor)
            self.labels.append(item['label'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ORLNet(nn.Module):
    def __init__(self):
        super(ORLNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 25 * 20, 1024)
        self.fc2 = nn.Linear(1024, 41)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 25 * 20)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


with open('../../data/stage_3_data/ORL', 'rb') as f:
    data = pickle.load(f)

train_dataset = MyDataset(data['train'])
test_dataset = MyDataset(data['test'])

train_dataloader = DataLoader(train_dataset, batch_size=360, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=40, shuffle=False)


cnn = ORLNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = cnn(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted = cnn(inputs.float())

        # calculate accuracy
        _, predicted_labels = torch.max(predicted, 1)
        correct = (predicted_labels == labels).sum().item()
        total = labels.size(0)
        acc = correct / total


        running_loss += loss.item()
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, 10, i + 1, len(train_dataloader), loss.item(), acc * 100))
            running_loss = 0.0

            # print results

print('Finished training')

import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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



with open('../../data/stage_3_data/MNIST', 'rb') as f:
    data = pickle.load(f)

train_dataset = MyDataset(data['train'])
test_dataset = MyDataset(data['test'])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)



cnn = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

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


        running_loss += loss.item()
        if i % 100 == 99:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, 10, i + 1, len(train_dataloader), loss.item(), acc * 100))
            running_loss = 0.0

            # print results


print('Finished training')

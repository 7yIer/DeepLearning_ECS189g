from Playground import MyDataset, Net

import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

with open('../../data/stage_3_data/ORL', 'rb') as f:
    data = pickle.load(f)

train_dataset_orl = MyDataset(data['train'])
test_dataset_orl = MyDataset(data['test'])

train_dataloader = DataLoader(train_dataset_orl, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset_orl, batch_size=64, shuffle=False)



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
import os
import glob
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl
import spacy
nlp = spacy.load("en_core_web_sm")
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def clean_text(text):
    # Remove punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    text = ' '.join(words)

    return text


def load_train_data(data_dir):
    pos_files = glob.glob(os.path.join(data_dir, "train", "pos", "*.txt"))
    neg_files = glob.glob(os.path.join(data_dir, "train", "neg", "*.txt"))

    pos_reviews = []
    neg_reviews = []
    for pos_file in pos_files:
        with open(pos_file, "r", encoding="utf-8") as f:
            pos_review = f.read().strip()
            pos_review = clean_text(pos_review)
            pos_reviews.append(pos_review)

    for neg_file in neg_files:
        with open(neg_file, "r", encoding="utf-8") as f:
            neg_review = f.read().strip()
            neg_review = clean_text(neg_review)
            neg_reviews.append(neg_review)
    return pos_reviews, neg_reviews


def load_test_data(data_dir):
    pos_files = glob.glob(os.path.join(data_dir, "test", "pos", "*.txt"))
    neg_files = glob.glob(os.path.join(data_dir, "test", "neg", "*.txt"))

    pos_reviews = []
    neg_reviews = []
    for pos_file in pos_files:
        with open(pos_file, "r", encoding="utf-8") as f:
            pos_review = f.read().strip()
            pos_review = clean_text(pos_review)
            pos_reviews.append(pos_review)

    for neg_file in neg_files:
        with open(neg_file, "r", encoding="utf-8") as f:
            neg_review = f.read().strip()
            neg_review = clean_text(neg_review)
            neg_reviews.append(neg_review)

    return pos_reviews, neg_reviews


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define hyperparameters
batch_size = 900
embedding_dim = 100
hidden_dim = 256
num_layers = 2
learning_rate = 0.005
num_epochs = 5


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_hidden_state = output[:, -1, :]
        logits = self.fc(last_hidden_state)
        output = self.sigmoid(logits)
        return output


data_dir = "/Users/paimannejrabi/Desktop/dd/DeepLearning_ECS189g/data/stage_4_data/text_classification"
train_pos, train_neg = load_train_data(data_dir)
train_data = train_pos + train_neg
train_labels = [1] * len(train_pos) + [0] * len(train_neg)
train_tokenized = []
vocab = {}
word_id = 2
for sentence in train_data:
    sentence = " ".join(sentence)
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    if tokens:
        train_tokenized.append(tokens)
vocab = {'<pad>': 0, '<unk>': 1}
for sentence in train_tokenized:
    for token in sentence:
        if token not in vocab:
            vocab[token] = len(vocab)
train_data_int = []
for sentence in train_tokenized:
    sentence_int = [vocab.get(word, vocab['<unk>']) for word in sentence]
    train_data_int.append(sentence_int)
train_data_padded = nn.utils.rnn.pad_sequence([torch.LongTensor(sentence) for sentence in train_data_int],
                                              batch_first=True)
print("Load Data is completed")

train_dataset = TensorDataset(train_data_padded, torch.FloatTensor(train_labels))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
data_dir = "/Users/paimannejrabi/Desktop/dd/DeepLearning_ECS189g/data/stage_4_data/text_classification"
test_pos, test_neg = load_test_data(data_dir)
test_dataset = test_pos + test_neg
test_dataloader = DataLoader(test_dataset, batch_size=40, shuffle=False)

model = RNN(len(vocab), embedding_dim, hidden_dim, num_layers).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def get_accuracy(output, target, batch_size):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = torch.sum(pred == target).item()
    return correct / batch_size


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        accuracy = get_accuracy(outputs, labels, len(labels))
        loss.backward()
        optimizer.step()
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

print("training completed")

all_predictions = []
all_labels = []
model.eval()


def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


print('Accuracy on test set: {:.2f}%'.format(accuracy))
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
accuracy = accuracy_score(all_labels, all_predictions) * 100
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1_score = f1_score(all_labels, all_predictions, average='weighted')

print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1_score))
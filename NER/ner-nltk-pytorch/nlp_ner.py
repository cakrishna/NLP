# Import necessary libraries
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import nltk
from src.data_preprocessing import load_data, clean_text, tokenize_text
from src.model import NERModel
from src.train import train_model

# Load the data
data = load_data('./data/sample_data.csv')
data['text'] = data['text'].apply(clean_text)
tokenized_data = data['text'].apply(tokenize_text)

# Prepare dataset and dataloaders
class NERDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = NERDataset(tokenized_data, data['labels'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model
model = NERModel()

# Train the model
train_model(model, train_loader, epochs=10)

# Evaluation (to be implemented)
def evaluate_model(model, test_loader):
    # Evaluation logic goes here
    pass

# Save the model
torch.save(model.state_dict(), 'ner_model.pth')
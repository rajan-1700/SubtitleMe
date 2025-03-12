import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

# Example dataset class (customized for LibriSpeech)
class AudioDataset(Dataset):
    def __init__(self, audio_files, labels, transform=None):
        """
        Args:
            audio_files (list): List of paths to audio files
            labels (list): Corresponding labels for each audio file
            transform (callable, optional): Optional transform to apply to the audio
        """
        self.audio_files = audio_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load audio file and apply transformation (e.g., spectrogram or MFCC)
        waveform, sample_rate = torchaudio.load(self.audio_files[idx])
        if self.transform:
            waveform = self.transform(waveform)
        label = self.labels[idx]
        return waveform, label

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)  # Batch size x 1 x Hidden size
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Only take the last time step
        return out

# Hyperparameters
input_size = 1      # Input size (depends on extracted features, such as MFCC or spectrogram)
hidden_size = 128   # Hidden state size of the RNN
output_size = 1     # Output size (depends on your task, e.g., 1 for regression)
num_layers = 1      # Number of RNN layers
learning_rate = 0.001
epochs = 50         # Number of epochs to train
batch_size = 32     # Batch size for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading (LibriSpeech dataset link and loading)
def download_and_prepare_librispeech(data_dir='./data'):
    # Download the LibriSpeech dataset (train-clean-100, for example)
    # Assuming torchaudio is used for LibriSpeech data loading
    dataset = torchaudio.datasets.LIBRISPEECH(root=data_dir, download=True, subset='train-clean-100')
    audio_files = []
    labels = []

    for i in range(len(dataset)):
        audio_files.append(dataset[i][0])  # Get the waveform
        labels.append(dataset[i][2])       # Get the label (for now assuming it's regression or any task)
    
    return audio_files, labels

# Example preprocessing (MFCC extraction)
def preprocess_audio(waveform):
    # Convert the waveform to MFCCs (using torchaudio)
    # captures the spectral characteristics of the audio.
    transform = torchaudio.transforms.MFCC(
        sample_rate=16000, melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23, 'center': False}
    )
    mfcc = transform(waveform)
    return mfcc

# Download and prepare LibriSpeech data
audio_files, labels = download_and_prepare_librispeech()

# Create DataLoader
dataset = AudioDataset(audio_files, labels, transform=preprocess_audio)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = RNNModel(input_size=23, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss (change based on task)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Reshaping input for RNN [batch_size, seq_length, input_size]
        inputs = inputs.transpose(1, 2)  # [batch_size, num_features, seq_length] for RNN

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(dataloader)
    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'rnn_model.pth')

# Plotting the loss curve
plt.plot(range(epochs), train_losses)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epochs')
plt.show()

print("Training completed and model saved.")

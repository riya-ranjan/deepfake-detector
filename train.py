import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model.cnn_lstm import CNN_LSTM_Model  
from model.data_loader import VideoDataset  
import argparse
import os

parser = argparse.ArgumentParser(description="training")
parser.add_argument("--data_root", type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    # Training parameters
    batch_size = 8
    learning_rate = 0.0001
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dir = os.path.join(args.data_root, "train")
    meta_data_dir = os.path.join(args.data_root, "metadata.json")
    train_dataset = VideoDataset(folder_path=train_dir, metadata_path=meta_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = CNN_LSTM_Model().to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (video, audio, label) in enumerate(train_loader):
            # Move data to device
            video = video.to(device)
            audio = audio.to(device)
            label = label.to(device)

            # Forward pass
            outputs = model(video, audio)
            loss = criterion(outputs, label)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), 'cnn_lstm_model.pth')
    print("Model saved to cnn_lstm_model.pth")
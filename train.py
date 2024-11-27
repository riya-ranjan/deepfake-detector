import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model.cnn_lstm import Combined_CNN_LSTM  
from model.data_loader import VideoDataset  
import argparse
import os
import wandb


parser = argparse.ArgumentParser(description="training")
parser.add_argument("--data_root", type=str)

if __name__ == '__main__':

    #setup wandb stuff for logging
    wandb.init(
        # set the wandb project where this run will be logged
        project="cnn-lstm-deepfake",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "architecture": "CNN-LSTM",
        "dataset": "LAV-DF",
        "epochs": 10,
        }
    )
    args = parser.parse_args()
    # Training parameters
    batch_size = 1
    learning_rate = 0.01
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dir = os.path.join(args.data_root, "train")
    meta_data_dir = os.path.join(args.data_root, "metadata.json")
    train_dataset = VideoDataset(folder_path=train_dir, metadata_path=meta_data_dir, data_source="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = Combined_CNN_LSTM(2048, 64).to(device) 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for i, (video, audio, label) in enumerate(train_loader):
            # Move data to device
            video = video.to(device)
            audio = audio.to(device)
            label = label.to(device)
            # Forward pass
            outputs = model(video, audio)
            loss = criterion(outputs, label.float())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == label).float().sum().item()
            total_correct += correct
            total_samples += label.size(0)
        
        running_accuracy = total_correct / total_samples * 100
        wandb.log({"acc": running_accuracy, "loss": running_loss})
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")
    wandb.finish()

    # Save the model
    torch.save(model.state_dict(), 'cnn_lstm_model.pth')
    print("Model saved to cnn_lstm_model.pth")
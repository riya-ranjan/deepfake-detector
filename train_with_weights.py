import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
from collections import Counter
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
        "learning_rate": 0.001,
        "architecture": "CNN-LSTM",
        "dataset": "LAV-DF",
        "epochs": 10,
        }
    )
    args = parser.parse_args()
    # Training parameters
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dir = os.path.join(args.data_root, "train")
    meta_data_dir = os.path.join(args.data_root, "metadata.json")
    train_dataset = VideoDataset(folder_path=train_dir, metadata_path=meta_data_dir, data_source="train")
    
    #use sampler to calibrate model against data imbalance
    label_counts = Counter(train_dataset.labels)
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=10, replacement=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    #weights for loss function
    real_weight = sum(label_counts.values()) / (2 * label_counts[0])
    fake_weight = sum(label_counts.values()) / (2 * label_counts[1])

    # Initialize model, loss function, and optimizer
    model = Combined_CNN_LSTM(2048, 64).to(device) 
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        false_pos = 0
        true_pos = 0
        false_neg = 0

        for i, (video, audio, label) in enumerate(train_loader):
            # Move data to device
            video = video.to(device)
            audio = audio.to(device)
            label = label.to(device)
            # Forward pass
            outputs = model(video, audio)
            loss = criterion(outputs, label)
            weights = torch.tensor([real_weight if l == 0 else fake_weight for l in label])
            weights = weights.to(device)
            weighted_loss = loss * weights  # Apply class weights
            final_loss = weighted_loss.mean() 
            # Backward pass and optimization
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            running_loss += final_loss.item()
            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {final_loss.item():.4f}")
            
            modified_outputs = outputs > 0.5
            total_fake = (label.float() == 1).sum().item()

            tp = ((modified_outputs == 1) & (label.float() == 1)).sum().item()  # True Positives
            fp = ((modified_outputs == 1) & (label.float() == 0)).sum().item()  # False Positives
            fn = ((modified_outputs == 0) & (label.float() == 1)).sum().item()  # False Negatives

            true_pos += tp
            false_pos += fp
            false_neg += fn

            correct = (modified_outputs == label.float()).float().sum().item()
            total_correct += correct
            total_samples += label.size(0)
        
        running_accuracy = total_correct / total_samples
        if true_pos + false_pos != 0:
            running_precision = true_pos / (true_pos + false_pos)
        if true_pos + false_neg != 0:
            running_recall = true_pos / (true_pos + false_neg)
        wandb.log({"acc": running_accuracy, "loss": running_loss, "precision": running_precision, "recall": running_recall})
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")
        path = "./experiments/cnn_lstm_model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), path)
    wandb.finish()

    # Save the model
    print(model.state_dict())
    torch.save(model.state_dict(), "./experiments/cnn_lstm_model.pth")
    print("Model saved to cnn_lstm_model.pth")
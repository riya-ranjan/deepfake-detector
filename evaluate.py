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
parser.add_argument("--model_root", type=str)

def evaluate(model, loss_fn, dataloader, device):
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    false_pos = 0
    true_pos = 0
    false_neg = 0

    for i, (video, audio, label) in enumerate(dataloader):

        video = video.to(device)
        audio = audio.to(device)
        label = label.to(device)

        outputs = model(video, audio)
        loss = loss_fn(outputs, label)

        running_loss += loss.item()
        modified_outputs = outputs > 0.5

        tp = ((modified_outputs == 1) & (label.float() == 1)).sum().item()  # True Positives
        fp = ((modified_outputs == 1) & (label.float() == 0)).sum().item()  # False Positives
        fn = ((modified_outputs == 0) & (label.float() == 1)).sum().item()  # False Negatives

        true_pos += tp
        false_pos += fp
        false_neg += fn

        correct = (modified_outputs == label.float()).float().sum().item()
        total_correct += correct
        total_samples += label.size(0)

        torch.cuda.empty_cache()

    running_accuracy = total_correct / total_samples
    running_precision = true_pos / (true_pos + false_pos)
    running_recall = true_pos / (true_pos + false_neg)
    running_loss /= len(dataloader)
    wandb.log({"acc": running_accuracy, "loss": running_loss, "precision": running_precision, "recall": running_recall})

if __name__ == '__main__':

    #initialize wandb
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dev datasets
    dev_dir = os.path.join(args.data_root, "dev")
    meta_data_dir = os.path.join(args.data_root, "metadata.json")
    dev_dataset = VideoDataset(folder_path=dev_dir, metadata_path=meta_data_dir, data_source="dev")
    
    #use sampler to calibrate model against data imbalance
    label_counts = Counter(dev_dataset.labels)
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in dev_dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=200, replacement=False)

    dev_loader = DataLoader(dev_dataset, batch_size=1, sampler=sampler)

    model = Combined_CNN_LSTM(2048, 64).to(device) 
    state_dict = torch.load(args.model_root)
    model.load_state_dict(state_dict)
    model.eval() #set to evaluation mode
    loss_fn = nn.BCELoss()
    evaluate(model, loss_fn, dev_loader, device)

    

    
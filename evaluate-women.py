import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
from collections import Counter
from model.cnn_lstm import Combined_CNN_LSTM  
from model.data_loader_women import VideoDatasetWomen
import argparse
import os

parser = argparse.ArgumentParser(description="training")
parser.add_argument("--data_root", type=str)
parser.add_argument("--model_root", type=str)
parser.add_argument("--metadata_path", type=str, help="Path to the metadata JSON file")


def evaluate(model, loss_fn, dataloader, device):
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    false_pos = 0
    true_pos = 0
    false_neg = 0

    # Lists to store predictions and labels
    all_predictions = []
    all_labels = []

    print(f"Total batches in dataloader: {len(dataloader)}") 
    for i, (video, audio, label) in enumerate(dataloader):
        video = video.to(device)
        audio = audio.to(device)
        label = label.to(device)

        with torch.no_grad():  
            outputs = model(video, audio)
            loss = loss_fn(outputs, label)
        
        running_loss += loss.item()
        modified_outputs = (outputs > 0.5).float()

        # Store predictions and labels
        all_predictions.extend(modified_outputs.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

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
    
    # Prevent division by zero
    running_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Precision calculation with zero division handling
    running_precision = (true_pos / (true_pos + false_pos)) if (true_pos + false_pos) > 0 else 0
    
    # Recall calculation with zero division handling
    running_recall = (true_pos / (true_pos + false_neg)) if (true_pos + false_neg) > 0 else 0
    
    running_loss /= len(dataloader) if len(dataloader) > 0 else 1

    # Print predictions and labels
    print("\nPredictions vs Labels:")
    for pred, label in zip(all_predictions, all_labels):
        print(f"Prediction: {pred}, Label: {label}")

    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Total Samples: {total_samples}")
    print(f"Total Correct: {total_correct}")
    print(f"Accuracy: {running_accuracy:.4f}")
    print(f"Precision: {running_precision:.4f}")
    print(f"Recall: {running_recall:.4f}")
    print(f"Average Loss: {running_loss:.4f}")

    return {
        'accuracy': running_accuracy,
        'precision': running_precision,
        'recall': running_recall,
        'loss': running_loss
    }
if __name__ == '__main__':

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dev datasets
    dev_dir = os.path.join(args.data_root, "dev")
    meta_data_dir = args.metadata_path

    dev_dataset = VideoDatasetWomen(folder_path=dev_dir, metadata_path=meta_data_dir, data_source="dev")
    
    #use sampler to calibrate model against data imbalance
    label_counts = Counter(dev_dataset.labels)
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in dev_dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=10, replacement=False)

    dev_loader = DataLoader(dev_dataset, batch_size=1, sampler=sampler)

    model = Combined_CNN_LSTM(2048, 64).to(device) 
    state_dict = torch.load(args.model_root, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() #set to evaluation mode
    loss_fn = nn.BCELoss()
    evaluate(model, loss_fn, dev_loader, device)

    

    
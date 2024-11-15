import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

class CNN_LSTM_Branch(nn.Module):
    def __init__(self, input_size):
        super(CNN_LSTM_Branch, self).__init__()
        # Load pretrained Inception v3 model
        self.cnn = models.inception_v3(pretrained=True, aux_logits=False)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Remove the final fully connected layer
        self.resize = transforms.Resize((299, 299))  # Inception v3 requires 299x299 input size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 2)  # Binary output: classes 0 and 1

    def forward(self, x):
        # Resize and reshape input for Inception
        batch_size, sequence_length, c, h, w = x.size()
        x = x.view(batch_size * sequence_length, c, h, w)  # Flatten sequence dimension for CNN
        x = self.resize(x)  # Resize to 299x299 for Inception
        cnn_features = self.cnn(x)  # Extract features with Inception
        cnn_features = cnn_features.view(batch_size, sequence_length, -1)  # Reshape for LSTM input

        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)  # LSTM output
        lstm_out = lstm_out[:, -1, :]  # Take last output of LSTM

        # Pass through fully connected layer
        output = self.fc(lstm_out)  # Output logits
        return F.softmax(output, dim=1)  # Softmax over classes

class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        # Define video and audio branches
        self.video_branch = CNN_LSTM_Branch(input_size=2048)  # Inception v3's output feature size
        self.audio_branch = CNN_LSTM_Branch(input_size=2048)

    def forward(self, video, audio):
        # Get outputs from both branches
        video_output = self.video_branch(video)  # Output shape: (batch_size, 2)
        audio_output = self.audio_branch(audio)  # Output shape: (batch_size, 2)

        # Take the positive class prediction for each branch
        video_pred = video_output[:, 1]  # Probability of class 1 for video
        audio_pred = audio_output[:, 1]  # Probability of class 1 for audio

        # If either prediction is 1, final output is 1
        final_output = torch.max(video_pred, audio_pred)
        return (final_output >= 0.5).float()  # Final binary prediction (0 or 1)

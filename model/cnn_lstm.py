import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchaudio

class CNN_LSTM_Audio(nn.Module):
    def __init__(self, input_size):
        super(CNN_LSTM_Audio, self).__init__()
        # Use a simple CNN for audio feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 2)  # Binary output: classes 0 and 1

    def forward(self, audio_spectrogram):
        batch_size, num_channels, num_mels, num_frames = audio_spectrogram.size()

        # Pass through CNN
        cnn_output = self.cnn(audio_spectrogram)  # (batch_size, num_channels, num_mels, num_frames)
        cnn_output = cnn_output.view(batch_size, num_frames, -1)  # Reshape for LSTM input

        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_output)  # LSTM processes sequence of audio features
        lstm_out = lstm_out[:, -1, :]  # Take the last output of LSTM

        # Pass through fully connected layer
        output = self.fc(lstm_out)  # Output logits
        return F.softmax(output, dim=1)  # Softmax over classes


class CNN_LSTM_Video(nn.Module):
    def __init__(self, input_size):
        super(CNN_LSTM_Video, self).__init__()
        # Load pretrained Inception v3 model without fully connected layers
        self.cnn = models.inception_v3(pretrained=True, aux_logits=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Remove the final fully connected layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 2)  # Binary output: classes 0 and 1

    def forward(self, video_frames):
        batch_size, sequence_length, c, h, w = video_frames.size()
        frame_features = []

        # Process each frame through the CNN individually
        for i in range(sequence_length):
            frame = video_frames[:, i, :, :, :]  # Extract one frame at a time
            cnn_output = self.cnn(frame)  # (batch_size, num_features, 1, 1)
            cnn_output = cnn_output.view(batch_size, -1)  # Flatten the output
            frame_features.append(cnn_output)

        # Stack the CNN features into a sequence (batch_size, sequence_length, num_features)
        cnn_features = torch.stack(frame_features, dim=1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)  # LSTM processes sequence of frame features
        lstm_out = lstm_out[:, -1, :]  # Take the last output of LSTM

        # Pass through fully connected layer
        output = self.fc(lstm_out)  # Output logits
        return F.softmax(output, dim=1)  # Softmax over classes
    
class Combined_CNN_LSTM(nn.Module):
    def __init__(self, video_input_size, audio_input_size):
        super(Combined_CNN_LSTM, self).__init__()
        
        # Create the individual video and audio branches
        self.video_branch = CNN_LSTM_Video(input_size=video_input_size)
        self.audio_branch = CNN_LSTM_Audio(input_size=audio_input_size)

        # Final classification layer that takes both the video and audio features
        self.fc = nn.Linear(2 * 2, 2)  # Concatenated output from video and audio (2 classes each)

    def forward(self, video_frames, audio_spectrogram):
        # Get outputs from the video and audio branches
        video_output = self.video_branch(video_frames)  # Video branch output
        audio_output = self.audio_branch(audio_spectrogram)  # Audio branch output
        
        # Concatenate the outputs from both branches
        combined_output = torch.cat((video_output, audio_output), dim=1)  # Shape: (batch_size, 4)

        # Final classification layer
        final_output = self.fc(combined_output)
        return F.softmax(final_output, dim=1)  # Softmax over classes

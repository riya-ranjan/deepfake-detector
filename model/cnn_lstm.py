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
    def __init__(self, num_classes=1, lstm_hidden_size=128, lstm_num_layers=2):
        super(CNN_LSTM_Audio, self).__init__()
        
        # Audio CNN (2D Convolutional Layers)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Reduce frequency and time resolution
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Further reduction
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # Final reduction
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=1024,  # Match the output channel size of the CNN
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.softmax = nn.Sigmoid()  # Binary classification
        
    def forward(self, x):
        # Input shape: (batch_size, 1, mel_bins, time_steps)
        
        # Pass through CNN
        cnn_out = self.cnn(x)  # Shape: (batch_size, 128, reduced_mel_bins, reduced_time_steps)
        cnn_out = cnn_out.permute(0, 3, 1, 2)  # Rearrange to (batch_size, time_steps, channels, mel_bins)
        
        # Flatten frequency dimension for LSTM
        batch_size, time_steps, channels, mel_bins = cnn_out.shape
        cnn_out = cnn_out.reshape(batch_size, time_steps, -1)  # Shape: (batch_size, time_steps, channels * mel_bins)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_out)  # Shape: (batch_size, time_steps, lstm_hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step's output
        
        # Pass through fully connected layer and activation
        output = self.fc(lstm_out)  # Shape: (batch_size, num_classes)
        output = self.softmax(output)
        
        return output

class CNN_LSTM_Video(nn.Module):
    def __init__(self, input_size):
        super(CNN_LSTM_Video, self).__init__()
        # Load Inception v3 model without fully connected layers
        self.cnn = models.inception_v3(pretrained=False, aux_logits=False)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Remove the final fully connected layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 1) 
        self.softmax = nn.Sigmoid()

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
        output = self.softmax(output)
        return output # Softmax over classes
    
class Combined_CNN_LSTM(nn.Module):
    def __init__(self, video_input_size, audio_input_size):
        super(Combined_CNN_LSTM, self).__init__()
        
        # Create the individual video and audio branches
        self.video_branch = CNN_LSTM_Video(input_size=video_input_size)
        self.audio_branch = CNN_LSTM_Audio()

    def forward(self, video_frames, audio_spectrogram):
        # Get outputs from the video and audio branches
        video_output = self.video_branch(video_frames)  # Video branch output
        audio_output = self.audio_branch(audio_spectrogram)  # Audio branch output
        print("video output")
        print(video_output)
        print("audio output")
        print(audio_output)
        # Concatenate the outputs from both branches
        combined_output =  0.5 * video_output + 0.5 * audio_output

        return combined_output

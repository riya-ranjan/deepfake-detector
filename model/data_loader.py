import torch
from torch.utils.data import Dataset
import os
import json
import torchaudio
import torchvision.transforms as transforms
from torchvision.io import read_video

class VideoDataset(Dataset):
    def __init__(self, folder_path, metadata_path, transform=None, audio_length=16000):
        self.folder_path = folder_path
        self.transform = transform
        self.metadata = self._load_metadata(metadata_path)
        self.video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        self.audio_length = audio_length  # Number of samples for a fixed audio length

        # Define video transform for Inception input size
        self.video_transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize frames to 299x299
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Inception normalization
        ])

        # Define audio transform to mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram()

    def _load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.folder_path, video_file)

        # Load video frames and audio
        video, audio, info = read_video(video_path, pts_unit='sec')
        
        # Process video frames
        video_frames = [self.video_transform(frame) for frame in video]
        video_tensor = torch.stack(video_frames)  # Shape: (num_frames, 3, 299, 299)

        # Process audio: Convert to mel spectrogram and ensure fixed length
        audio_waveform, sample_rate = audio, info['audio_fps']
        audio_waveform = self._pad_or_truncate_audio(audio_waveform)
        mel_spectrogram = self.mel_transform(audio_waveform)  # Shape: (num_mels, num_frames)

        # Get label from metadata
        label = self.metadata.get(video_file, {}).get('n_fakes', 0)
        if label > 0:
            label = 1

        return video_tensor, mel_spectrogram, label

    def _pad_or_truncate_audio(self, audio_waveform):
        # Pad or truncate audio waveform to ensure a fixed length
        if audio_waveform.size(1) > self.audio_length:
            return audio_waveform[:, :self.audio_length]
        else:
            padding = self.audio_length - audio_waveform.size(1)
            return torch.nn.functional.pad(audio_waveform, (0, padding))

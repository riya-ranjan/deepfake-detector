import torch
from torch.utils.data import Dataset
import os
import json
import torchaudio
import torchvision.transforms as transforms
from torchvision.io import read_video

class VideoDataset(Dataset):
    def __init__(self, folder_path, metadata_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.metadata = self._load_metadata(metadata_path)
        self.video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

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

        # Process audio: convert to mel spectrogram
        audio_waveform, sample_rate = audio, info['audio_fps']
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(audio_waveform)

        # Transform video frames if transform is specified
        if self.transform:
            video = self.transform(video)

        # Get label from metadata
        label = self.metadata.get(video_file, {}).get('n_fakes', 0)
        if label > 0:       # for our purposes, we don't care much about the number of fakes -- just whether or not it is fake
            label = 1

        return video, mel_spectrogram, label

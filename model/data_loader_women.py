import torch
from torch.utils.data import Dataset
import os
import json
import torchaudio
import torchvision.transforms as transforms
from torchvision.io import read_video

class VideoDatasetWomen(Dataset):
    def __init__(self, folder_path, metadata_path, data_source, audio_length=int(512/25*16000)):
        self.folder_path = folder_path

        self.metadata = self._load_metadata(metadata_path)

        self.data_source = data_source
        self.video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        self.audio_length = audio_length
        self.max_frames = 512

        # We want the mel spectrogram to have 64 features
        self.mel_transform = torchaudio.transforms.MelSpectrogram(n_mels=64)
        # Create label array to use for weighted sampling
        self.true_labels = []
        for file in self.video_files:
            metadata_entry = next((item for item in self.metadata if item['filename'] == file), None)
            self.true_labels.append(metadata_entry.get('label', 0))
        self.labels = [1 if label > 0 else 0 for label in self.true_labels]

    def _load_metadata(self, metadata_path, data_source="dev"):
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        # Ensure the data source exists
        if data_source not in all_metadata:
            raise ValueError(f"Data source {data_source} not found in metadata. Available sources: {list(all_metadata.keys())}")
        
        # Get metadata for specific split (train/dev/test)
        metadata = all_metadata[data_source]
        
        print(f"Loaded {len(metadata)} entries for {data_source}")
        return metadata

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.folder_path, video_file)
        video, audio, info = read_video(video_path, pts_unit='sec')

        if video.shape[0] > self.max_frames:
            video = video[:self.max_frames]  # Truncate
        elif video.shape[0] < self.max_frames:
            padding = torch.zeros(self.max_frames - video.shape[0], *video.shape[1:])  # Pad
            video = torch.cat([video, padding], 0)
    
        # Remove alpha channel if present
        if video.shape[-1] > 3:
            video = video[:, :, :, :3]

        video_tensor = video.permute(0, 3, 1, 2)  # Convert to shape (num_frames, 3, height, width)

        audio_waveform, sample_rate = audio, info['audio_fps']
        audio_waveform = self._pad_or_truncate_audio(audio_waveform)
        mel_spectrogram = self.mel_transform(audio_waveform)  # Shape: (num_mels, num_frames)

        # Get label from metadata
        metadata_entry = next((item for item in self.metadata if item['filename'] == video_file), None)
        
        # If metadata is found, extract the label and other information
        if metadata_entry is not None:
            label = metadata_entry.get('n_fakes', 0)  # Default to 0
        else:
            label = 0  # Default label in case metadata is missing

        if label > 0:
            label = 1

        label = float(label)
        label = torch.tensor(label).unsqueeze(0)

        return video_tensor, mel_spectrogram, label

    def _pad_or_truncate_audio(self, audio_waveform):
        if audio_waveform.size(1) > self.audio_length:
            return audio_waveform[:, :self.audio_length]
        else:
            padding = self.audio_length - audio_waveform.size(1)
            return torch.nn.functional.pad(audio_waveform, (0, padding))
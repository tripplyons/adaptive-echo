import os
import re

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class WavDataset(Dataset):
    def __init__(self, file, transform=None, target_sample_rate=None):
        """
        Args:
            csv (str): Path to csv containing .wav files and parameters.
            transform (callable, optional): Optional transform to apply on waveform.
            target_sample_rate (int, optional): Resample to this rate if provided.
        """
        self.file = file
        self.transform = transform
        self.target_sample_rate = target_sample_rate

        self.wav_files = dict()

        with open(file, "r") as f:
            for line in f:
                sep = line.split(",")
                self.wav_files[sep[0]] = torch.Tensor([float(s) for s in sep[1:]])

        self.file_ids = sorted(
            self.wav_files.keys(), key=lambda x: self._extract_index(x)
        )

        # Collect all .wav files
        # self.wav_files = sorted(
        #     [f for f in os.listdir(directory) if f.endswith('.wav')],
        #     key=self._extract_index  # sort by index
        # )

    def _extract_index(self, filename):
        """Extract numeric index from filename (e.g., 'audio_23.wav' -> 23)."""
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else -1

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        wav_path = os.path.join(self.directory, self.wav_files[file_id])
        waveform, sample_rate = torchaudio.load(wav_path)

        # Resample if needed
        if self.target_sample_rate and sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate

        # Apply optional transform (e.g., normalization, augmentation)
        if self.transform:
            waveform = self.transform(waveform)

        # Return both waveform and index (parsed from filename)
        # index = self._extract_index(self.wav_files[idx])
        return waveform, self.wav_files[file_id]

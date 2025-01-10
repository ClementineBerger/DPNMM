"""
Class of dataloader for evaluation
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import numpy as np

from utils.masking_thresholds import MaskingThresholds


class MusicNoiseDataset(Dataset):
  def __init__(
          self,
          root_dir,
          csv_file,
          nfft,
          sr,
          set,
          hpss_music,
          hpss_noise):

    self.root_dir = root_dir
    # set_metadata, files_id = load_set(set, csv_file)
    self.metadata = pd.read_csv(csv_file)
    self.metadata = self.metadata[self.metadata['set'] == set]
    self.nfft = nfft
    self.sr = sr

    # global amplitude maximum in the dataset.
    self.global_maximum_amplitude = 43048712.0

    self.masking_thresholds = MaskingThresholds(
        nfft=self.nfft,
        sr=self.sr
    )

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    music_path = os.path.join(
        self.root_dir,
        self.metadata['music_path'].iloc[idx]
    )

    noise_path = os.path.join(
        self.root_dir,
        self.metadata['noise_path'].iloc[idx]
    )

    music_waveform, _ = torchaudio.load(uri=music_path)
    noise_waveform, _ = torchaudio.load(uri=noise_path)

    max_amplitude_couple = self.metadata['max_amplitude_couple'].iloc[idx]

    music_waveform = max_amplitude_couple * music_waveform / self.global_maximum_amplitude
    noise_waveform = max_amplitude_couple * noise_waveform / self.global_maximum_amplitude

    return music_waveform.squeeze(0), noise_waveform.squeeze(0), idx

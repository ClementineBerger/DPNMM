"""
Dataloader for the music_noise dataset.
"""


import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchaudio

import numpy as np

from utils.analysis import stft
from utils.masking_thresholds import MaskingThresholds
from utils.utils import linear2dB


class MusicNoiseDataset(Dataset):
  def __init__(
          self,
          root_dir,
          csv_file,
          nfft,
          sr,
          set):

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

    # set path list
    self.musics_path_list = [
        os.path.join(self.root_dir, music_path)
        for music_path in self.metadata['music_path'].values]
    self.noises_path_list = [
        os.path.join(self.root_dir, noise_path)
        for noise_path in self.metadata['noise_path'].values]

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    # if torch.is_tensor(idx):
    #   idx = idx.tolist()

    music_path = self.musics_path_list[idx]
    noise_path = self.noises_path_list[idx]

    music_waveform, _ = torchaudio.load(uri=music_path)
    noise_waveform, _ = torchaudio.load(uri=noise_path)

    max_amplitude_couple = self.metadata['max_amplitude_couple'].iloc[idx]

    music_waveform = max_amplitude_couple * music_waveform / self.global_maximum_amplitude
    noise_waveform = max_amplitude_couple * noise_waveform / self.global_maximum_amplitude

    music_stft = stft(audio=music_waveform, nfft=self.nfft, overlap=0.75)
    noise_stft = stft(audio=noise_waveform, nfft=self.nfft, overlap=0.75)

    mTbark = self.masking_thresholds.compute_thresholds(abs(music_stft) ** 2)
    music_bark = self.masking_thresholds.convert_hz2bark(
        abs(music_stft) ** 2)

    noise_bark = self.masking_thresholds.convert_hz2bark(
        abs(noise_stft) ** 2)

    input_cnn = torch.vstack(
        (
            linear2dB(music_bark, 10),
            linear2dB(noise_bark, 10),
            linear2dB(mTbark, 10),
        )
    )

    return input_cnn, (music_stft.squeeze(), music_waveform, noise_waveform)


def main():
  root_dir = os.environ['DATA']
  music_noise_dir = os.path.join(root_dir, "music_noise")
  csv_file = os.path.join(music_noise_dir, "metadata.csv")

  dataset = MusicNoiseDataset(
      root_dir=root_dir,
      csv_file=csv_file,
      nfft=2048,
      sr=44100,
      set='train')
  print(dataset.__len__())

  input_cnn, others = dataset.__getitem__(idx=23)

  music_spectrum = others[0]
  music_harmonic = others[1]

  print(input_cnn.shape)
  print(music_spectrum.shape)
  print(music_harmonic.shape)


if __name__ == "__main__":
  main()

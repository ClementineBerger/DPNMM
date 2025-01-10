"""
DDSP functions : Bark scale, Yule-Walker, AR filter, etc.
"""

import torch
import torch.nn as nn
from utils.utils import linear2dB


def rfft(x: torch.Tensor, nfft: int):
  """Compute rfft of the input x with nfft frequential resolution."""
  return torch.fft.rfft(x * torch.hann_window(nfft).reshape(1, -1), n=nfft)


def fftfreq(nfft: int, sr: int):
  """Computes the sample frequencies for rfft() with a signal of size n."""
  return torch.fft.rfftfreq(n=nfft, d=1 / sr)


def spectral_density(x: torch.Tensor,
                     nfft: int):
  """
  Compute the power spectral density of the signal x.

  Parameters
  ----------
  x : torch.Tensor
      Signal tensor
  nfft : int
      Frequency resolution
  """

  return abs(rfft(x, nfft))**2


def hz2bark(freq: torch.Tensor):
  """
  Conversion of frequency in Hz to the Bark scale.

  Parameters
  ----------
  freq : torch.Tensor
      Frequencies in Hz.

  Returns
  -------
  torch.Tensor
      Bark frequencies
  """
  bark = 13 * torch.arctan(0.76 * freq / 1000) + 3.5 * \
      torch.arctan(freq / 7500) ** 2
  return bark


def bark_index(freq: torch.Tensor):
  """
  Computation of critical band index from the frequencies tensor in Hz.

  Parameters
  ----------
  freq : torch.Tensor
      Frequencies in Hz.

  Returns
  -------
  torch.IntTensor, torch.IntTensor
      Lower and upper indexes of the critical bands in the frequencies tensor.
  """
  bark = hz2bark(freq)
  low_bark = [0]
  for i in range(1, len(bark)):
    if torch.floor(bark[i]) > torch.floor(bark[i - 1]):
      low_bark.append(i)
  low_bark = torch.IntTensor(low_bark)
  high_bark = torch.zeros_like(low_bark)
  high_bark[:-1] = low_bark[1:] - 1
  high_bark[-1] = len(bark) - 1

  return low_bark, high_bark

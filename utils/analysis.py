"""Utils from spectral analysis : STFT, frequency weightings etc..."""

import torch
import torchaudio.transforms as T

from utils.utils import safe_log


def pad_for_stft(audio, nfft, hop_size):
  """
  Padding function for stft.

  Parameters
  ----------
  audio : torch.Tensor [batch_size, nb_timesteps]
      Audio waveforms in batch
  nfft : _type_
      _description_
  hop_size : _type_
      _description_
  """

  audio_size = audio.shape[1]
  num_frames = audio_size // hop_size

  pad_size = max(0, nfft + (num_frames - 1) * hop_size - audio_size)

  if pad_size == 0:
    return audio

  else:
    audio = torch.nn.functional.pad(audio, pad=(0, pad_size))
    return audio


def stft(
        audio: torch.Tensor,
        nfft: int,
        overlap: float,
        window_size=None,
        center=True,
        pad_end=True):
  """
  Differentiable stft in pytorch, computed in batch.

  Parameters
  ----------
  audio : torch.Tensor
      Batch of mono waveforms [n_batch, nfft]
  nfft : int
      Size of Fourier transform.
  overlap : float
      Portion of overlapping window
  center : bool, optional
      by default False
  pad_end : bool, optional
      Padding applied to the audio or not, by default True

  Returns
  -------
  torch.Tensor
      stft [batch_size, nfft//2 + 1, n_frames]
  """

  hop_size = int(nfft * (1. - overlap))

  if pad_end:
    audio = pad_for_stft(audio=audio, nfft=nfft, hop_size=hop_size)
  # pb du center et de la istft à régler, est-ce que ça change qqch pour le
  # padding ?

  if window_size is None:
    window_size = nfft

  window = torch.hann_window(window_size).to(device=audio.device)

  spectrogram = torch.stft(
      input=audio,
      n_fft=nfft,
      hop_length=hop_size,
      win_length=window_size,
      window=window,
      center=center,
      return_complex=True
  )

  return spectrogram.transpose(-2, -1)


def istft(
        stft: torch.Tensor,
        nfft: int = 2048,
        overlap=0.75,
        center=True,
        length=441000):
  """Differentiable istft in PyTorch, computed in batch."""

  # input stft [batch_size, n_frames, nfft//2 + 1], need to transpose the
  # time and frequency dimensions

  stft = stft.transpose(-2, -1)
  hop_length = int(nfft * (1.0 - overlap))

  assert nfft * overlap % 2.0 == 0.0
  window = torch.hann_window(int(nfft), device=stft.device)
  s = torch.istft(
      input=stft,
      n_fft=int(nfft),
      hop_length=hop_length,
      win_length=int(nfft),
      window=window,
      center=center,
      length=length,
      onesided=True,
      return_complex=False)
  return s


def slidding_mean_power(power_spec, hop_size, window_size, centered=False):
  """
  Compute the sliding mean power of a power spectrum.

  This function calculates the mean power of a given power spectrum using a sliding window approach.
  It computes both the linear and logarithmic mean power, then averages them to obtain the final mean power.

  Args:
    power_spec (torch.Tensor): The input power spectrum tensor.
    hop_size (int): The hop size for the sliding window.
    window_size (int): The size of the sliding window.
    centered (bool, optional): If True, the window is centered. Defaults to False.

  Returns:
    torch.Tensor: The sliding mean power tensor.
  """

  mean_lin = 10 * safe_log(
      torch.mean(power_spec, dim=2)
  )

  mean_lin_frames = mean_lin.unfold(
      dimension=1,
      size=window_size,
      step=hop_size
  )

  mean_log = torch.mean(
      10 * safe_log(power_spec), dim=2
  )

  mean_log_frames = mean_log.unfold(
      dimension=1,
      size=window_size,
      step=hop_size
  )

  mean_power = .5 * (torch.mean(mean_lin_frames,
                     dim=2) + torch.mean(mean_log_frames, dim=2))

  return mean_power


def weighting_function(freq):
  """
  A-weighting function used to compute level in dBA.

  Parameters
  ----------
  freq : torch.Tensor
      Frequencies
  """
  num = (12194**2) * (freq**4)
  denom = ((freq**2) + (20.6**2)) * \
      ((freq**2) + (12194**2)) *\
      torch.sqrt(((freq**2) + (107.7**2))) * \
      torch.sqrt(((freq**2) + (737.9**2)))

  return num / denom


def a_weightings(freq):
  """
  Normalized weighting function.

  Parameters
  ----------
  freq : torch.Tensor
      Frequencies

  Returns
  -------
  float or torch.Tensor
      Normalized A-weighting function
  """

  weights = weighting_function(
      freq) / weighting_function(torch.Tensor([1000]).to(freq.device))

  return weights


def apply_weights_to_spectrum(spectrum, weights):
  """
  Apply weights to a given spectrum.

  This function multiplies the input spectrum by the provided weights. The input spectrum can be a 1D, 2D, or 3D tensor.
  The weights are reshaped to match the dimensions of the spectrum before multiplication.

  Parameters:
  ----------
  spectrum (torch.Tensor): The input spectrum tensor. It can have 1, 2, or 3 dimensions.
  weights (torch.Tensor): The weights tensor to be applied to the spectrum. The shape of the weights must match the
              frequency dimension of the spectrum.

  Returns:
  -------
  torch.Tensor: The weighted spectrum.

  Raises:
  -------
  ValueError: If the input spectrum has more than 3 dimensions or if the number of weights does not match the
        frequency dimension of the input spectrum.
  """

  if spectrum.ndim == 3:
    batch_size, nframe, nfreq = spectrum.shape
    weights = weights.reshape(1, 1, -1)
  elif spectrum.ndim == 2:
    nframe, nfreq = spectrum.shape
    weights = weights.reshape(1, -1)
  elif spectrum.ndim == 1:
    nfreq = spectrum.shape[0]
    weights = torch.flatten(weights)
  else:
    raise ValueError(
        f"Wrong shape for input data, expected at least 1 dim and maximum 3, got {spectrum.shape} .")

  if weights.shape[-1] != nfreq:
    raise ValueError(
        f"Number of weights ({weights.shape[-1]}) doesn't match the size of frequency dimension in input ({nfreq}).")

  weighted_spectrum = spectrum * weights

  return weighted_spectrum


def compute_rms_level_from_spectrum(spectrum):
  """
  Compute the Root Mean Square (RMS) level from a given spectrum.

  Parameters:
  ----------
  spectrum (torch.Tensor): Input tensor representing the spectrum.
               It can have 1, 2, or 3 dimensions:
               - 1D: (nfreq,)
               - 2D: (nframe, nfreq)
               - 3D: (batch_size, nframe, nfreq)

  Returns:
  -------
  torch.Tensor or float: The computed RMS level. The shape of the output depends on the input:
               - If input is 3D, returns a tensor of shape (batch_size, nframe)
               - If input is 2D, returns a tensor of shape (nframe,)
               - If input is 1D, returns a float

  Raises:
  -------
  ValueError: If the input tensor has more than 3 dimensions or less than 1 dimension.
  """

  if spectrum.ndim == 3:
    batch_size, nframe, nfreq = spectrum.shape
  elif spectrum.ndim == 2:
    nframe, nfreq = spectrum.shape
    spectrum.reshape((1, nframe, nfreq))
  elif spectrum.ndim == 1:
    nfreq = spectrum.shape[0]
    spectrum.reshape((1, 1, nfreq))
  else:
    raise ValueError(
        f"Wrong shape for input data, expected at least 1 dim and maximum 3, got {spectrum.shape} .")

  rms_level = (1 / (2 * nfreq - 2)) * torch.sqrt(
      2 * torch.sum(abs(spectrum)**2, dim=-1) - abs(spectrum[:, :, 0])**2
  )

  if spectrum.ndim == 3:
    return rms_level
  elif spectrum.ndim == 2:
    rms_level.reshape(nframe)
  elif spectrum.ndim == 1:
    rms_level = rms_level[0, 0]

  return rms_level

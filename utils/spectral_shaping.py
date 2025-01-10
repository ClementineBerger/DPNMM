"""
Spectral shaping modules : cepstrum computation, target filter computation.
"""

import torch
import torch.nn as nn
import utils.ddsp as ddsp
from utils.utils import linear2dB, to_device
from utils.ddsp import apply_filter_in_complex_domain


def cos_window(n, N):
  pi = 3.14159265359
  return 0.5 * (1 - torch.cos(2 * pi * n / N / 2))


def patterns_matrix(low_bark, nfft):
  """
  Creates a Tensor containing the spectral pattern in dB per bark bands for generating the target filters.

  Parameters
  ----------
  low_bark : torch.Tensor
    A tensor containing the lower bark band edges. Shape: (n_bark,)
  nfft : int
    The number of FFT points.

  Returns
  -------
  torch.Tensor
    A tensor containing the spectral patterns. Shape: (n_bark, nfft // 2 + 1)
  """

  n_bark = low_bark.size(0)

  patterns = torch.zeros((n_bark, nfft // 2 + 1)).float()

  for i in range(n_bark):
    if i == n_bark - 1:
      patterns[i, low_bark[i]:] = 1.0
    elif i == 0:
        # compensate the fact that the first bark band has no adjacent bands below
      # it
      patterns[i, low_bark[i]: low_bark[i + 1]] = 2.0

    else:
      patterns[i, low_bark[i]: low_bark[i + 1]] = 1.0

    if i > 1:
      n = torch.arange(low_bark[i] - low_bark[i - 2])
      N = n.size(0)
      patterns[i, low_bark[i - 2]: low_bark[i]] = cos_window(n, N)

    if i < n_bark - 3:
      n = torch.arange(low_bark[i + 3] - low_bark[i + 1])
      N = n.size(0)
      if i == 0:
          # compensate the fact that the first bark band has no adjacent bands below
        # it
        patterns[i, low_bark[i + 1]: low_bark[i + 3]
                 ] = 2. * cos_window(N - n, N)
      else:
        patterns[i, low_bark[i + 1]: low_bark[i + 3]] = cos_window(N - n, N)

    if i == n_bark - 3:
      n = torch.arange((nfft // 2 + 1) - low_bark[i + 1])
      N = n.size(0)
      patterns[i, low_bark[i + 1]:] = cos_window(N - n, N)

    patterns[1, low_bark[0]: low_bark[1]] = 1.0
    patterns[n_bark - 2, low_bark[n_bark - 2 + 1]:] = 1.0

  return patterns


class SpectralEnvelope:
  """
  A class used to compute the spectral envelope of a given spectrum.

  Parameters
  ----------
  nfft : int
    The number of FFT points.
  sr : int
    The sampling rate of the audio signal.
  order : int
    The order of the cepstrum used for computing the spectral envelope.

  Attributes
  ----------
  nfft : int
    The number of FFT points.
  sr : int
    The sampling rate of the audio signal.
  order : int
    The order of the cepstrum used for computing the spectral envelope.
  fftfrequencies : torch.Tensor
    The FFT frequencies computed using the given nfft and sr.

  Methods
  -------
  compute_cepstrum(spectrum)
    Computes the cepstrum of the given spectrum.
  compute_spectral_envelope(spectrum)
    Computes the spectral envelope based on the cepstrum computation.
  """

  def __init__(self, nfft: int, sr: int, order: int):
    self.nfft = nfft
    self.sr = sr
    self.order = order
    self.fftfrequencies = ddsp.fftfreq(nfft=self.nfft, sr=self.sr)

  def compute_cepstrum(self, spectrum: torch.Tensor):
    """
    Compute the cepstrum of a given spectrum.

    Parameters
    ----------
    spectrum : torch.Tensor
      The input spectrum as a tensor.

    Returns
    -------
    torch.Tensor
      The computed cepstrum.
    """
    log_spectrum = torch.log(spectrum)
    cepstrum = torch.fft.irfft(
        log_spectrum,
        dim=1)
    return torch.real(cepstrum)

  def compute_spectral_envelope(self, spectrum):
    """
    Compute the spectral envelope of the given spectrum.

    Parameters
    ----------
    spectrum : torch.Tensor
      Input spectrum tensor of shape [batch_size, n_frames, nfft//2 + 1].

    Returns
    -------
    torch.Tensor
      Spectral envelope tensor of shape [batch_size, n_frames, nfft//2 + 1].
    """

    spectrum = spectrum.view(-1, spectrum.shape[-2], spectrum.shape[-1])

    # size of input : [batch_size, n_frames, nfft//2 + 1]
    batch_size, n_frames = spectrum.shape[-3], spectrum.shape[-2]

    spectrum = spectrum.reshape(-1, spectrum.shape[-1])

    real_cepstrum = self.compute_cepstrum(spectrum)
    mask = torch.ones_like(real_cepstrum)
    mask[:, self.order:-self.order] = 0.
    envelope = real_cepstrum * mask
    envelope = torch.real(torch.fft.rfft(envelope, dim=1))
    envelope = torch.exp(envelope)

    envelope = envelope.reshape(batch_size, n_frames, -1)

    return envelope


class TargetFilter(nn.Module):
  """
  Module designed to apply a target filter to an input signal.
  It computes the target spectral envelope and applies it to the input signal to achieve the desired frequency response.

  nfft : int
    Number of FFT points.
  sr : int
    Sampling rate of the input signal.
  filter_order : int
    Order of the filter to be applied.
  old_wrapper : bool, optional
    Flag to determine whether to use the old wrapper for patterns matrix (default is False).

  Attributes
  ----------
  nfft : int
    Number of FFT points.
  sr : int
    Sampling rate of the input signal.
  filter_order : int
    Order of the filter to be applied.
  fftfrequencies : torch.Tensor
    FFT frequencies computed using ddsp.fftfreq.
  low_bark : torch.Tensor
    Lower bark index computed using ddsp.bark_index.
  high_bark : torch.Tensor
    Higher bark index computed using ddsp.bark_index.
  patterns : torch.Tensor
    Patterns matrix used for computing gains.

  Methods
  -------
  compute_target_filter(gains)
    Computes the target filter from the predicted gains.
  compute_target_envelope(input_envelope, gains)
    Computes the target spectral envelope from the initial input envelope and the predicted gains.
  target_frequency_response(target_envelope, input_envelope)
    Computes the target frequency response from the target and input envelopes.
  """

  def __init__(self, nfft: int, sr: int, filter_order: int, old_wrapper=False):
    super(TargetFilter, self).__init__()

    self.nfft = nfft
    self.sr = sr
    self.filter_order = filter_order
    self.fftfrequencies = ddsp.fftfreq(
        nfft=self.nfft, sr=self.sr)
    low_bark, high_bark = ddsp.bark_index(
        freq=self.fftfrequencies)
    self.low_bark = low_bark
    self.high_bark = high_bark

    if old_wrapper:
      self.patterns = patterns_matrix(
          low_bark=self.low_bark,
          nfft=self.nfft)

    else:
      self.register_buffer("patterns", patterns_matrix(
          low_bark=self.low_bark,
          nfft=self.nfft))

  def compute_target_filter(
      self,
      gains: torch.Tensor,
  ):
    """
    Compute the target filter from the input gains in dB and the patterns per bark band.

    Parameters
    ----------
    gains : torch.Tensor
      A tensor of shape (batch_size, n_frames, n_bark) containing the input gains in dB.

    Returns
    -------
    torch.Tensor
      A tensor of shape (batch_size, n_frames, nfft//2 + 1) containing the computed target filters.
    """

    batch_size, n_frames = gains.shape[-3], gains.shape[-2]

    # reshape to [batch_size*n_frames, n_bark]
    gains = gains.reshape(-1, gains.shape[-1])   # gains already in dB
    all_gains = torch.matmul(gains, self.patterns)
    # all_gains : [batch_size*n_frames, nfft//2 + 1]

    # reshape all_gains to [batch_size, n_frames, nfft//2 + 1]

    all_gains = all_gains.reshape(batch_size, n_frames, -1)

    return 10**(all_gains / 20)

  def compute_target_envelope(
          self,
          input_envelope: torch.Tensor,
          gains: torch.Tensor):
    """
    Compute the target spectrum envelope from the initial input envelope
    and the gains computed by the neural network.

    Parameters
    ----------
    input_envelope : torch.Tensor
        Initial spectral cepstrum envelope
        Shape : [nb_freq, nb_frames]
    gains : torch.Tensor
        Predicted gains (in dB) to be applied on each bark band.
        Shape : [nb_bark, nb_frames]

    Returns
    -------
    torch.Tensor
        Target envelope
    """
    input_db = linear2dB(input_envelope, 20)

    # patterns : [n_bark, n_fft//2 + 1]
    # gains : [batch_size, n_frames, n_bark]

    batch_size, n_frames = gains.shape[-3], gains.shape[-2]

    # reshape to [batch_size*n_frames, n_bark]
    gains = gains.reshape(-1, gains.shape[-1])
    all_gains = torch.matmul(gains, self.patterns)
    # all_gains : [batch_size*n_frames, nfft//2 + 1]

    # reshape all_gains to [batch_size, n_frames, nfft//2 + 1]

    all_gains = all_gains.reshape(batch_size, n_frames, -1)

    spectral_target = input_db + all_gains

    return 10 ** (spectral_target / 20)

  def target_frequency_response(self, target_envelope, input_envelope):
    return target_envelope / input_envelope


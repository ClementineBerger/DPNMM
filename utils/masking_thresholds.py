"""
Pytorch implementation of the computation of the masking thresholds via the Johnston model.
"""

import torch
import torch.nn as nn
from torch.nn import Module

import torch.distributed as dist

import utils.ddsp as ddsp
from utils.utils import safe_log, to_device


def sprdng(nu: int, j: int):
  """
  Johnston's spreading function.


  Parameters
  ----------
  nu : integer
      Maskee band index
  j : integer
      Masker band index
  """
  delta_nu = nu - j
  B = 15.81 + 7.5 * (delta_nu + 0.474) - 17.5 * torch.sqrt(torch.tensor(1 +
                                                           (delta_nu + 0.474) ** 2))
  B = 10 ** (B / 10)
  return B


def spreading_matrix(nbands: int):
  """
  Compute the spreading matrix of size (nbands x nbands).
  Describes how each critical band affects the others.

  Parameters
  ----------
  nbands : int
      Number of critical band.
  """
  mSpreading = torch.zeros((nbands, nbands))

  for i in range(nbands):
    for j in range(nbands):
      mSpreading[i, j] = sprdng(i, j)

  return mSpreading


def hz2bark_matrix(nbands: int, nfft: int, low_bark: torch.IntTensor):
  """
  Creates a matrix that maps frequencies to Bark scale bands.

  Parameters
  ----------
  nbands : int
    The number of Bark scale bands.
  nfft : int
    Window size of the FFT.
  low_bark : torch.IntTensor
    A tensor containing the lower bounds of each Bark band in terms of frequency bins.

  Returns
  -------
  torch.Tensor
    A matrix of shape (nfreqs, nbands) where nfreqs is `nfft // 2 + 1`. Each row corresponds to a frequency bin,
    and each column corresponds to a Bark band. The matrix contains 1s where the frequency bin falls within the
    corresponding Bark band and 0s elsewhere.
  """
  nfreqs = nfft // 2 + 1
  matrix = torch.zeros((nfreqs, nbands))
  for i in range(len(low_bark) - 1):
    matrix[low_bark[i]:low_bark[i + 1], i] = 1.
  matrix[low_bark[-1]:, len(low_bark) - 1] = 1.
  return matrix


def bark2hz_matrix(nbands: int, nfft: int, low_bark: torch.IntTensor):
  """
  Converts a Bark scale matrix to a frequency (Hz) scale matrix.

  Parameters
  ----------
  nbands : int
    Number of Bark bands.
  nfft : int
    Number of FFT points.
  low_bark : torch.IntTensor
    A tensor containing the lower bounds of each Bark band in terms of frequency bins.

  Returns
  -------
  torch.Tensor
    A tensor representing the Bark to frequency (Hz) conversion matrix.
  """
  freq2bark = hz2bark_matrix(nbands, nfft, low_bark)
  bark2freq = freq2bark.T    # shape(nbands, nfreqs)
  norm = torch.sum(bark2freq, dim=1)  # shape(nbands)
  return torch.divide(bark2freq, norm.reshape(-1, 1))


class MaskingThresholds(Module):
  """
  MaskingThresholds is a module that computes masking thresholds
  in the bark scale for a given power spectrum.

  nfft : int
    Number of FFT points.
  sr : int
    Sampling rate.
  old_wrapper : bool, optional
    If True, uses old wrapper method for matrix initialization. Default is False.

  Attributes
  ----------
  nfft : int
    Number of FFT points.
  sr : int
    Sampling rate.
  fftfrequencies : torch.Tensor
    FFT frequencies.
  low_bark : torch.Tensor
    Lower bark indices.
  high_bark : torch.Tensor
    Higher bark indices.
  nbands : int
    Number of bark bands.
  spreadmatrix : torch.Tensor
    Spreading matrix.
  hz2bark_matrix : torch.Tensor
    Matrix to convert Hz to bark scale.
  bark2hz_matrix : torch.Tensor
    Matrix to convert bark to Hz scale.

  Methods
  -------
  convert_hz2bark(psd_by_hz_linear)
    Convert density spectrum (power spectrum) to the bark scale.
  compute_SFMdB(psd_by_hz_linear)
    Compute Spectral Flatness Measure in dB.
  compute_alpha(SFMdB)
    Compute alpha coefficient from SFMdB.
  compute_offset(alpha, bark)
    Compute offset.
  compute_thresholds(psd_by_hz_linear)
  convert_bark2hz(mTbark)
    Convert masking thresholds from bark scale to Hz scale.
  """

  def __init__(self, nfft: int,
               sr: int, old_wrapper=False):

    super(MaskingThresholds, self).__init__()

    self.nfft = nfft
    self.sr = sr
    self.fftfrequencies = ddsp.fftfreq(nfft=self.nfft, sr=self.sr)
    low_bark, high_bark = ddsp.bark_index(freq=self.fftfrequencies)
    self.low_bark = low_bark
    self.high_bark = high_bark
    self.nbands = len(low_bark)

    if old_wrapper:
      self.spreadmatrix = spreading_matrix(nbands=self.nbands)
      self.hz2bark_matrix = hz2bark_matrix(
          nbands=self.nbands,
          nfft=self.nfft,
          low_bark=self.low_bark
      )
      self.bark2hz_matrix = bark2hz_matrix(
          nbands=self.nbands,
          nfft=self.nfft,
          low_bark=self.low_bark
      )

    else:
      self.register_buffer(
          "spreadmatrix",
          spreading_matrix(
              nbands=self.nbands))
      self.register_buffer("hz2bark_matrix", hz2bark_matrix(
          nbands=self.nbands,
          nfft=self.nfft,
          low_bark=self.low_bark
      ))
      self.register_buffer("bark2hz_matrix", bark2hz_matrix(
          nbands=self.nbands,
          nfft=self.nfft,
          low_bark=self.low_bark
      ))

  def convert_hz2bark(self, psd_by_hz_linear):
    """
    Convert density spectrum (power spectrum) to the bark scale,
    computing the density per critical band.

    Parameters
    ----------
    psd_by_hz_linear : torch.Tensor
        [batch_size, n_frames, nfft//2 + 1]
        Power spectrum on the Hz scale and linear amplitude scale.

    Returns
    -------
    torch.Tensor
        [batch_size, n_frames, n_bark]
    """
    initial_dims = psd_by_hz_linear.shape

    psd_by_hz_linear = psd_by_hz_linear.reshape(-1, psd_by_hz_linear.shape[-1])

    psd_by_bark_linear = torch.matmul(psd_by_hz_linear, self.hz2bark_matrix)

    if len(initial_dims) > 2:
      batch_size, nframes = initial_dims[:2]
      psd_by_bark_linear = psd_by_bark_linear.reshape(
          (batch_size, nframes, -1))
    else:
      nframes = initial_dims[0]
      psd_by_bark_linear = psd_by_bark_linear.reshape((nframes, -1))

    return psd_by_bark_linear

  def compute_SFMdB(self, psd_by_hz_linear: torch.Tensor):
    """
    Compute Spectral Flatness Measure in dB.

    Parameters
    ----------
    psd_by_hz_linear : torch.Tensor
      Power spectrum on the Hz scale and linear amplitude scale.
      Shape: [batch_size*n_frames, n_freq]

    Returns
    -------
    torch.Tensor
      Spectral Flatness Measure in dB.
      Shape: [batch_size*n_frames, 1]
    """

    original_ndim = psd_by_hz_linear.ndim

    if original_ndim > 2:
      batch_size, n_frames, n_freq = psd_by_hz_linear.shape
      psd_by_hz_linear = psd_by_hz_linear.reshape(batch_size * n_frames, -1)

    SFMdB = 10 * torch.mean(safe_log(psd_by_hz_linear), dim=1
                            ) - 10 * safe_log(torch.mean(psd_by_hz_linear, dim=1))

    if original_ndim > 2:
      SFMdB = SFMdB.reshape((batch_size, n_frames))

    return SFMdB

  def compute_alpha(self, SFMdB: torch.Tensor):
    """
    Compute alpha coefficient from SFMdB.

    Parameters
    ----------
    SFMdB : torch.Tensor
      Spectral Flatness Measure in dB.
      Shape: [n_bark, 1]

    Returns
    -------
    torch.Tensor
      Alpha coefficient.
      Shape: [n_bark, 1]
    """
    return 1. - nn.ReLU()(1. - SFMdB / (-60.))

  def compute_offset(self, alpha: torch.Tensor, bark: torch.Tensor):
    """
    Compute offset.

    Parameters
    ----------
    alpha : torch.Tensor
      Alpha coefficient.
      Shape: [n_bark, 1]
    bark : torch.Tensor
      Bark scale values.
      Shape: [n_bark, 1]

    Returns
    -------
    torch.Tensor
      Offset values.
      Shape: [n_bark, 1]
    """
    return alpha * (14.5 + bark) + (1 - alpha) * 5.5

  def compute_thresholds(self, psd_by_hz_linear: torch.Tensor):
    """
    Compute the masking thresholds per bark band.

    Parameters
    ----------
    psd_by_hz_linear : torch.Tensor
      Power spectrum on the Hz scale and linear amplitude scale.
      Shape: [batch_size, n_frames, nfft//2 + 1]

    Returns
    -------
    torch.Tensor
      Masking thresholds per bark band.
      Shape: [batch_size, n_frames, n_bark]
    """
    initial_dims = psd_by_hz_linear.shape

    # reshape to [batch_size*n_frames, nfft//2 + 1] to compute the thresholds
    # for all the frames at once
    psd_by_hz_linear = psd_by_hz_linear.reshape(-1, psd_by_hz_linear.shape[-1])

    psd_by_bark_linear = torch.matmul(psd_by_hz_linear, self.hz2bark_matrix)
    # [batch_size*n_frames, n_barks]

    simultaneous_masking = torch.matmul(  # manque le paramètre a (considéré égal à 1 ici)
        psd_by_bark_linear,
        self.spreadmatrix)  # [batch_size*n_frames, n_barks]

    SFMdB = self.compute_SFMdB(psd_by_hz_linear).reshape(
        -1, 1)  # [batch_size*n_frames, 1]

    alpha = self.compute_alpha(SFMdB)
    offset = self.compute_offset(alpha, bark=torch.arange(
        0, self.nbands, device=psd_by_hz_linear.device).reshape(1, -1))

    threshold_by_bark = 10 ** (
        safe_log(
            simultaneous_masking
        ) -
        offset /
        10)

    # back to [batch_size, n_frames, n_bark]

    if len(initial_dims) > 2:
      batch_size, nframes = initial_dims[:2]
      threshold_by_bark = threshold_by_bark.reshape((batch_size, nframes, -1))
    else:
      nframes = initial_dims[0]
      threshold_by_bark = threshold_by_bark.reshape((nframes, -1))

    return threshold_by_bark

  def convert_bark2hz(self, mTbark: torch.Tensor):
    """
    Convert a tensor from Bark scale to Hertz scale using a predefined matrix.

    Parameters
    ----------
    mTbark : torch.Tensor
      Input tensor in Bark scale with shape [batch_size, n_frames, n_bark] or [n_frames, n_bark].

    Returns
    -------
    torch.Tensor
      Output tensor in Hertz scale with shape [batch_size, n_frames, n_hz] or [n_frames, n_hz].
    """
    # mTbark [batch_size, n_frames, n_bark] or [n_frames, n_bark]

    initial_dims = mTbark.shape

    # mTbark = mTbark.reshape(-1, mTbark.shape[-2], mTbark.shape[-1])

    # batch_size, n_frames = mTbark.shape[0], mTbark.shape[1]
    mTbark = mTbark.reshape(-1, mTbark.shape[-1])

    mThz = torch.matmul(mTbark, self.bark2hz_matrix)
    if len(initial_dims) > 2:
      batch_size, nframes = initial_dims[:2]
      mThz = mThz.reshape((batch_size, nframes, -1))
    else:
      nframes = initial_dims[0]
      mThz = mThz.reshape((nframes, -1))

    return mThz

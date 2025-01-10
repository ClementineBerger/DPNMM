"""
Losses computation
"""

import torch
import torch.nn as nn

from utils.masking_thresholds import MaskingThresholds
from utils.utils import linear2dB, to_device, a_weightings
from utils.analysis import stft, slidding_mean_power, apply_weights_to_spectrum


class LossFunction(nn.Module):
  def __init__(
          self,
          nfft: int,
          sr: int,
          nb_bark: int,
          thresholds_margins,
          loss_type: str,
          mean_power_constraint: bool = False,
          mdmm=None,
          epsilon: float = 1e-12,):
    """
    Class to compute the loss function of the model.

    Parameters
    ----------
    nfft : int
        Window size, frequency resolution.
    sr : int
        Sampling rate
    nb_bark : int
        Number of bark band
    thresholds_margins : list, array or tensor
        List of optional tolerance margin for the threshold level. Usually 0.
    loss_type : str
        Type of loss to use on the thresholds, can be "abs" or "relu".
    mean_power_constraint : bool, optional
        Whether or not the constraint on mean power is used, by default False
    mdmm : None or class object, optional
        If computing the constraint on mean power, class to use MDMM to weight this loss term, by default None
    epsilon : float, optional
        , by default 1e-12
    """

    super(LossFunction, self).__init__()

    self.nfft = nfft
    self.sr = sr

    self.nb_bark = nb_bark
    self.register_buffer(
        "thresholds_margins",
        self.float2tensor(thresholds_margins))

    self.epsilon = epsilon

    self.loss_type = loss_type

    self.masking_thresholds = MaskingThresholds(
        nfft=nfft, sr=sr)

    self.mean_power_constraint = mean_power_constraint

    self.register_buffer(
        "a_weights", a_weightings(
            freq=torch.fft.rfftfreq(
                nfft, 1 / sr)))

    if self.mean_power_constraint:
      self.mdmm = mdmm

  def float2tensor(self, x):
    """
    Converts a float or a list/array to a PyTorch tensor.

    If the input is a float, it creates a tensor filled with the float value,
    repeated `nb_bark` times. If the input is already a list or array, it
    converts it directly to a tensor.

    Args:
      x (float or list/array): The input value to be converted to a tensor.

    Returns:
      torch.Tensor: The resulting tensor.
    """
    if isinstance(x, float):
      x = x * torch.ones(self.nb_bark)
    else:
      x = torch.tensor(x)
    return x

  def type_of_band(
          self,
          init_masking_thresholds_by_bark_in_db,
          noise_psd_by_bark_in_db):
    """
    Determines the type of band (too low threshold or not) for each Bark band.
    Returns 0 if the threshold is too low, 1 otherwise.


    Parameters
    ----------
    mTbark : Initial thresholds levels
    noise_level_by_bark : Noise PSD by bark bands

    Returns
    -------
    Tensor
        Tensor filled with 0 and 1.
    """
    # mTbark : [batch_size, n_frames, n_bark]
    # noise_level : [batch_size, n_frames, n_bark]
    # margins : [n_bark]

    noise_level_with_margins = noise_psd_by_bark_in_db + \
        self.thresholds_margins.reshape(1, 1, -1)

    condition = init_masking_thresholds_by_bark_in_db - noise_level_with_margins + self.epsilon

    band_type = condition / torch.abs(condition)
    band_type = nn.ReLU()(band_type)

    return band_type

  def masking_loss(
          self,
          new_masking_thresholds_by_bark_in_db,
          noise_psd_by_bark_in_db):
    """
    Compute the loss function on masking thresholds.

    Parameters
    ----------
    new_masking_thresholds_by_bark_in_db : torch.Tensor
      The new masking thresholds by bark in dB.
    noise_psd_by_bark_in_db : torch.Tensor
      The noise power spectral density by bark in dB.

    Returns
    -------
    torch.Tensor
      The computed loss based on the specified loss type and penalty.
    Notes
    -----
    """

    if self.loss_type == "abs":
      loss_too_low = torch.abs(
          new_masking_thresholds_by_bark_in_db -
          (noise_psd_by_bark_in_db + self.thresholds_margins.reshape(1, 1, -1)))
    elif self.loss_type == "relu":
      loss_too_low = nn.ReLU()((noise_psd_by_bark_in_db +
                                self.thresholds_margins.reshape(1, 1, -
                                                                1)) -
                               new_masking_thresholds_by_bark_in_db)
    else:
      raise ValueError(
          f"Unknown loss type {self.loss_type}. Should be 'abs' or 'relu'.")
    return loss_too_low

  def loss_on_mean_power(
          self,
          init_music_psd_by_hz_linear,
          new_music_psd_by_hz_linear):
    """
    Compute the loss on the mean difference of power between initial music and processed music.

    Parameters
    ----------
    init_music_psd_by_hz_linear : Tensor
        Initial psd of the music. [batch_size, nframe, nfft//2 + 1]
    new_music_psd_by_hz_linear : Tensor
        New psd of the music. [batch_size, nframe, nfft//2 + 1]

    Returns
    -------
    torch.Tensor
        The computed loss on the mean difference of power between initial music and processed music.

    """
    init_weighted = apply_weights_to_spectrum(
        spectrum=init_music_psd_by_hz_linear,
        weights=self.a_weights**2  # power, because the input is the psd
    )

    new_weighted = apply_weights_to_spectrum(
        spectrum=new_music_psd_by_hz_linear,
        weights=self.a_weights**2
    )

    # [batch_size, nframe]
    # Compute the RMS level from the spectrum (only positive frequencies)
    init_rms_level = (1 / (2 * (self.nfft // 2 + 1) - 2)) * torch.sqrt(
        2 * torch.sum(init_weighted, dim=-1) - init_weighted[:, :, 0] + 1e-12
    )
    init_dba_level = linear2dB(init_rms_level, 20)
    new_rms_level = (1 / (2 * (self.nfft // 2 + 1) - 2)) * torch.sqrt(
        2 * torch.sum(new_weighted, dim=-1) - new_weighted[:, :, 0] + 1e-12
    )
    new_dba_level = linear2dB(new_rms_level, 20)

    return torch.abs(init_dba_level - new_dba_level)

  def complete_loss(
          self,
          init_masking_thresholds_by_bark_in_db,
          new_masking_thresholds_by_bark_in_db,
          noise_psd_by_bark_in_db,
          new_music_psd_by_hz_linear,
          init_music_psd_by_hz_linear,
          init_music_waveform,
          new_music_waveform):
    """
    Compute the complete loss for perceptual noise cancellation.

    Parameters:
    -----------
    init_masking_thresholds_by_bark_in_db : torch.Tensor
      Initial masking thresholds by Bark scale in dB.
    new_masking_thresholds_by_bark_in_db : torch.Tensor
      New masking thresholds by Bark scale in dB.
    noise_psd_by_bark_in_db : torch.Tensor
      Noise power spectral density by Bark scale in dB.
    new_music_psd_by_hz_linear : torch.Tensor
      New music power spectral density by Hz in linear scale.
    init_music_psd_by_hz_linear : torch.Tensor
      Initial music power spectral density by Hz in linear scale.
    init_music_waveform : torch.Tensor
      Initial music waveform.
    new_music_waveform : torch.Tensor
      New music waveform.

    Returns:
    --------
    complete_loss : torch.Tensor
      The computed complete loss.
    all_losses : tuple of torch.Tensor
      Tuple containing the mean of the loss due to low thresholds and the mean power loss.
    """

    # discrimination of the 2 types of bands
    band_types = self.type_of_band(
        init_masking_thresholds_by_bark_in_db=init_masking_thresholds_by_bark_in_db,
        noise_psd_by_bark_in_db=noise_psd_by_bark_in_db)
    # return 0 if the threshold is too low
    # 1 else

    loss_too_low = (
        1 - band_types) * self.masking_loss(
        new_masking_thresholds_by_bark_in_db=new_masking_thresholds_by_bark_in_db,
        noise_psd_by_bark_in_db=noise_psd_by_bark_in_db) + band_types * nn.ReLU()(
        noise_psd_by_bark_in_db - new_masking_thresholds_by_bark_in_db)

    if self.mean_power_constraint:
      loss_mean_power = self.loss_on_mean_power(
          new_music_psd_by_hz_linear=new_music_psd_by_hz_linear,
          init_music_psd_by_hz_linear=init_music_psd_by_hz_linear
      )
      if self.mdmm is not None:
        all_losses = (torch.mean(loss_too_low), torch.mean(loss_mean_power))

        complete_loss = self.mdmm.combine_losses(
            primary_loss=all_losses[0],
            secondary_loss=all_losses[1:]
        )
      else:
        # ablation study without adaptive weighting
        complete_loss = torch.mean(
            loss_too_low) + 10 * nn.ReLU()(torch.mean(loss_mean_power - 2.))
        all_losses = (torch.mean(loss_too_low), torch.mean(loss_mean_power))

    else:
      loss_mean_power = self.loss_on_mean_power(
          new_music_psd_by_hz_linear=new_music_psd_by_hz_linear,
          init_music_psd_by_hz_linear=init_music_psd_by_hz_linear
      )  # just to monitor

      complete_loss = torch.mean(loss_too_low)
      all_losses = (torch.mean(loss_too_low), torch.mean(loss_mean_power))

    return complete_loss, all_losses

  def forward(self,
              new_masking_thresholds_by_bark_in_db,
              init_masking_thresholds_by_bark_in_db,
              noise_psd_by_bark_in_db,
              new_music_psd_by_hz_linear,
              init_music_psd_by_hz_linear,
              new_music_waveform,
              init_music_waveform):

    all_losses_terms = self.complete_loss(
        init_masking_thresholds_by_bark_in_db=init_masking_thresholds_by_bark_in_db,
        new_masking_thresholds_by_bark_in_db=new_masking_thresholds_by_bark_in_db,
        noise_psd_by_bark_in_db=noise_psd_by_bark_in_db,
        new_music_psd_by_hz_linear=new_music_psd_by_hz_linear,
        init_music_psd_by_hz_linear=init_music_psd_by_hz_linear,
        new_music_waveform=new_music_waveform,
        init_music_waveform=init_music_waveform)
    return all_losses_terms

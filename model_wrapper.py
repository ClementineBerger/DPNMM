"""
Model wrapper for the complete neural system.
"""

import torch
import torch.nn as nn
import torchaudio

import librosa

from model import DPNMM
from utils.masking_thresholds import MaskingThresholds
from utils.spectral_shaping import SpectralEnvelope, TargetFilter
from utils.utils import linear2dB
from utils.analysis import stft, istft
from utils.post_process import smoothing_gains, find_activated_bands

import yaml

import os

path = os.environ['REPO']

results_path = os.path.join(
    path,
    "trained_models",
)


class ModelWrapper(nn.Module):
  """
  A wrapper class for the DPNMM model.

  Parameters
  ----------
  conf_name : str
    The name of the configuration file.

  Attributes
  ----------
  conf : dict
    The configuration loaded from the YAML file.
  model : DPNMM
    The perceptual noise reduction model.
  nfft : int
    The number of FFT points.
  sr : int
    The sampling rate.
  masking_thresholds : MaskingThresholds
    The masking thresholds object.
  target_filter : TargetFilter
    The target filter object.
  spectral_envelope : SpectralEnvelope
    The spectral envelope object.

  Methods
  -------
  compute_input_cnn(
    music_waveform,
    noise_waveform,
    music_percussive_mask=None,
     noise_percussive_mask=None)
    Computes the input for the CNN model.
  forward(
    music_waveform,
    noise_waveform,
    mean_gains=True,
    remove_non_activated_bands=False,
    return_dict=False,
    music_percussive_mask=None,
     noise_percussive_mask=None)
    Forward pass through the model.
  """

  def __init__(self, conf_name, job_id):
    super(ModelWrapper, self).__init__()

    conf_path = os.path.join(
        results_path,
        conf_name
    )

    model_path = os.path.join(
        conf_path,
        job_id,
        "best_model.pth"
    )

    config_path = os.path.join(
        conf_path,
        job_id,
        "conf.yml"
    )

    with open(config_path, 'r') as f:
      self.conf = yaml.safe_load(f)

    self.model = DPNMM(
        nfft=self.conf['model']['nfft'],
        nb_bark=self.conf['model']['nb_bark'],
        input_ch=self.conf['model']['input_ch'],
        conv_ch=self.conf['model']['conv_ch'],
        conv_kernel_inp=self.conf['model']['conv_kernel_inp'],
        conv_kernel=self.conf['model']['conv_kernel'],
        emb_hidden_dim=self.conf['model']['emb_hidden_dim'],
        emb_num_layers=self.conf['model']['emb_num_layers'],
        lin_groups=self.conf['model']['lin_groups'],
        enc_lin_groups=self.conf['model']['enc_lin_groups'],
        rnn_type=self.conf['model']['rnn_type'],
        trans_conv_type=self.conf['model']['trans_conv_type'],
        max_positive_clamping_value=self.conf['model']['gain_clamping']
        ['max_positive_clamping_value'],
        min_negative_clamping_value=self.conf['model']['gain_clamping']
        ['min_negative_clamping_value'],
        remove_high_bands=self.conf['model']['gain_clamping']['remove_high_bands'])
    

    self.model.load_state_dict(torch.load(model_path))

    self.mode = self.conf['mode']
    self.hpss_music = self.conf["data"]["hpss_music"]
    self.hpss_noise = self.conf["data"]["hpss_noise"]

    self.nfft = self.conf['audio']['nfft']
    self.sr = self.conf['audio']['sr']

    self.masking_thresholds = MaskingThresholds(
        nfft=self.conf['audio']['nfft'],
        sr=self.conf['audio']['sr'],
        # device="cuda"
    )

    self.target_filter = TargetFilter(
        nfft=self.conf['audio']['nfft'],
        sr=self.conf['audio']['sr'],
        filter_order=self.conf['audio']['filter_order'],
        # device="cuda"
    )

    self.spectral_envelope = SpectralEnvelope(
        nfft=self.conf['audio']['nfft'],
        sr=self.conf['audio']['sr'],
        order=self.conf['audio']['envelope_order']
    )

  def compute_input_cnn(
          self,
          music_waveform,
          noise_waveform,):
    """
    Compute the input for the CNN model by processing the music and noise waveforms to compute bark features.

    Parameters
    ----------
    music_waveform : torch.Tensor
      The waveform of the music signal. Shape: (batch_size, channels, samples) or (batch_size, samples).
    noise_waveform : torch.Tensor
      The waveform of the noise signal. Shape: (batch_size, channels, samples) or (batch_size, samples).

    Returns
    -------
    input_cnn : torch.Tensor
      The input tensor for the CNN model. Shape: (batch_size, 3, freq_bins, time_frames).
    threshold_mask : torch.Tensor
      The threshold mask indicating where the noise exceeds the masking threshold. Shape: (batch_size, freq_bins, time_frames).
    additional_outputs : tuple
      Additional outputs depending on whether harmonic-percussive source separation (HPSS) is applied to the music signal.
      (music_stft, music_waveform, noise_waveform)
    """

    if music_waveform.ndim == 3:
      music_waveform = music_waveform.squeeze(
          1)  # remove channel if batched data
    if noise_waveform.ndim == 3:
      noise_waveform = noise_waveform.squeeze(
          1)  # remove channel if batched data

    music_stft = stft(audio=music_waveform, nfft=self.nfft, overlap=0.75)
    noise_stft = stft(audio=noise_waveform, nfft=self.nfft, overlap=0.75)

    mTbark = self.masking_thresholds.compute_thresholds(abs(music_stft) ** 2)
    music_bark = self.masking_thresholds.convert_hz2bark(
        abs(music_stft) ** 2)

    noise_bark = self.masking_thresholds.convert_hz2bark(
        abs(noise_stft) ** 2)

    input_cnn = torch.stack(
        (
            linear2dB(music_bark, 10),
            linear2dB(noise_bark, 10),
            linear2dB(mTbark, 10),
        ), dim=-3
    )

    threshold_mask = torch.where(
        linear2dB(
            noise_bark,
            10) > linear2dB(
            mTbark,
            10),
        torch.tensor(1.),
        torch.tensor(0.))

    return input_cnn, threshold_mask, (music_stft.squeeze(
    ), music_waveform, noise_waveform)

  def forward(self,
              music_waveform,
              noise_waveform,
              mean_gains=True,
              remove_non_activated_bands=False,
              return_dict=False,
              ):
    """
    Forward pass for the model.

    Parameters
    ----------
    music_waveform : torch.Tensor
      The waveform of the music signal.
    noise_waveform : torch.Tensor
      The waveform of the noise signal.
    mean_gains : bool, optional
      Whether to apply smoothing to the gains, by default True.
    remove_non_activated_bands : bool, optional
      Whether to remove non-activated bands, by default False.
    return_dict : bool, optional
      Whether to return a dictionary of results, by default False.

    Returns
    -------
    torch.Tensor or dict
      If `return_dict` is False, returns the new music waveform as a torch.Tensor.
      If `return_dict` is True, returns a dictionary with the following keys:
      - "stft": Initial and new STFTs of the music and noise.
      - "envelopes": Initial and new spectral envelopes.
      - "psd_bark": Initial and new power spectral densities in Bark scale.
      - "thr_bark": Initial and new masking thresholds in Bark scale.
      - "thr_hz": Initial and new masking thresholds in Hz scale.
      - "audio": Initial and new waveforms of the music and noise.
      - "filter": Target filters.
      - "gains": Gains applied to the bands.
      - "original_gains": Original gains before smoothing.
    """

    input_cnn, threshold_mask, others = self.compute_input_cnn(
        music_waveform=music_waveform,
        noise_waveform=noise_waveform,
    )

    input_cnn = input_cnn.squeeze(0)

    music_stft, music_waveform, noise_waveform = others

    gains = self.model(input_cnn)

    if remove_non_activated_bands:
      # searching for bands that are "activated" (ie have an impact on bands
      # where the thresholds need to be raised)
      activated_bands = find_activated_bands(
          threshold_mask=threshold_mask,
          spreadwidth=self.conf['gains']['spreadwidth'])  # 3
      gains = gains * activated_bands

    if mean_gains:
      original_gains = gains
      gains = smoothing_gains(
          gains_db=gains,
          N=self.nfft,
          tau_attack=250e-3,
          tau_release=250e-3,
          sr=self.sr
      )

    target_filters = self.target_filter.compute_target_filter(gains)
    new_music_stft = torch.multiply(music_stft, target_filters)  # linear

    new_music_waveform = istft(
        stft=new_music_stft,
        nfft=self.conf['audio']['nfft'],
        overlap=0.75,
        length=music_waveform.shape[-1])

    if return_dict:

      noise_stft = stft(audio=noise_waveform, nfft=self.nfft, overlap=0.75)

      noise_bark = self.masking_thresholds.convert_hz2bark(
          abs(noise_stft) ** 2)

      new_psd = torch.abs(new_music_stft)**2

      envelopes_init = self.spectral_envelope.compute_spectral_envelope(
          spectrum=music_stft)
      new_envelopes = self.spectral_envelope.compute_spectral_envelope(
          spectrum=new_music_stft)

      new_mT = self.model.masking_thresholds.compute_thresholds(new_psd)

      # all results in linear scale
      results = {
          "stft": {
              "init": music_stft.squeeze(dim=0),
              "new": new_music_stft.squeeze(dim=0).detach(),
              "noise": noise_stft.squeeze(dim=0)
          },
          "envelopes": {
              "init": envelopes_init,
              "new": new_envelopes.detach()
          },
          "psd_bark": {
              "init": self.masking_thresholds.convert_hz2bark(abs(music_stft)**2).squeeze(dim=0),
              "new": self.masking_thresholds.convert_hz2bark(abs(new_music_stft)**2).squeeze(dim=0).detach(),
              "noise": noise_bark.squeeze(dim=0),
          },
          "thr_bark": {
              "init": 10**(input_cnn[2, :, :] / 10),
              "new": new_mT.squeeze(dim=0).detach()
          },
          "thr_hz": {
              "init": self.masking_thresholds.convert_bark2hz(10**(input_cnn[2, :, :] / 10)).squeeze(dim=0),
              "new": self.masking_thresholds.convert_bark2hz(new_mT).squeeze(dim=0).detach()
          },
          "audio": {
              "init": music_waveform,
              "new": new_music_waveform.detach(),
              "noise": noise_waveform,
          },
          "filter": target_filters[0].detach(),
          "gains": gains[0].detach(),
          "original_gains": original_gains[0].detach(),
      }

      return results

    return new_music_waveform

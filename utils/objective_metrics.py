"""
Objective metrics for evaluation of the models.
"""

import torch

from utils.masking_thresholds import MaskingThresholds
from utils.analysis import stft, a_weightings, apply_weights_to_spectrum, compute_rms_level_from_spectrum
from utils.utils import linear2dB
from utils.ddsp import bark_index


class ObjectiveMetrics(MaskingThresholds):
  """
    A class used to compute various objective metrics for audio signals.

        Sampling rate.
        Parameter for this code to work with an older version of the neural model, by default False.

    Attributes
    ----------
        Window size for computing global level difference.
    gld_freqs : torch.Tensor
        Frequencies for the global level difference computation.
    a_weights : torch.Tensor
        A-weighting coefficients for the frequencies.
    epsilon : float
        Small value to avoid division by zero.

    Methods
    -------
    time_average_level_difference(init_waveform, output_waveform)
        Computes the time-averaged level difference between the initial and output waveforms.
    noise_to_mask_ratio(music_waveform, noise_waveform, mask=None)
        Computes the noise-to-mask ratio for given music and noise waveforms.
    spectral_flatness_measure(waveform)
        Computes the spectral flatness measure for a given waveform.
  """

  def __init__(
          self,
          nfft: int,
          sr: int,
          gld_nfft: int,
          old_wrapper=False):
    """
    Initialization of the class.

    Parameters
    ----------
    nfft : int
        Analysis window for masking thresholds computation.
    sr : int
        Sampling rate
    gld_nfft : int
        Window size for global level difference computation.
    old_wrapper : bool, optional
        Parameter for this code to work with an older version of the neural model, by default False
    """
    super().__init__(nfft, sr, old_wrapper)

    # Window size for computing global level difference, not the same as NMR
    self.gld_nfft = gld_nfft

    self.register_buffer("gld_freqs", torch.fft.rfftfreq(gld_nfft, 1 / sr))

    self.register_buffer(
        "a_weights", a_weightings(
            freq=self.gld_freqs).reshape(
            1, -1))

    self.epsilon = 1e-16

  def time_average_level_difference(self, init_waveform, output_waveform):
    """
    Compute the time-averaged level difference between the initial and output waveforms
    across different frequency bands (broadband, low, medium, high).

    Parameters
    ----------
    init_waveform : torch.Tensor
        The initial waveform tensor.
    output_waveform : torch.Tensor
        The output waveform tensor.

    Returns
    -------
    tuple
        A tuple containing:
        - broadband_gld (torch.Tensor): The broadband global level difference.
        - (low_gld, medium_gld, high_gld) (tuple of torch.Tensor): The global level differences
          for the low, medium, and high frequency bands respectively.
    """

    init_spectrum = stft(audio=init_waveform, nfft=self.gld_nfft, overlap=0.75)
    output_spectrum = stft(
        audio=output_waveform,
        nfft=self.gld_nfft,
        overlap=0.75)

    init_spectrum_weights = apply_weights_to_spectrum(
        spectrum=init_spectrum,
        weights=self.a_weights
    )

    output_spectrum_weights = apply_weights_to_spectrum(
        spectrum=output_spectrum,
        weights=self.a_weights
    )

    # Computing the limit of low band, medium band, high band
    low_bark, high_bark = bark_index(freq=self.gld_freqs)
    n_bark = self.nbands  # same number of bark bands as used to compute masking thresholds

    low_bark_limit = n_bark // 3
    medium_bark_limit = 2 * (n_bark // 3)

    # Broadband levels
    init_dba_level = compute_rms_level_from_spectrum(
        spectrum=init_spectrum_weights
    )
    init_dba_level = linear2dB(
        x=init_dba_level,
        gain=20
    )

    # [batch_size, nframes]
    output_dba_level = compute_rms_level_from_spectrum(
        spectrum=output_spectrum_weights
    )
    output_dba_level = linear2dB(
        x=output_dba_level,
        gain=20
    )

    broadband_gld = torch.mean(
        abs(
            output_dba_level -
            init_dba_level),
        dim=-
        1)  # [batch_size, ]

    # Low band
    init_band_spectrum = init_spectrum_weights.clone()
    # all that is above the low band is put to 0
    init_band_spectrum[:, :, low_bark[low_bark_limit]:] = self.epsilon

    output_band_spectrum = output_spectrum_weights.clone()
    output_band_spectrum[:, :, low_bark[low_bark_limit]:] = self.epsilon

    init_dba_level = compute_rms_level_from_spectrum(
        spectrum=init_band_spectrum
    )
    init_dba_level = linear2dB(
        x=init_dba_level,
        gain=20
    )

    output_dba_level = compute_rms_level_from_spectrum(
        spectrum=output_band_spectrum
    )
    output_dba_level = linear2dB(
        x=output_dba_level,
        gain=20
    )

    low_gld = torch.mean(
        abs(
            output_dba_level -
            init_dba_level),
        dim=-
        1)  # [batch_size, ]

    # Middle band level
    init_band_spectrum = init_spectrum_weights.clone()
    init_band_spectrum[:, :, :low_bark[low_bark_limit]] = self.epsilon
    init_band_spectrum[:, :, low_bark[medium_bark_limit]:] = self.epsilon

    output_band_spectrum = output_spectrum_weights.clone()
    output_band_spectrum[:, :, :low_bark[low_bark_limit]] = self.epsilon
    output_band_spectrum[:, :, low_bark[medium_bark_limit]:] = self.epsilon

    init_dba_level = compute_rms_level_from_spectrum(
        spectrum=init_band_spectrum
    )
    init_dba_level = linear2dB(
        x=init_dba_level,
        gain=20
    )

    output_dba_level = compute_rms_level_from_spectrum(
        spectrum=output_band_spectrum
    )
    output_dba_level = linear2dB(
        x=output_dba_level,
        gain=20
    )

    medium_gld = torch.mean(
        abs(
            output_dba_level -
            init_dba_level),
        dim=-
        1)  # [batch_size, ]

    # High level
    init_band_spectrum = init_spectrum_weights.clone()
    init_band_spectrum[:, :, :low_bark[medium_bark_limit]] = self.epsilon

    output_band_spectrum = output_spectrum_weights.clone()
    output_band_spectrum[:, :, :low_bark[medium_bark_limit]] = self.epsilon

    init_dba_level = compute_rms_level_from_spectrum(
        spectrum=init_band_spectrum
    )
    init_dba_level = linear2dB(
        x=init_dba_level,
        gain=20
    )

    output_dba_level = compute_rms_level_from_spectrum(
        spectrum=output_band_spectrum
    )
    output_dba_level = linear2dB(
        x=output_dba_level,
        gain=20
    )

    high_gld = torch.mean(
        abs(
            output_dba_level -
            init_dba_level),
        dim=-
        1)  # [batch_size, ]

    return broadband_gld, (low_gld, medium_gld, high_gld)

  def noise_to_mask_ratio(self, music_waveform, noise_waveform, mask=None):
    """
    Compute the noise-to-mask ratio (NMR) for given music and noise waveforms.

    Parameters
    ----------
    music_waveform : torch.Tensor
        The music waveform tensor.
    noise_waveform : torch.Tensor
        The noise waveform tensor.
    mask : torch.Tensor, optional
        Mask of critical bands per frame to compute NMR on. [batch_size, n_frame, n_bark]
        If None, the mask is computed on the given music and noise.

    Returns
    -------
    tuple
        A tuple containing:
        - broadband_mean_nmr (torch.Tensor): The broadband mean noise-to-mask ratio.
        - (low_mean_nmr, medium_mean_nmr, high_mean_nmr) (tuple of torch.Tensor): The mean noise-to-mask ratios
          for the low, medium, and high frequency bands respectively.
        - mask (torch.Tensor): The mask used for the NMR computation.
    """

    n_bark = self.nbands  # same number of bark bands as used to compute masking thresholds

    low_bark_limit = n_bark // 3
    medium_bark_limit = 2 * (n_bark // 3)

    music_spectrum = stft(audio=music_waveform, nfft=self.nfft, overlap=0.75)
    noise_spectrum = stft(audio=noise_waveform, nfft=self.nfft, overlap=0.75)

    # [batch_size, nframe, nbark]
    music_thr = self.compute_thresholds(
        psd_by_hz_linear=abs(music_spectrum)**2)
    music_thr_db = linear2dB(
        x=music_thr,
        gain=10
    )

    noise_psd_by_bark = self.convert_hz2bark(
        psd_by_hz_linear=abs(noise_spectrum)**2)
    noise_psd_by_bark_db = linear2dB(
        x=noise_psd_by_bark,
        gain=10
    )

    # output : [batch_size, ] -> mean per audios
    # Broadband mean NMR
    eps = 1e-12

    if mask is None:
      mask = noise_psd_by_bark_db > music_thr_db

    def compute_mean_nmr(noise_db, music_db, mask):
      """
      Compute the mean Noise-to-Mask Ratio (NMR) on selected critical bands.

      Parameters
      ----------
      noise_db : torch.Tensor
          A tensor containing noise decibel values with shape (batch_size, ...).
      music_db : torch.Tensor
          A tensor containing music decibel values with shape (batch_size, ...).
      mask : torch.Tensor
          A tensor used to mask the values with shape (batch_size, ...).

      Returns
      -------
      torch.Tensor
          A tensor containing the mean NMR for each item in the batch.
      """
      # small function to automatize the compute on mean NMR using the mask
      batch_size = noise_db.shape[0]
      noise_reshaped = noise_db.reshape(batch_size, -1)
      music_reshaped = music_db.reshape(batch_size, -1)
      mask_reshape = mask.reshape(batch_size, -1)

      mean_nmr = torch.sum(
          (noise_reshaped - music_reshaped) * mask_reshape, dim=-1
      ) / (torch.sum(mask_reshape, dim=-1) + eps)
      return mean_nmr

    broadband_mean_nmr = compute_mean_nmr(noise_db=noise_psd_by_bark_db,
                                          music_db=music_thr_db,
                                          mask=mask)

    low_mean_nmr = compute_mean_nmr(
        noise_db=noise_psd_by_bark_db[:, :, : low_bark_limit],
        music_db=music_thr_db[:, :, : low_bark_limit],
        mask=mask[:, :, : low_bark_limit])

    medium_mean_nmr = compute_mean_nmr(
        noise_db=noise_psd_by_bark_db
        [:, :, low_bark_limit: medium_bark_limit],
        music_db=music_thr_db[:, :, low_bark_limit: medium_bark_limit],
        mask=mask[:, :, low_bark_limit: medium_bark_limit])

    high_mean_nmr = compute_mean_nmr(
        noise_db=noise_psd_by_bark_db[:, :, medium_bark_limit:],
        music_db=music_thr_db[:, :, medium_bark_limit:],
        mask=mask[:, :, medium_bark_limit:])

    return broadband_mean_nmr, (low_mean_nmr,
                                medium_mean_nmr, high_mean_nmr), mask

  def spectral_flatness_measure(self, waveform):
    """
    Compute the spectral flatness measure (SFM) for a given waveform.

    Parameters
    ----------
    waveform : numpy.ndarray
        The input audio waveform.

    Returns
    -------
    broadband_sfm_db : float
        The broadband spectral flatness measure in decibels.
    band_sfm_db : tuple of floats
        A tuple containing the spectral flatness measures in decibels for the low, medium, and high frequency bands.

    Notes
    -----
    The spectral flatness measure is computed using the Short-Time Fourier Transform (STFT) of the waveform.
    The waveform is divided into three frequency bands (low, medium, high) based on the Bark scale.
    """
    waveform_spectrum = stft(audio=waveform, nfft=self.nfft, overlap=0.75)

    n_bark = self.nbands  # same number of bark bands as used to compute masking thresholds

    low_bark_limit = n_bark // 3
    medium_bark_limit = 2 * (n_bark // 3)

    broadband_sfm_db = self.compute_SFMdB(
        psd_by_hz_linear=abs(waveform_spectrum)**2
    )

    # Low band sfm
    lowband_sfm_db = self.compute_SFMdB(psd_by_hz_linear=abs(
        waveform_spectrum[:, :, :self.low_bark[low_bark_limit]])**2)

    # Medium band sfm
    mediumband_sfm_db = self.compute_SFMdB(
        psd_by_hz_linear=abs(
            waveform_spectrum
            [:, :, self.low_bark[low_bark_limit]: self.low_bark
             [medium_bark_limit]]) ** 2)

    # High band sfm
    highband_sfm_db = self.compute_SFMdB(
        psd_by_hz_linear=abs(
            waveform_spectrum
            [:, :, self.low_bark[medium_bark_limit]:]) ** 2)

    return broadband_sfm_db, (lowband_sfm_db,
                              mediumband_sfm_db, highband_sfm_db)

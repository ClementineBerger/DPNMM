"""
Code that defines the training, val steps and the logging of the experiment
"""

import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

import numpy as np

from utils.utils import linear2dB
from utils.analysis import istft
from utils.post_process import find_activated_bands


class System(pl.LightningModule):
  """
  Pytorch lightning system to train the model and log useful informations.
  """

  default_monitor: str = "val/loss"

  def __init__(self,
               model,
               optimizer,
               scheduler,
               loss_func,
               train_loader,
               val_loader,
               config) -> None:
    super().__init__()

    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.conf = config

    self.loss_func = loss_func

    if self.conf['loss']['mdmm']:
      self.automatic_optimization = False

  def forward(self, x):
    "Applies forward pass of the model."
    res = self.model(x)
    return res

  def configure_optimizers(self):
    """Initialize optimizers"""

    if self.scheduler is None:
      return self.optimizer

    else:
      return [self.optimizer], [{"scheduler": self.scheduler,
                                 "interval": "step"}]
    # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

  def training_step(self, train_batch, batch_idx):
    """Defines a single step of the training loop."""

    input_cnn, others = train_batch

    music_stft, music_waveform, noise_waveform = others

    gains = self.model(input_cnn)

    if self.conf['gains']['remove_non_activated_bands']:
      # Remove non activated bands
      thresh_mask = self.loss_func.type_of_band(
          init_masking_thresholds_by_bark_in_db=input_cnn[:, 2, :, :],
          noise_psd_by_bark_in_db=input_cnn[:, 1, :, :])
      activated_bands = find_activated_bands(
          1 - thresh_mask, spreadwidth=self.conf['gains']['spreadwidth']).detach()
      gains = gains * activated_bands

    target_filters = self.model.target_filter.compute_target_filter(
        gains)  # linear

    # Using directly the target filters as the frequency responses
    new_music_stft = torch.multiply(music_stft, target_filters)  # linear
    new_music_psd = torch.abs(new_music_stft)**2
    new_mT = self.model.masking_thresholds.compute_thresholds(new_music_psd)

    new_music_waveform = istft(
        stft=new_music_stft,
        nfft=self.conf['audio']['nfft'],
        overlap=0.75,
        length=music_waveform.shape[-1])

    loss, all_losses = self.loss_func.forward(
        new_masking_thresholds_by_bark_in_db=linear2dB(new_mT, 10),
        init_masking_thresholds_by_bark_in_db=input_cnn[:, 2, :, :],
        noise_psd_by_bark_in_db=input_cnn[:, 1, :, :],
        new_music_psd_by_hz_linear=new_music_psd,
        init_music_psd_by_hz_linear=torch.abs(music_stft)**2,
        new_music_waveform=new_music_waveform,
        init_music_waveform=music_waveform
    )

    loss_too_low, loss_mean_power = all_losses

    self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
    self.log(
        "train/loss too low init thresholds",
        loss_too_low,
        on_epoch=True,
        sync_dist=True)
    self.log(
        "train/loss mean power",
        loss_mean_power,
        on_epoch=True,
        sync_dist=True)

    if self.conf['loss']['mdmm']:

      for i in range(self.loss_func.mdmm.n_task - 1):
        self.log("train/Multiplier_" + str(i + 1),
                 self.loss_func.mdmm.multiplier[i])

      # Manual backward if mdmm
      opt = self.optimizers()
      opt.zero_grad()
      self.manual_backward(loss=loss)
      self.clip_gradients(
          self.optimizer,
          gradient_clip_val=5.0,
          gradient_clip_algorithm="norm")
      opt.step()
      self.loss_func.mdmm.update_multiplier(secondary_loss=all_losses[1:])

      if self.scheduler is not None:
        # scheduler step
        scheduler = self.scheduler
        scheduler.step()

    # Log the learning rate
    lr = self.trainer.optimizers[0].param_groups[0]['lr']
    self.log('learning_rate', lr, on_step=False, on_epoch=True)

    return loss

  def validation_step(self, val_batch, batch_idx):
    input_cnn, others = val_batch

    music_stft, music_waveform, noise_waveform = others

    gains = self.model(input_cnn)
    if self.conf['gains']['remove_non_activated_bands']:
      thresh_mask = self.loss_func.type_of_band(
          init_masking_thresholds_by_bark_in_db=input_cnn[:, 2, :, :],
          noise_psd_by_bark_in_db=input_cnn[:, 1, :, :])
      activated_bands = find_activated_bands(
          1 - thresh_mask, spreadwidth=self.conf['gains']['spreadwidth']).detach()
      gains = gains * activated_bands
    target_filters = self.model.target_filter.compute_target_filter(
        gains)  # linear

    new_music_stft = torch.multiply(music_stft, target_filters)  # linear
    new_music_psd = torch.abs(new_music_stft)**2
    new_mT = self.model.masking_thresholds.compute_thresholds(new_music_psd)

    new_music_waveform = istft(
        stft=new_music_stft,
        nfft=self.conf['audio']['nfft'],
        overlap=0.75,
        length=music_waveform.shape[-1])

    loss, all_losses = self.loss_func.forward(
        new_masking_thresholds_by_bark_in_db=linear2dB(new_mT, 10),
        init_masking_thresholds_by_bark_in_db=input_cnn[:, 2, :, :],
        noise_psd_by_bark_in_db=input_cnn[:, 1, :, :],
        new_music_psd_by_hz_linear=new_music_psd,
        init_music_psd_by_hz_linear=torch.abs(music_stft)**2,
        new_music_waveform=new_music_waveform,
        init_music_waveform=music_waveform,
    )

    loss_too_low, loss_mean_power = all_losses

    self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
    self.log(
        "val/loss too low init thresholds",
        loss_too_low,
        on_epoch=True,
        on_step=False,
        sync_dist=True)
    self.log(
        "val/loss mean power",
        loss_mean_power,
        on_epoch=True,
        on_step=False,
        sync_dist=True)

  def on_validation_epoch_end(self) -> None:

    tensorboard = self.logger.experiment

    val_batch, others = next(
        iter(self.val_dataloader()))

    music_stft, music_waveform, noise_waveform = others

    val_batch = val_batch.to('cuda')
    music_stft = music_stft.to('cuda')

    gains = self.model(val_batch)
    if self.conf['gains']['remove_non_activated_bands']:
      thresh_mask = self.loss_func.type_of_band(
          init_masking_thresholds_by_bark_in_db=val_batch[:, 2, :, :],
          noise_psd_by_bark_in_db=val_batch[:, 1, :, :])
      activated_bands = find_activated_bands(
          1 - thresh_mask, spreadwidth=self.conf['gains']['spreadwidth']).detach()
      gains = gains * activated_bands
    target_filters = self.model.target_filter.compute_target_filter(
        gains)  # linear

    new_music_stft = torch.multiply(music_stft, target_filters)  # linear
    new_psd = torch.abs(new_music_stft)**2
    new_mT = self.model.masking_thresholds.compute_thresholds(new_psd)

    new_music_waveform = istft(
        stft=new_music_stft,
        nfft=self.conf['audio']['nfft'],
        overlap=0.75,
        length=music_waveform.shape[-1])
    new_psd = torch.abs(new_music_stft)**2

    envelopes_init = self.model.spectral_envelope.compute_spectral_envelope(
        spectrum=music_stft)
    new_envelopes = self.model.spectral_envelope.compute_spectral_envelope(
        spectrum=new_music_stft)

    new_mT = self.model.masking_thresholds.compute_thresholds(new_psd)

    freqs = self.model.masking_thresholds.fftfrequencies.detach().numpy()

    # logging band types :
    band_types = self.loss_func.type_of_band(
        init_masking_thresholds_by_bark_in_db=val_batch[:, 2, :, :],
        noise_psd_by_bark_in_db=val_batch[:, 1, :, :])
    prop_too_low_band = torch.sum(1 - band_types) / torch.numel(band_types)
    prop_enough_music_band = 1 - prop_too_low_band

    fig, ax = plt.subplots()

    ax.bar('band type proportion',
           prop_too_low_band.detach().cpu().numpy(), label="too low")
    ax.bar(
        'band type proportion',
        prop_enough_music_band.detach().cpu().numpy(),
        label="enough music",
        bottom=prop_too_low_band.detach().cpu().numpy())
    ax.legend()

    tensorboard.add_figure(
        'band type proportion', plt.gcf(),
        global_step=self.current_epoch)
    plt.clf()

    # idx = 28

    nb_audios = val_batch.shape[0]
    audio = [1, nb_audios // 3 + 1, nb_audios - 23]
    nb_frames = new_mT.shape[-2]
    frame = [nb_frames // 4, nb_frames // 2, 3 * nb_frames // 4]
    # frame = [0, nb_frames // 2, nb_frames - 1]

    ### Masking thresholds evolution ###
    fig, ax = plt.subplots(
        nrows=len(audio),
        ncols=len(frame),
        figsize=(18, 20))
    plt.suptitle("PSD and masking thresolds")
    for i in range(len(audio)):
      for j in range(len(frame)):
        ax[i, j].stairs(
            val_batch[audio[i], 0, frame[j], :].detach().cpu().numpy(),
            label="Music PSD", color='indianred')
        ax[i, j].stairs(
            val_batch[audio[i], 1, frame[j], :].detach().cpu().numpy(),
            label="Noise PSD", color="green")
        ax[i, j].stairs(val_batch[audio[i], 2, frame[j], :].detach(
        ).cpu().numpy(), label="Music masking thresholds", color='k')
        ax[i, j].stairs(linear2dB(new_mT[audio[i], frame[j], :], 10).detach().cpu(
        ).numpy(), label="New masking thresholds", color='k', linestyle='--')
        ax[i, j].legend(loc="lower left")

    tensorboard.add_figure(
        'PSD', plt.gcf(),
        global_step=self.current_epoch)
    plt.clf()
    ###

    # Predicted filter
    fig, ax = plt.subplots(
        nrows=len(audio),
        ncols=len(frame),
        figsize=(18, 20))
    plt.suptitle("Predicted filter and approximation")
    for i in range(len(audio)):
      for j in range(len(frame)):
        ax[i, j].plot(
            freqs, linear2dB(target_filters[audio[i], frame[j], :],
                             20).detach().cpu().numpy(), color="k",
            label="Predicted target filters")
        ax[i, j].legend(loc="lower left")

    tensorboard.add_figure(
        'filters', plt.gcf(),
        global_step=self.current_epoch)
    plt.clf()

    # Gains prediction
    fig, ax = plt.subplots(nrows=len(audio), ncols=len(frame), figsize=(18, 6))
    plt.suptitle("Predicted gains")
    for i in range(len(audio)):
      for j in range(len(frame)):
        ax[i, j].stairs(gains[audio[i], frame[j], :].detach().cpu().numpy())
    tensorboard.add_figure(
        'gains', plt.gcf(),
        global_step=self.current_epoch)
    plt.clf()

    # Power spectrums
    fig, ax = plt.subplots(
        nrows=len(audio),
        ncols=len(frame),
        figsize=(18, 12))
    plt.suptitle("Power Spectrum")
    for i in range(len(audio)):
      for j in range(len(frame)):
        ax[i, j].plot(
            freqs, linear2dB(abs(music_stft[audio[i], frame[j], :]),
                             20).detach().cpu().numpy(),
            label="Init spectrum", color='indianred')
        ax[i, j].plot(
            freqs,
            linear2dB(abs(new_music_stft[audio[i], frame[j], :]),
                      20).detach().cpu().numpy(),
            label="Final spectrum", alpha=0.7, color='dodgerblue')
        ax[i, j].plot(
            freqs,
            linear2dB(abs(envelopes_init[audio[i], frame[j], :]), 20).detach().cpu().numpy(),
            label="Init envelope",
            color="red"
        )
        ax[i, j].plot(
            freqs,
            linear2dB(abs(new_envelopes[audio[i], frame[j], :]), 20).detach().cpu().numpy(),
            label="New envelope",
            color="blue"
        )

        ax[i, j].legend(loc="lower left")

    tensorboard.add_figure(
        'Spectrum', plt.gcf(),
        global_step=self.current_epoch)
    plt.clf()

    # Logging audios

    music_waveform = music_waveform.detach().cpu().numpy()
    new_music_waveform = new_music_waveform.detach().cpu().numpy()
    noise_waveform = noise_waveform.detach().cpu().numpy()

    for i in range(len(audio)):
      tensorboard.add_audio(
          'Initial music audio n°' + str(i),
          music_waveform[audio[i]],
          global_step=self.current_epoch,
          sample_rate=self.conf['audio']['sr'])

      tensorboard.add_audio(
          'Noise audio n°' + str(i),
          noise_waveform[audio[i]],
          global_step=self.current_epoch,
          sample_rate=self.conf['audio']['sr'])

      tensorboard.add_audio(
          'New music audio n°' + str(i),
          new_music_waveform[audio[i]],
          global_step=self.current_epoch,
          sample_rate=self.conf['audio']['sr'])

      tensorboard.add_audio(
          'Initial mix n°' + str(i),
          music_waveform[audio[i]] + noise_waveform[audio[i]],
          global_step=self.current_epoch,
          sample_rate=self.conf['audio']['sr'])

      tensorboard.add_audio(
          'New mix n°' + str(i),
          new_music_waveform[audio[i]] + noise_waveform[audio[i]],
          global_step=self.current_epoch,
          sample_rate=self.conf['audio']['sr'])

    return super().on_validation_epoch_end()

  def train_dataloader(self):
    """Training dataloader"""
    return self.train_loader

  def val_dataloader(self):
    """Validation dataloader"""
    return self.val_loader

  def on_save_checkpoint(self, checkpoint):
    """Overwrite if you want to save more things in the checkpoint."""
    checkpoint["training_config"] = self.conf
    return checkpoint

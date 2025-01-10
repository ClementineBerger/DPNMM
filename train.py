"""
Training pipeline using 'system'
"""

import numpy as np
import random
import os
import yaml
import json
from pprint import pprint
import argparse

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from config import conf, common_parameters
from system import System
from custom_dataloader import MusicNoiseDataset


from model import DPNMM
from losses import LossFunction
from utils.multiloss_framework import ModifiedDifferentialMultipliersMethod

from utils.utils import merge_dicts

parser = argparse.ArgumentParser()
parser.add_argument("--conf_id", default="power_2", 
                    help="Conf tag, used to get the right config")
parser.add_argument("--debug", type=bool, default=False,
                    help="If true save to specific directory")


def main(conf):
  """
  Main function to run the training

  Parameters
  ----------
  conf : dict
      Configuration dictionary.
  """

  # Reproducibility
  pl.seed_everything(conf["seed"])

  random.seed(conf["seed"])
  np.random.seed(conf["seed"])
  torch.manual_seed(conf["seed"])
  torch.cuda.manual_seed_all(conf["seed"])
  torch.set_float32_matmul_precision("high")

  train_dataset = MusicNoiseDataset(
      root_dir=conf['dataset']['root_dir'],
      csv_file=conf['dataset']['csv_file'],
      nfft=conf['audio']['nfft'],
      sr=conf['audio']['sr'],
      set='train',
      hpss_music=conf['data']['hpss_music'],
      hpss_noise=conf['data']['hpss_noise']
  )

  val_dataset = MusicNoiseDataset(
      root_dir=conf['dataset']['root_dir'],
      csv_file=conf['dataset']['csv_file'],
      nfft=conf['audio']['nfft'],
      sr=conf['audio']['sr'],
      set='val',
      hpss_music=conf['data']['hpss_music'],
      hpss_noise=conf['data']['hpss_noise']
  )

  # dataloader reproducibility
  generator = torch.Generator()
  generator.manual_seed(conf["seed"])

  def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

  # Define dataloaders
  train_loader = DataLoader(
      dataset=train_dataset,
      batch_size=conf["optim"]["batch_size"],
      shuffle=True,
      num_workers=conf["process"]["num_workers"],
      persistent_workers=True,
      pin_memory=True,
      prefetch_factor=conf["process"]["prefetch"],
      worker_init_fn=seed_worker,
      generator=generator,
      drop_last=False
  )

  val_loader = DataLoader(
      dataset=val_dataset,
      batch_size=conf["optim"]["batch_size"],
      num_workers=conf["process"]["num_workers"],
      persistent_workers=True,
      pin_memory=True,
      prefetch_factor=conf["process"]["prefetch"],
      worker_init_fn=seed_worker,
      generator=generator
  )

  # Setting model

  model = DPNMM(
      nfft=conf['model']['nfft'],
      nb_bark=conf['model']['nb_bark'],
      input_ch=conf['model']['input_ch'],
      conv_ch=conf['model']['conv_ch'],
      conv_kernel_inp=conf['model']['conv_kernel_inp'],
      conv_kernel=conf['model']['conv_kernel'],
      emb_hidden_dim=conf['model']['emb_hidden_dim'],
      emb_num_layers=conf['model']['emb_num_layers'],
      lin_groups=conf['model']['lin_groups'],
      enc_lin_groups=conf['model']['enc_lin_groups'],
      rnn_type=conf['model']['rnn_type'],
      trans_conv_type=conf['model']['trans_conv_type'],
      max_positive_clamping_value=conf['model']['gain_clamping']
      ['max_positive_clamping_value'],
      min_negative_clamping_value=conf['model']['gain_clamping']
      ['min_negative_clamping_value'],
      remove_high_bands=conf['model']['gain_clamping']['remove_high_bands'])

  # Define optimizers (=/= de configure_optimizer dans system.py ?)
  optimizer = torch.optim.Adam(
      params=model.parameters(),
      lr=conf['optim']['lr'],
      betas=conf['optim']['betas'],
      weight_decay=conf['optim']['weight_decay']
  )

  # Define schedulers
  try:
    if conf["optim"]["lr_scheduler"]:
      if conf["optim"]["lr_scheduler"] == "OneCycleLR":
        # Taille du batch et nombre de dispositifs
        batch_size = conf["optim"]["batch_size"]
        devices = conf["process"]["devices"]
        num_nodes = conf["process"]["num_nodes"]

        # Nombre total de batches par epoch
        total_batches_per_epoch = (
            len(train_dataset) + batch_size * devices * num_nodes - 1) // (
            batch_size * devices * num_nodes)

        # Nombre total de steps sur tous les epochs
        total_steps = conf["optim"]["epochs"] * total_batches_per_epoch

        conf["optim"]["lr_scheduler_args"]["total_steps"] = total_steps
      scheduler = getattr(
          torch.optim.lr_scheduler,
          conf["optim"]["lr_scheduler"])(
          optimizer,
          **conf["optim"]["lr_scheduler_args"])

    else:
      scheduler = None
  except BaseException:
    scheduler = None

  if conf['loss']['mdmm']:
    mdmm = ModifiedDifferentialMultipliersMethod(
        constraint_thr=conf['mdmm']['constraint_thr'],
        damping_coeff=conf['mdmm']['damping_coeff'],
        multiplier_lr=conf['mdmm']['multiplier_lr'],
        initial_value=conf['mdmm']['initial_value'],
        n_task=2
    )
  else:
    mdmm = None

  # Setting the loss
  loss = LossFunction(
      nfft=conf['loss']['nfft'],
      sr=conf['loss']['sr'],
      mean_power_constraint=conf['loss']['mean_power_constraint'],
      nb_bark=conf['loss']['nb_bark'],
      thresholds_margins=conf['loss']['thresholds_margins'],
      loss_type=conf['loss']['loss_type'],
      mdmm=mdmm,
      epsilon=conf['loss']['epsilon'],
  )

  # Saving config file
  log_dir = os.path.join(conf["exp_dir"],
                         conf["conf_id"],
                         conf["job_id"])
  os.makedirs(log_dir, exist_ok=True)

  conf_path = os.path.join(log_dir, "conf.yml")
  with open(conf_path, "w") as outfile:
    yaml.safe_dump(conf, outfile)

  # System defining training and logging procedures
  system = System(
      model=model,
      loss_func=loss,
      optimizer=optimizer,
      scheduler=scheduler,
      train_loader=train_loader,
      val_loader=val_loader,
      config=conf
  )

  # Setting callbacks
  callbacks = []
  checkpoint_dir = os.path.join(log_dir, "checkpoint")

  # Save your model
  checkpoint = ModelCheckpoint(
      checkpoint_dir,
      monitor="val/loss",
      mode="min",
      save_last=True,
      save_top_k=5,
      verbose=True)
  callbacks.append(checkpoint)

  # EarlyStopping
  if conf['optim']['patience'] is not None:
    callbacks.append(
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=conf['optim']['patience'],
            verbose=True))

  # TRAINER
  # Define trainer

  try:
    # limit train batch
    lmt_train_bt = conf["debug"]["lmt_train_bt"]
    lmt_val_bt = conf["debug"]["lmt_val_bt"]       # limit val batch
  except BaseException:
    lmt_train_bt = None
    lmt_val_bt = None

  if conf['loss']['mdmm']:
    gradient_clip_val = None  # defined in the manual train step
  else:
    gradient_clip_val = 5.

  trainer = pl.Trainer(
      max_epochs=conf["optim"]["epochs"],
      callbacks=callbacks,
      default_root_dir=log_dir,
      devices=conf["process"]["devices"],
      accelerator="gpu",
      num_nodes=conf["process"]["num_nodes"],
      limit_train_batches=lmt_train_bt,  # Useful for fast experiment
      limit_val_batches=lmt_val_bt,  # Useful for fast experiment
      gradient_clip_val=gradient_clip_val,
      logger=TensorBoardLogger(log_dir, log_graph=False,
                               default_hp_metric=False),
      deterministic=True,
      # precision="16-mixed"
  )

  # Train from scratch
  if conf["checkpoint_path"] is None:
    trainer.fit(system)

  # Train from checkpoint
  else:
    print(f"resume training from checkpoint: {conf['checkpoint_path']}")
    trainer.fit(system, ckpt_path=conf["checkpoint_path"])

  # Record top 5 systems
  best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
  with open(os.path.join(log_dir, "best_k_models.json"), "w") as f:
    json.dump(best_k, f, indent=0)

  # Save best in a special place
  state_dict = torch.load(checkpoint.best_model_path)
  system.load_state_dict(state_dict=state_dict["state_dict"])
  system.cpu()
  torch.save(system.model.state_dict(),
             os.path.join(log_dir, "best_model.pth"))


if __name__ == "__main__":
  args = parser.parse_args()
  args = vars(args)
  conf = merge_dicts(common_parameters, conf[args["conf_id"]])
  conf = {**conf, **args}
  pprint(conf)

  main(conf)

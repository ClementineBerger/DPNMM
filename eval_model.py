"""
Code for evaluation of the neural models.
"""


from eval_dataloader import MusicNoiseDataset
from utils.objective_metrics import ObjectiveMetrics
from model_wrapper import ModelWrapper

import torch
from torch.utils.data import DataLoader

import numpy as np

import pandas as pd

import json

import os
import argparse
from pprint import pprint
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--conf_id", default="001",   # change default for debbug
                    help="Conf tag, used to get the right config")
parser.add_argument("--job_id", default="001",
                    help="Job id within the trained models from this configuration.")
parser.add_argument("--remove_bands", default=False,   # change default for debbug
                    help="Remove non activated bands or not")


def main(conf_id, job_id, remove_non_activated_bands):

  # def seed_worker(worker_id):
  #   worker_seed = torch.initial_seed() % 2**32
  #   np.random.seed(worker_seed)
  #   random.seed(worker_seed)

  model = ModelWrapper(conf_name=conf_id, job_id=job_id)
  model.eval()
  if torch.cuda.is_available():
    model = model.cuda()

  root_dir = os.environ['DATA']
  music_noise_dir = os.path.join(root_dir, "music_noise")
  csv_file = os.path.join(music_noise_dir, "metadata.csv")

  test_dataset = MusicNoiseDataset(
      root_dir=root_dir,
      csv_file=csv_file,
      nfft=model.conf['audio']['nfft'],
      sr=model.conf['audio']['sr'],
      set='test',
  )

  test_dataloader = DataLoader(test_dataset,
                               batch_size=32,
                               shuffle=False,
                               num_workers=4,
                               drop_last=False,
                               pin_memory=True)

  metrics = ObjectiveMetrics(
      nfft=model.conf['audio']['nfft'],
      sr=model.conf['audio']['sr'],
      gld_nfft=int(
          0.1 *
          model.conf['audio']['sr']),
  )  # 100 ms
  if torch.cuda.is_available():
    metrics = metrics.cuda()

  # save indexes in metadata in order to retrieve the audios info if necessary
  indexes_in_metadata = []
  # level difference in dBA between filtered music and init music
  all_global_level_difference = []
  all_gld_low = []
  all_gld_medium = []
  all_gld_high = []

  # Noise-to-Mask Ratios
  all_nmr_init = []  # noise-to-mask ratio initial
  all_nmr_system = []  # noise-to-mask ratio after system
  all_nmr_init_low = []
  all_nmr_init_medium = []
  all_nmr_init_high = []
  all_nmr_system_low = []
  all_nmr_system_medium = []
  all_nmr_system_high = []

  # Spectral Flatness Measure (SFM_dB) (moyen par frames ?)
  all_sfm_init = []
  all_sfm_init_low = []
  all_sfm_init_medium = []
  all_sfm_init_high = []

  all_sfm_system = []
  all_sfm_system_low = []
  all_sfm_system_medium = []
  all_sfm_system_high = []

  # Model to evaluation

  for data in tqdm(test_dataloader):
    music_waveforms, noise_waveforms, idx = data

    indexes_in_metadata.append(idx)

    with torch.no_grad():
      if torch.cuda.is_available():
        music_waveforms = music_waveforms.cuda()
        noise_waveforms = noise_waveforms.cuda()

      new_music_waveforms = model.forward(
          music_waveforms,
          noise_waveforms,
          mean_gains=True,
          remove_non_activated_bands=remove_non_activated_bands,
          return_dict=False,
      )

      # Compute global level difference
      global_level_difference, bands_gld = metrics.time_average_level_difference(
          init_waveform=music_waveforms, output_waveform=new_music_waveforms, )

      all_global_level_difference.append(global_level_difference)
      all_gld_low.append(bands_gld[0])
      all_gld_medium.append(bands_gld[1])
      all_gld_high.append(bands_gld[2])

      # Compute NMR
      nmr_init, band_nmr_init, mask = metrics.noise_to_mask_ratio(
          music_waveform=music_waveforms,
          noise_waveform=noise_waveforms,
          mask=None
      )
      nmr_system, band_nmr_system, _ = metrics.noise_to_mask_ratio(
          music_waveform=new_music_waveforms,
          noise_waveform=noise_waveforms,
          mask=mask
      )

      all_nmr_init.append(nmr_init)
      all_nmr_init_low.append(band_nmr_init[0])
      all_nmr_init_medium.append(band_nmr_init[1])
      all_nmr_init_high.append(band_nmr_init[2])

      all_nmr_system.append(nmr_system)
      all_nmr_system_low.append(band_nmr_system[0])
      all_nmr_system_medium.append(band_nmr_system[1])
      all_nmr_system_high.append(band_nmr_system[2])

      # SFM dB
      sfm_init, band_sfm_init = metrics.spectral_flatness_measure(
          waveform=music_waveforms
      )

      all_sfm_init.append(sfm_init)
      all_sfm_init_low.append(band_sfm_init[0])
      all_sfm_init_medium.append(band_sfm_init[1])
      all_sfm_init_high.append(band_sfm_init[2])

      sfm_system, band_sfm_system = metrics.spectral_flatness_measure(
          waveform=new_music_waveforms
      )

      all_sfm_system.append(sfm_system)
      all_sfm_system_low.append(band_sfm_system[0])
      all_sfm_system_medium.append(band_sfm_system[1])
      all_sfm_system_high.append(band_sfm_system[2])

  results = {
      "idx_metadata": torch.concatenate(indexes_in_metadata).cpu().tolist(),
      "gld": {
          'broadband': torch.concatenate(all_global_level_difference).cpu().tolist(),
          "low": torch.concatenate(all_gld_low).cpu().tolist(),
          "medium": torch.concatenate(all_gld_medium).cpu().tolist(),
          "high": torch.concatenate(all_gld_high).cpu().tolist(),
      },
      "nmr_init": {
          "broadband": torch.concatenate(all_nmr_init).cpu().tolist(),
          "low": torch.concatenate(all_nmr_init_low).cpu().tolist(),
          "medium": torch.concatenate(all_nmr_init_medium).cpu().tolist(),
          "high": torch.concatenate(all_nmr_init_high).cpu().tolist()},
      "nmr_system": {
          "broadband": torch.concatenate(all_nmr_system).cpu().tolist(),
          "low": torch.concatenate(all_nmr_system_low).cpu().tolist(),
          "medium": torch.concatenate(all_nmr_system_medium).cpu().tolist(),
          "high": torch.concatenate(all_nmr_system_high).cpu().tolist()},
      "sfm_init": {
          "broadband": torch.concatenate(all_sfm_init).cpu().tolist(),
          "low": torch.concatenate(all_sfm_init_low).cpu().tolist(),
          "medium": torch.concatenate(all_sfm_init_medium).cpu().tolist(),
          "high": torch.concatenate(all_sfm_init_high).cpu().tolist()},
      "sfm_system": {
          "broadband": torch.concatenate(all_sfm_system).cpu().tolist(),
          "low": torch.concatenate(all_sfm_system_low).cpu().tolist(),
          "medium": torch.concatenate(all_sfm_system_medium).cpu().tolist(),
          "high": torch.concatenate(all_sfm_system_high).cpu().tolist()},
  }

  saving_dir = os.path.join(
      os.environ['REPO'],
      "trained_models",
      model.conf["conf_id"],
      "evaluation"
  )

  os.makedirs(saving_dir, exist_ok=True)

  if remove_non_activated_bands:
    saving_path = os.path.join(
        saving_dir,
        "combined_results_remove_bands.json"
    )

  else:
    saving_path = os.path.join(
        saving_dir,
        "combined_results.json"
    )

  # Sauvegarder en format .json
  with open(saving_path, 'w') as f:
    json.dump(results, f)

  print("Done")

  return results


if __name__ == "__main__":
  args = parser.parse_args()
  args = vars(args)

  if isinstance(args['remove_bands'], str):
    if args['remove_bands'] == "False":
      remove_bands = False
    elif args['remove_bands'] == "True":
      remove_bands = True
  else:
    remove_bands = args['remove_bands']

  main(args["conf_id"], args["job_id"], remove_bands)

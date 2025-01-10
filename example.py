"""Example script for the complete system."""

import argparse


from model_wrapper import ModelWrapper
import torchaudio
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--conf_id", default="power_2",
                    help="Conf tag, used to get the right config")
parser.add_argument("--job_id", default="001",
                    help="Job id within the trained models from this configuration.")


def main(music_waveform, noise_waveform, conf_id, job_id):
  model = ModelWrapper(conf_name=conf_id, job_id=job_id)
  model.eval()

  processed_music_waveform = model.forward(
      music_waveform=music_waveform,
      noise_waveform=noise_waveform,
      remove_non_activated_bands=True,
      return_dict=False,
  )

  return processed_music_waveform


if __name__ == "__main__":
  args = parser.parse_args()
  args = vars(args)

  conf_name = args["conf_id"]

  noise_path = "path/to/noise.wav"
  music_path = "path/to/music.wav"

  noise_waveform, _ = torchaudio.load(noise_path)
  music_waveform, _ = torchaudio.load(music_path)

  # currently only mono signals are supported
  if noise_waveform.shape[0] > 1:
    noise_waveform = torch.mean(noise_waveform, dim=0).reshape(1, -1)
  if music_waveform.shape[0] > 1:
    music_waveform = torch.mean(music_waveform, dim=0).reshape(1, -1)

  processed_music_waveform = main(music_waveform, noise_waveform, conf_name)

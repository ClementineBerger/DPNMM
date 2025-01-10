"""
Some useful functions.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

# Utils function to merge two dictionaries


def merge_dicts(dict1, dict2):
  """
  Recursively merge two dictionaries, with values from dict2
  taking precedence.

  Parameters
  ----------
  dict1 : dict
    First dictionary.
  dict2 : dict
    Second dictionary.

  Returns
  -------
  dict
    The merged dictionary.

  Author: Joris Cosentino
  """
  merged = dict1.copy()
  for key, value in dict2.items():
    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):  # nopep8
      merged[key] = merge_dicts(merged[key], value)
    else:
      merged[key] = value
  return merged


def safe_log(x, eps=1e-12):
  """
  Computes the logarithm base 10 of the input tensor after applying a ReLU activation and adding a small epsilon value for numerical stability.

  Parameters
  ----------
  x : torch.Tensor
    Input tensor for which the logarithm is to be computed.
  eps : float, optional
    A small value added to the input to avoid taking the logarithm of zero. Default is 1e-12.

  Returns
  -------
  torch.Tensor
    The logarithm base 10 of the input tensor after applying ReLU and adding epsilon.
  """
  return torch.log10(nn.ReLU()(x) + eps)


def linear2dB(x, gain, eps=1e-12):
  """
  Converts a linear scale value to decibels (dB).

  Parameters
  ----------
  x : torch.Tensor
    Input tensor containing linear scale values.
  gain : float
    Gain factor to be applied to the logarithmic result.
  eps : float, optional
    Small value to avoid taking the logarithm of zero, by default 1e-12.

  Returns
  -------
  torch.Tensor
    Tensor containing the values converted to decibels.
  """
  return gain * safe_log(torch.abs(x), eps=eps)


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


def custom_clamping_values(max_positive_clamping_value=10. / 3.,
                           min_negative_clamping_value=-5. / 3.,
                           remove_high_bands=True,
                           n_bark=26
                           ):
  """
  Compute clamping values for gains.

  Parameters
  ----------
  max_positive_clamping_value : float, optional
    Maximum positive clamping value. Default is 10/3.
  min_negative_clamping_value : float, optional
    Minimum negative clamping value. Default is -5/3.
  remove_high_bands : bool, optional
    If True, remove high bands by setting their gains to 0. Default is True.
  n_bark : int, optional
    Number of Bark bands. Default is 26.

  Returns
  -------
  max_gains : torch.Tensor
    Tensor of maximum gains for each Bark band.
  min_gains : torch.Tensor
    Tensor of minimum gains for each Bark band.
  """
  max_gains = max_positive_clamping_value * torch.ones(n_bark)
  min_gains = min_negative_clamping_value * torch.ones(n_bark)

  if remove_high_bands:
    min_gains[24:] = 0.  # voir si 23 ou 24
    max_gains[24:] = 0.

  return max_gains, min_gains


def custom_clamp(gains, min_val, max_val):
  """
  Clamps the values in the `gains` tensor to be within the range [min_val, max_val].

  Parameters
  ----------
  gains : torch.Tensor
    A tensor of shape (batch_size, n_frames, n_bark) representing the gains.
  min_val : float
    The minimum value to clamp to.
  max_val : float
    The maximum value to clamp to.

  Returns
  -------
  torch.Tensor
    A tensor of the same shape as the input `gains` with values clamped to the range [min_val, max_val].
  """
  # gains : (batch_size, n_frames, n_bark)

  batch_size, n_frames = gains.shape[-3], gains.shape[-2]

  # reshape to [batch_size*n_frames, n_bark]
  gains = gains.reshape(-1, gains.shape[-1])

  clamped_gains = torch.clamp(gains, min=min_val, max=max_val)

  clamped_gains = clamped_gains.reshape(batch_size, n_frames, -1)

  return clamped_gains

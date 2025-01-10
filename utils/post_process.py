"""
Some useful functions post process of the gains predicted by the system.
"""
import torch
import torch.nn.functional as F
import numpy as np


def find_activated_bands(threshold_mask, spreadwidth=2):
  """
  Identify activated bands in a threshold mask using convolution.

  This function applies a convolution operation to a threshold mask to identify
  activated bands. The input can be either a NumPy array or a PyTorch tensor (for
  the signal processing baselines and the neural model).
  The convolution is performed with a kernel of ones, and the result is a binary
  mask indicating if the bands are activated.

  Parameters
  ----------
  threshold_mask : np.ndarray or torch.Tensor
    The input threshold mask. It can be a NumPy array or a PyTorch tensor.
  spreadwidth : int, optional
    The width of the convolution kernel. Default is 2.

  Returns
  -------
  np.ndarray or torch.Tensor
    A binary mask indicating if the bands are activated. The type of the
    returned mask matches the type of the input threshold_mask.

  Raises
  ------
  ValueError
    If the input threshold_mask is neither a NumPy array nor a PyTorch tensor.

  Examples
  --------
  >>> import numpy as np
  >>> threshold_mask = np.array([0, 1, 0, 0, 1, 1, 0])
  >>> find_activated_bands(threshold_mask, spreadwidth=1)
  array([1, 1, 1, 1, 1, 1, 1])

  >>> import torch
  >>> threshold_mask = torch.tensor([0, 1, 0, 0, 1, 1, 0], dtype=torch.float32)
  >>> find_activated_bands(threshold_mask, spreadwidth=1)
  tensor([1., 1., 1., 1., 1., 1., 1.])
  """
  if isinstance(threshold_mask, np.ndarray):

    kernel = np.ones(2 * spreadwidth + 1)

    def apply_convolution(seq):
      conv_result = np.convolve(seq, kernel, mode="same")
      return np.where(conv_result > 0, 1, 0)

    return np.apply_along_axis(apply_convolution, axis=-1, arr=threshold_mask)

  elif isinstance(threshold_mask, torch.Tensor):

    kernel = torch.ones(
        2 * spreadwidth + 1,
        dtype=threshold_mask.dtype).unsqueeze(0).unsqueeze(0).to(
        threshold_mask.device)
    nb_dim = threshold_mask.ndim
    initial_dims = threshold_mask.shape

    if nb_dim == 1:
      mask = threshold_mask.unsqueeze(0).unsqueeze(0)

    elif nb_dim == 2:
      mask = threshold_mask.unsqueeze(1)

    elif nb_dim == 3:
      mask = threshold_mask.reshape((initial_dims[0] * initial_dims[1], 1, -1))

    res = F.conv1d(mask, kernel, padding=spreadwidth)
    res = torch.where(res > 0, torch.tensor(1.0), torch.tensor(0.))

    res = res.reshape(initial_dims)

    return res


def smoothing_gains(gains_db, N, tau_attack, tau_release, sr):
  """
  Apply smoothing to the gains in decibels using a dynamic compressor approach.

  Parameters
  ----------
  gains_db : np.ndarray or torch.Tensor
    The input gains in decibels. Can be a 2D or 3D array/tensor.
  N : int
    Window size.
  tau_attack : float
    The attack time constant.
  tau_release : float
    The release time constant.
  sr : int
    The sampling rate.

  Returns
  -------
  np.ndarray or torch.Tensor
    The smoothed gains with the same shape as the input `gains_db`.

  Notes
  -----
  - If `gains_db` is a 3D array/tensor, the first dimension is treated as the batch dimension.
  - If `tau_attack` is equal to `tau_release`, a single smoothing factor is used.
  - If `tau_attack` is different from `tau_release`, separate attack and release smoothing factors are used.
  """

  nb_dim = gains_db.ndim
  initial_dims = gains_db.shape

  if nb_dim > 2:
    batch_dim, nframe, nbark = initial_dims
    # First and second dimension are not combined hre it would create likeage of one
    # audio onto an other.
  else:
    gains_db = gains_db.reshape((1, initial_dims[-2], initial_dims[-1]))

  if isinstance(gains_db, np.ndarray):
    smoothed_gains = np.zeros_like(gains_db)
  elif isinstance(gains_db, torch.Tensor):
    smoothed_gains = torch.zeros_like(gains_db)

  smoothed_gains[:, 0, :] = gains_db[:, 0, :]

  if tau_attack == tau_release:
    gamma = N / tau_attack / sr
    for i in range(1, gains_db.shape[-2]):
      smoothed_gains[:, i] = gamma * gains_db[:, i] + \
          (1 - gamma) * smoothed_gains[:, i - 1]

  else:
    gamma_attack = N / tau_attack / sr
    gamma_release = N / tau_release / sr

    for i in range(1, gains_db.shape[-2]):
      mask = gains_db[:, i] > smoothed_gains[:, i - 1]

      gain_attack = gamma_attack * gains_db[:, i] + (
          1 - gamma_attack) * smoothed_gains[:, i - 1]
      gain_release = gamma_release * gains_db[:, i] + (
          1 - gamma_release) * smoothed_gains[:, i - 1]

      smoothed_gains[:, i] = mask * gain_attack + ~mask * gain_release

  return smoothed_gains.reshape(initial_dims)

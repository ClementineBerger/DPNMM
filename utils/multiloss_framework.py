"""
ModifiedDifferentialMultipliersMethod: Implements the Modified Differential Multipliers Method.
"""
import torch
import torch.nn as nn


class ModifiedDifferentialMultipliersMethod(nn.Module):
  """
  A PyTorch module implementing the Modified Differential Multipliers Method for combining and optimizing primary and secondary losses with constraints.

  Parameters
  ----------
  constraint_thr : float or list of floats
    The threshold values for the constraints.
  damping_coeff : float or list of floats
    The damping coefficients for the constraints.
  multiplier_lr : float
    The learning rate for updating the multipliers.
  initial_value : float
    The initial value for the Lagrangian multipliers.
  n_task : int
    The number of tasks or constraints.
  *args : tuple
    Additional positional arguments for the superclass.
  **kwargs : dict
    Additional keyword arguments for the superclass.

  Methods
  -------
  combine_losses(primary_loss, secondary_loss)
    Combines the primary and secondary losses, applying the constraints.
  update_multiplier(secondary_loss)
    Updates the Lagrangian multipliers based on the secondary loss values.
  """

  def __init__(
          self,
          constraint_thr,
          damping_coeff,
          multiplier_lr,
          initial_value,
          n_task,
          *args,
          **kwargs):
    super().__init__(*args, **kwargs)

    self.register_buffer("constraint_thr", torch.as_tensor(constraint_thr))
    self.register_buffer("damping_coeff", torch.as_tensor(damping_coeff))
    self.register_buffer("multiplier_lr", torch.as_tensor(multiplier_lr))
    self.n_task = n_task

    # Initialize Lagrangian Multipliers
    # multipliers only before secondary constraints

    self.multiplier = initial_value * \
        torch.ones(n_task - 1, requires_grad=False)

  def combine_losses(self, primary_loss, secondary_loss):
    """
    Combines the primary and secondary losses with damping and constraints.

    Parameters
    ----------
    primary_loss : torch.Tensor
      The primary loss value.
    secondary_loss : torch.Tensor or tuple of torch.Tensor
      The secondary loss value(s). If a tuple, it should contain multiple secondary loss values.

    Returns
    -------
    torch.Tensor
      The combined loss value after applying damping and constraints.
    """
    if isinstance(secondary_loss, tuple):
      with torch.no_grad():
        damp = [
            self.damping_coeff[i] *
            (self.constraint_thr[i] - secondary_loss[i])
            for i in range(self.n_task - 1)]
      all_secondary_contraints = tuple(
          [(self.multiplier[i] - damp[i]) *
           (self.constraint_thr[i] - secondary_loss[i])
           for i in range(self.n_task - 1)])
      overall_secondary_contraints = sum(all_secondary_contraints)
    else:
      with torch.no_grad():
        damp = self.damping_coeff * (self.constraint_thr - secondary_loss)
      overall_secondary_contraints = (
          self.multiplier - damp) * (
          self.constraint_thr - secondary_loss)

    complete_loss = primary_loss - overall_secondary_contraints

    return complete_loss

  def update_multiplier(self, secondary_loss):
    """
    Update the multiplier based on the secondary loss values.

    Parameters
    ----------
    secondary_loss : list or tuple
      A list or tuple containing the secondary loss values for each task.

    Notes
    -----
    This method updates the `multiplier` attribute by applying a ReLU activation
    function to the current multiplier value adjusted by the learning rate and
    the difference between the secondary loss values and the constraint threshold.
    The update is performed differently depending on whether `secondary_loss` is
    a tuple or not.
    """
    loss_values = tuple([secondary_loss[i].detach()
                         for i in range(self.n_task - 1)])
    # after optimization step
    if isinstance(secondary_loss, tuple):
      for i in range(self.n_task - 1):
        self.multiplier[i] = nn.ReLU()(
            self.multiplier[i] + self.multiplier_lr *
            (loss_values[i] - self.constraint_thr[i]))

    else:
      self.multiplier = nn.ReLU()(
          self.multiplier + self.multiplier_lr *
          (loss_values - self.constraint_thr))

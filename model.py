import torch
from torch.nn.parameter import Parameter
from torch import Tensor, nn

from functools import partial
from typing import Tuple, Union, Iterable, Callable, Optional
import math
import os

from utils.analysis import stft
from utils.masking_thresholds import MaskingThresholds
from utils.spectral_shaping import SpectralEnvelope, TargetFilter
from utils.utils import custom_clamp, custom_clamping_values


class Conv2dNormAct(nn.Sequential):
  """A combination of Conv2d, normalization, and activation layers in sequence.

  Parameters
  ----------
  in_ch : int
    Number of input channels.
  out_ch : int
    Number of output channels.
  kernel_size : Union[int, Iterable[int]]
    Size of the convolution kernel.
  fstride : int, default=1
    Stride of the convolution on the feature dimension.
  dilation : int, default=1
    Dilation factor for convolution.
  fpad : bool, default=True
    Whether to apply padding on the feature dimension.
  bias : bool, default=True
    Whether to include bias in the convolution.
  separable : bool, default=False
    Whether to use separable convolutions.
  norm_layer : default=[Callable[..., torch.nn.Module]]
    Normalization layer.
  activation_layer : default=[Callable[..., torch.nn.Module]]
    Activation layer.
  causal : bool, default=True.
    Whether to apply causal padding on the time axis.

  Methods
  -------
  forward(x)
    Applies the convolutional layer to the input tensor.

  """

  def __init__(
      self,
      in_ch: int,
      out_ch: int,
      kernel_size: Union[int, Iterable[int]],
      fstride: int = 1,
      dilation: int = 1,
      fpad: bool = True,
      bias: bool = True,
      separable: bool = False,
      norm_layer: Optional[Callable[..., torch.nn.Module]
                           ] = torch.nn.BatchNorm2d,
      activation_layer: Optional[Callable[..., torch.nn.Module]
                                 ] = torch.nn.ReLU,
      causal=True,
  ):
    layers = []
    lookahead = 0  # This needs to be handled on the input feature side
    # Padding on time axis
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else tuple(kernel_size)
    )
    if fpad:
      fpad_ = kernel_size[1] // 2 + dilation - 1
    else:
      fpad_ = 0
    if causal:
      pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
    else:
      pad = (0,)
      layers.append(nn.Identity())  # trick to have consistent #lyayers
    if any(x > 0 for x in pad):
      layers.append(nn.ConstantPad2d(pad, 0.0))
    groups = math.gcd(in_ch, out_ch) if separable else 1
    if groups == 1:
      separable = False
    if max(kernel_size) == 1:
      separable = False
    layers.append(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=(0, fpad_),
            stride=(1, fstride),  # Stride over time is always 1
            dilation=(1, dilation),  # Same for dilation
            groups=groups,
            bias=bias,
        )
    )
    if separable:
      layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
    if norm_layer is not None:
      layers.append(norm_layer(out_ch))
    if activation_layer is not None:
      layers.append(activation_layer())
    super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
  """A PyTorch sequential block containing a transposed convolutional
  layer with default normalization and activation.

  Parameters
  ----------
  in_ch : int
    Number of input channels.
  out_ch : int
    Number of output channels.
  kernel_size : int or Tuple[int, int]
    Size of the convolutional kernel.
  fstride : int, default=1
    Stride in the frequency (width) dimension of the input.
  dilation : int, default=1
    Dilation rate for the kernel.
  fpad : bool, default=True
    Whether to apply padding to the frequency (width) dimension of
    the input.
  bias : bool, default=True
    Whether to include a bias term in the convolutional layer.
  separable : bool, default=False
    Whether to use a separable convolution.
  norm_layer : callable, default=torch.nn.BatchNorm2d
    A callable that returns a normalization layer to apply after
    the convolution.
  activation_layer : callable, default=torch.nn.ReLU
    A callable that returns an activation layer to
    apply after normalization.
  trans_conv_type : str, default="conv_transpose"
    The type of transposed convolution to use. Options are
    "conv_transpose" and "up_sample".

  """

  def __init__(
      self,
      in_ch: int,
      out_ch: int,
      kernel_size: Union[int, Tuple[int, int]],
      fstride: int = 1,
      dilation: int = 1,
      fpad: bool = True,
      bias: bool = True,
      separable: bool = False,
      norm_layer: Optional[Callable[..., torch.nn.Module]
                           ] = torch.nn.BatchNorm2d,
      activation_layer: Optional[Callable[...,
                                          torch.nn.Module]] = torch.nn.ReLU,
      trans_conv_type: str = "conv_transpose",
  ):
    self.in_ch = in_ch
    self.out_ch = out_ch
    self.kernel_size = kernel_size
    self.fstride = fstride
    self.dilation = dilation
    self.bias = bias
    self.trans_conv_type = trans_conv_type

    # Padding on time axis, with lookahead = 0
    lookahead = 0  # This needs to be handled on the input feature side
    kernel_size = (kernel_size, kernel_size) if isinstance(
        kernel_size, int) else kernel_size
    if fpad:
      fpad_ = kernel_size[1] // 2
    else:
      fpad_ = 0
    self.fpad = fpad_

    pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
    self.layers = []
    if any(x > 0 for x in pad):
      self.layers.append(nn.ConstantPad2d(pad, 0.0))
    groups = math.gcd(in_ch, out_ch) if separable else 1
    self.groups = groups
    if groups == 1:
      separable = False
    if trans_conv_type == "conv_transpose":
      self.layers.append(
          nn.ConvTranspose2d(
              in_ch,
              out_ch,
              kernel_size=kernel_size,
              padding=(kernel_size[0] - 1, fpad_ + dilation - 1),
              output_padding=(0, fpad_),
              stride=(1, fstride),  # Stride over time is always 1
              dilation=(1, dilation),
              groups=groups,
              bias=bias,
          )
      )
    else:
      self.layers.append(
          SeparatedTransposedConv2d(
              self.in_ch,
              self.out_ch,
              kernel_size=self.kernel_size,
              padding=(self.kernel_size[0] - 1,
                       self.fpad + self.dilation - 1),
              output_padding=(0, self.fpad),
              stride=(1, self.fstride),  # Stride over time is always 1
              groups=self.groups,
              bias=self.bias,
              consistent=False,
          ))

    if separable:
      self.layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
    if norm_layer is not None:
      self.layers.append(norm_layer(out_ch))
    if activation_layer is not None:
      self.layers.append(activation_layer())
    super().__init__(*self.layers)


class GroupedLinearEinsum(nn.Module):
  """Applies a linear transformation to the input tensor using
  grouped weights.

  Parameters
  ----------
  input_size : int
    The number of expected features in the input.
  hidden_size : int
    The number of output features.
  groups : int, default=1
    Number of groups to divide the weights and input tensor into.

  """

  def __init__(self, input_size: int, hidden_size: int, groups: int = 1):

    super().__init__()
    # self.weight: Tensor
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.groups = groups
    assert input_size % groups == 0, f"Input size {input_size} not divisible by {groups}"
    assert hidden_size % groups == 0, f"Hidden size {hidden_size} not divisible by {groups}"
    self.ws = input_size // groups
    self.register_parameter(
        "weight",
        Parameter(
            torch.zeros(groups, input_size // groups, hidden_size // groups),
            requires_grad=True),
    )
    self.reset_parameters()

  def reset_parameters(self):
    """Resets the weights."""
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore

  def forward(self, x: Tensor) -> Tensor:
    """Applies the grouped linear einsum transformation to the input
    tensor.

    Parameters
    ----------
    x : torch.Tensor
      The input tensor of shape [B, T, input_size].

    Returns
    -------
    torch.Tensor
      The output tensor of shape [B, T, hidden_size].

    """
    # x: [..., I]
    b, t, _ = x.shape
    # new_shape = list(x.shape)[:-1] + [self.groups, self.ws]
    new_shape = (b, t, self.groups, self.ws)
    x = x.view(new_shape)
    # The better way, but not supported by torchscript
    # x = x.unflatten(-1, (self.groups, self.ws))  # [..., G, I/G]
    x = torch.einsum("btgi,gih->btgh", x, self.weight)  # [..., G, H/G]
    x = x.flatten(2, 3)  # [B, T, H]
    return x


class SqueezedRNN_S(nn.Module):

  """A PyTorch module that implements a squeezed GRU, which is a variant
  of GRU with a smaller number of parameters.

  Parameters
  ----------
  input_size : int
    The number of expected features in the input tensor.
  hidden_size : int
    The number of features in the hidden state tensor.
  output_size : int, default=None
    The number of expected features in the output tensor.
    If None, the identity function is used as the output layer
  linear_groups : int, default=8
    The number of groups to use for the grouped linear layers.
  batch_first : bool, default=True
    If True, then the input and output tensors are provided
    as (batch, seq, feature).
  gru_skip_op : Callable[..., torch.nn.Module], default=None
    A callable function to apply as a skip connection.
    The default value of None means that no skip connection is used.
  linear_act_layer : Callable[..., torch.nn.Module], default=nn.Identity
    A callable function to use as an activation function for the linear
    layers.


  Methods
  -------
  forward(input, h=None)
    Perform the forward pass of the SqueezedGRU module.

  """

  def __init__(
      self,
      rnn_type: str,
      input_size: int,
      hidden_size: int,
      output_size: Optional[int] = None,
      num_layers: int = 1,
      linear_groups: int = 8,
      batch_first: bool = True,
      rnn_skip_op: Optional[Callable[..., torch.nn.Module]] = None,
      linear_act_layer: Callable[..., torch.nn.Module] = nn.Identity,
  ):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.linear_in = nn.Sequential(
        GroupedLinearEinsum(input_size, hidden_size,
                            linear_groups), linear_act_layer()
    )
    if rnn_type == "LiGRU":
      self.rnn = OptimizedLightGRU(hidden_size, hidden_size,
                                   num_layers=num_layers,
                                   batch_first=batch_first)
    else:
      self.rnn = getattr(nn, rnn_type.upper())(hidden_size, hidden_size,
                                               num_layers=num_layers,
                                               batch_first=batch_first)

    self.rnn_skip = rnn_skip_op() if rnn_skip_op is not None else None
    if output_size is not None:
      self.linear_out = nn.Sequential(
          GroupedLinearEinsum(hidden_size, output_size,
                              linear_groups), linear_act_layer()
      )
    else:
      self.linear_out = nn.Identity()

  def forward(self, input: Tensor, h=None) -> Tuple[Tensor, Tensor]:
    x = self.linear_in(input)
    x, h = self.rnn(x, h)
    x = self.linear_out(x)
    if self.rnn_skip is not None:
      x = x + self.rnn_skip(input)
    return x, h


class BaseModel(nn.Module):
  """Abstract class for all models.

  Methods
  -------
  from_pretrained(path)
    Load a trained model from a given path.
  from_checkpoint(path)
    Load a trained model from a checkpoint given path.
  """

  def __init__(self):
    super().__init__()
    self.name = "base_model"

  @classmethod
  def from_pretrained(cls, path, *args):
    """Load a trained model from a given path.

    Parameters
    ----------
    path : str
      The path to the saved model checkpoint.

    Returns
    -------
    Model
      A new instance of the `Model` class with the same architecture
      and parameters as the trained model checkpoint.
    """
    if os.path.isdir(path):
      path = os.path.join(path, "best_model.pth")
    state = torch.load(path, map_location="cpu")
    state_dict = state["state_dict"]
    key_to_remove = cls.key_to_remove()
    for key in list(state["config"].keys()):
      if key in key_to_remove:
        state["config"].pop(key)
    model = cls(*args, **state["config"])
    model.load_state_dict(state_dict)
    return model

  @classmethod
  def from_checkpoint(cls, path, *args):
    """Load a trained model from a given path.

    Parameters
    ----------
    path : str
      The path to the saved model checkpoint.

    Returns
    -------
    Model
      A new instance of the `Model` class with the same architecture
      and parameters as the trained model checkpoint.

    Note
    ----
    The difference between this method and `from_pretrained` is that
    this method loads the model from a checkpoint saved by the
    `Trainer` class that also contains the state of the optimizer,
    callbacks, etc..., while `from_pretrained` loads the model from an
    isolated version of this checkpoint that only contains the model's
    state dict and config.
    """
    state = torch.load(path, map_location="cpu")
    state_dict = state["state_dict"]
    key_to_remove = cls.key_to_remove()
    for key in list(state["training_config"]["model"].keys()):
      if key in key_to_remove:
        state["training_config"]["model"].pop(key)
    model = cls(*args, **state["training_config"]["model"])
    new_state = {}
    for key in state_dict.keys():
      new_state[key.replace("model.", "")] = state_dict[key]
    model.load_state_dict(new_state)
    return model

  @classmethod
  def key_to_remove(cls):
    """Get the key to remove from the state dict when loading a model"""
    return []


class GainEncoder(nn.Module):
  """Encoder module for DPNMM. This module processes the input Bark features.

  Parameters
  ----------
  input_ch : int, default=3
    Number of input channels (default is 3).
  conv_ch : int, default=64
    Number of channels used in the convolutional layers (default is 64).
  conv_kernel_inp : tuple[int], default=(3, 3)
    Kernel size for the initial convolutional layer that processes the
    input waveform (default is (3, 3)).
  conv_kernel : tuple[int], default=(1, 3)
    Kernel size for the convolutional layers that follow the initial
    layer, (default is (1, 3)).
  nb_bark : int, default=26
    Number of bark bands (default is 26).
  emb_hidden_dim : int, default=256
    Number of units in each layer of the encoder's GRU (default is 256).
  emb_num_layers : int, default=1
    Number of layers in the encoder's GRU (default is 1).
  lin_groups : int, default=32
    Number of groups to use in the grouped linear layers (default is 32).
  enc_lin_groups : int, default=32
    Number of groups to use in the encoder's grouped linear layers (default is 32).
  rnn_type : str, default="gru"
    Type of RNN to use in the encoder (default is "gru").

  Attributes
  ----------
  conv0 : Conv2dNormAct
    Initial convolutional layer.
  conv1, conv2, conv3 : Conv2dNormAct
    Convolutional layers used to compress the waveform representation.
  bark_bins : int
    Number of bark bands.
  emb_in_dim, emb_out_dim, emb_dim : int
    Dimensions of the encoder's GRU.
  lin0 : GroupedLinearEinsum
    Grouped linear layer before the GRU.
  emb_gru : SqueezedRNN_S
    GRU used to compress the waveform representation.

  Methods
  -------
  forward(feat_bark)
    Applies the encoder to the input Bark features to obtain a compressed
    representation.
  """

  def __init__(
      self,
      input_ch: int = 3,
      conv_ch: int = 64,
      conv_kernel_inp: (int) = (3, 3),
      conv_kernel: (int) = (1, 3),
      nb_bark: int = 26,
      emb_hidden_dim: int = 256,
      emb_num_layers: int = 1,
      lin_groups: int = 32,    # 96 dans le code initial
      enc_lin_groups: int = 32,
      rnn_type="gru"
  ):
    super().__init__()
    self.conv0 = Conv2dNormAct(
        input_ch,
        conv_ch,
        kernel_size=conv_kernel_inp,
        fpad=False,
        bias=False,
        separable=True)
    conv_layer = partial(
        Conv2dNormAct,
        in_ch=conv_ch,
        out_ch=conv_ch,
        bias=False,
        separable=True,
    )

    self.conv1 = conv_layer(kernel_size=conv_kernel, fstride=2)
    self.conv2 = conv_layer(kernel_size=conv_kernel, fstride=2)
    self.conv3 = conv_layer(kernel_size=conv_kernel, fstride=1)

    self.bark_bins = nb_bark
    self.emb_in_dim = conv_ch * (nb_bark // 4)
    self.emb_dim = emb_hidden_dim
    self.emb_out_dim = conv_ch * (nb_bark // 4)
    self.lin0 = GroupedLinearEinsum(
        conv_ch * (nb_bark // 4), self.emb_in_dim, groups=enc_lin_groups
    )
    self.emb_gru = SqueezedRNN_S(
        rnn_type,
        self.emb_in_dim,
        self.emb_dim,
        output_size=self.emb_out_dim,
        num_layers=emb_num_layers,
        batch_first=True,
        rnn_skip_op=None,
        linear_groups=lin_groups,
        linear_act_layer=partial(nn.ReLU, inplace=True),
    )

  def forward(
      self, feat_bark: Tensor
  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Forward pass of the Encoder module.

    Parameters
    ----------
    feat_bark : torch.Tensor
      The input tensor  with shape [batch_size, 3, nb_frames, nb_bark].

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor]
      A tuple of tensors containing the following:
      - e0: the output tensor of the first convolution layer.
      - e1: the output tensor of the second convolution layer.
      - e2: the output tensor of the third convolution layer.
      - e3: the output tensor of the fourth convolution layer.
      - emb: the output tensor of the GRU layer with shape [batch_size, nb_frames, -1]..
    """

    feat_bark = feat_bark.view(-1,
                               feat_bark.shape[-3],
                               feat_bark.shape[-2],
                               feat_bark.shape[-1])

    e0 = self.conv0(feat_bark)  # [B, C, N, F]
    e1 = self.conv1(e0)  # [B, C, N, F/2]
    e2 = self.conv2(e1)  # [B, C, N, F/4]
    e3 = self.conv3(e2)  # [B, C, N, F/4]
    emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, N, C * F/4]
    emb = self.lin0(emb)
    emb, _ = self.emb_gru(emb)  # [B, N, -1]
    return e0, e1, e2, e3, emb


class GainDecoder(nn.Module):
  """A neural network module that decodes the features extracted by the
  `GainEncoder` module.

  Parameters
  ----------
  conv_ch : int, default=64
    Number of channels in convolution layers.
  conv_kernel : tuple[int], default=(1, 3)
    Kernel size of the convolution layers.
  nb_bark : int, default=26
    Number of bark bands.
  emb_hidden_dim : int, default=256
    The number of features in the GRU's hidden state.
  emb_num_layers : int, default=2
    The number of layers in the GRU.
  lin_groups : int, default=32
    Number of groups for the linear convolution layers.
  rnn_type : str, default="gru"
    Type of RNN to use in the decoder.
  trans_conv_type : str, default="conv_transpose"
    The type of transposed convolution to use. Options are
    "conv_transpose" and "up_sample".

  Attributes
  ----------
  emb_in_dim : int
    The number of input features to the GRU.
  emb_dim : int
    The number of features in the GRU's hidden state.
  emb_out_dim : int
    The number of features in the GRU's output.

  Methods
  -------
  forward(emb, e3, e2, e1, e0)
    Forward pass through the network.

  """

  def __init__(
      self,
      conv_ch: int = 64,
      conv_kernel: (int) = (1, 3),
      nb_bark: int = 26,
      emb_hidden_dim: int = 256,
      emb_num_layers: int = 2,
      lin_groups: int = 32,
      rnn_type="gru",
      trans_conv_type="conv_transpose"

  ):
    super().__init__()

    self.emb_in_dim = conv_ch * (nb_bark // 4)
    self.emb_dim = emb_hidden_dim
    self.emb_out_dim = conv_ch * (nb_bark // 4)

    self.emb_gru = SqueezedRNN_S(
        rnn_type,
        self.emb_in_dim,
        self.emb_dim,
        output_size=self.emb_out_dim,
        num_layers=emb_num_layers,
        batch_first=True,
        rnn_skip_op=None,
        linear_groups=lin_groups,
        linear_act_layer=partial(nn.ReLU, inplace=True),
    )

    tconv_layer = partial(
        ConvTranspose2dNormAct,
        kernel_size=conv_kernel,
        bias=False,
        separable=True,
        trans_conv_type=trans_conv_type,

    )
    conv_layer = partial(
        Conv2dNormAct,
        bias=False,
        separable=True,
    )
    # convt: TransposedConvolution, convp: Pathway (encoder to decoder)
    # convolutions
    self.conv3p = conv_layer(conv_ch, conv_ch, kernel_size=1)
    self.convt3 = conv_layer(conv_ch, conv_ch, kernel_size=conv_kernel)
    self.conv2p = conv_layer(conv_ch, conv_ch, kernel_size=1)
    self.convt2 = tconv_layer(conv_ch, conv_ch, fstride=2)
    self.conv1p = conv_layer(conv_ch, conv_ch, kernel_size=1)
    self.convt1 = tconv_layer(conv_ch, conv_ch, fstride=2)
    self.conv0p = conv_layer(conv_ch, conv_ch, kernel_size=1)
    # Last convolution is also a transposed convolution to bring back the
    # number of features at 26
    self.conv0_out = tconv_layer(
        conv_ch, 1, kernel_size=conv_kernel, activation_layer=None, fpad=False
    )

  def forward(self, emb, e3, e2, e1, e0) -> Tensor:
    """Calculates the gains in dB per Bark band to apply to the input.

    Parameters
    ----------
    emb : torch.Tensor
      Input tensor with shape [B, T, C], where B is the batch size,
      T is the time steps, and C is the number of input channels.
    e3 : torch.Tensor
      Input tensor with shape [B, C, T, F/4].
    e2 : torch.Tensor
      Input tensor with shape [B, C, T, F/2].
    e1 : torch.Tensor
      Input tensor with shape [B, C, T, F].
    e0 : torch.Tensor
      Input tensor with shape [B, 1, T, F].

    Returns
    -------
    torch.Tensor
      The predicted gains in dB with shape [B, T, F].
    """

    b, _, t, f = e3.shape
    emb, _ = self.emb_gru(emb)
    emb = emb.view(b, t, f, -1).permute(0, 3, 1, 2)  # [B, C, N, F/8]
    e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C, N, F/4]
    e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C, N, F/2]
    e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, N, F]
    m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, N, F]

    return m


class DPNMM(BaseModel):
  """
  DPNMM Model for Perceptual Noise Masking.

  Encoder-Decoder model that predicts the gains in dB per Bark band to apply to
  the input music to raise its masking thresholds above the noise level.

  Input Bark features : 3 channels [B, 3, N, F]
  - music PSD
  - noise PSD
  - music masking thresholds

  Parameters
  ----------
  nfft : int, optional
    Number of FFT points, by default 2048.
  sr : int, optional
    Sampling rate, by default 44100.
  filter_order : int, optional
    Order of the target filter, by default 80.
  envelope_order : int, optional
    Order of the spectral envelope, by default 80.
  nb_bark : int, optional
    Number of Bark bands, by default 26.
  input_ch : int, optional
    Number of input channels, by default 3.
  conv_ch : int, optional
    Number of convolutional channels, by default 64.
  conv_kernel_inp : tuple of int, optional
    Kernel size for input convolution, by default (3, 3).
  conv_kernel : tuple of int, optional
    Kernel size for convolution, by default (1, 3).
  emb_hidden_dim : int, optional
    Hidden dimension for the embedding, by default 256.
  emb_num_layers : int, optional
    Number of GRU layers in the encoder, by default 1.
  lin_groups : int, optional
    Number of linear groups, by default 32.
  enc_lin_groups : int, optional
    Number of linear groups in the encoder, by default 32.
  rnn_type : str, optional
    Type of RNN to use, by default "gru".
  trans_conv_type : str, optional
    Type of transposed convolution to use, by default "conv_transpose".
  max_positive_clamping_value : float, optional
    Maximum positive clamping value, by default 10. / 3.
  min_negative_clamping_value : float, optional
    Minimum negative clamping value, by default -5. / 3.
  remove_high_bands : bool, optional
    Whether to remove high bands, by default True.

  Methods
  -------
  forward(input)
    Forward pass through the model.

  """

  def __init__(
      self,
      nfft: int = 2048,
      sr: int = 44100,
      filter_order: int = 80,
      envelope_order: int = 80,
      nb_bark: int = 26,
      input_ch: int = 3,
      conv_ch: int = 64,
      conv_kernel_inp: (int) = (3, 3),
      conv_kernel: (int) = (1, 3),
      emb_hidden_dim: int = 256,
      emb_num_layers: int = 1,   # nb of GRU layers in encoder
      lin_groups: int = 32,
      enc_lin_groups: int = 32,
      rnn_type="gru",
      trans_conv_type="conv_transpose",
      max_positive_clamping_value=10. / 3.,
      min_negative_clamping_value=-5. / 3.,
      remove_high_bands=True
  ):

    super().__init__()
    self.name = "DPNMM"

    # Initialization of ddsp models
    self.nfft = nfft
    self.sr = sr
    self.filter_order = filter_order
    self.envelope_order = envelope_order

    self.spectral_envelope = SpectralEnvelope(
        nfft=nfft, sr=sr, order=envelope_order)
    self.target_filter = TargetFilter(
        nfft=nfft, sr=sr, filter_order=filter_order)
    self.masking_thresholds = MaskingThresholds(
        nfft=nfft, sr=sr)

    # Initialization of the DNN

    self.nb_bark = nb_bark
    self.input_ch = input_ch
    self.conv_ch = conv_ch
    self.conv_kernel_inp = conv_kernel_inp
    self.conv_kernel = conv_kernel
    self.emb_hidden_dim = emb_hidden_dim
    self.emb_num_layers = emb_num_layers
    self.lin_groups = lin_groups
    self.enc_lin_groups = enc_lin_groups

    self.perceptual_clamping = perceptual_clamping
    self.max_positive_clamping_value = max_positive_clamping_value
    self.min_negative_clamping_value = min_negative_clamping_value
    self.remove_high_bands = remove_high_bands

    # Output of the network is clamped to different values depending on the
    # config
    max_clamp_val, min_clamp_val = custom_clamping_values(
        max_positive_clamping_value=self.max_positive_clamping_value,
        min_negative_clamping_value=self.min_negative_clamping_value,
        remove_high_bands=self.remove_high_bands,
        n_bark=self.nb_bark
    )

    self.register_buffer(
        "min_clamp_val",
        min_clamp_val
    )

    self.register_buffer(
        "max_clamp_val",
        max_clamp_val
    )

    self.encoder = GainEncoder(
        input_ch=input_ch,
        conv_ch=conv_ch,
        conv_kernel_inp=conv_kernel_inp,
        conv_kernel=conv_kernel,
        nb_bark=nb_bark,
        emb_hidden_dim=emb_hidden_dim,
        emb_num_layers=emb_num_layers,
        lin_groups=lin_groups,
        enc_lin_groups=lin_groups,
        rnn_type=rnn_type
    )

    self.decoder = GainDecoder(
        conv_ch=conv_ch,
        conv_kernel=conv_kernel,
        nb_bark=nb_bark,
        emb_hidden_dim=emb_hidden_dim,
        emb_num_layers=emb_num_layers,
        lin_groups=lin_groups,
        rnn_type=rnn_type,
        trans_conv_type=trans_conv_type
    )

  def forward(
      self,
      input,
  ):

    # input [batch_size, 3, n_frames, n_bark]
    # Music PSD, Noise PSD, Music masking thresholds, all on 26 bark bands

    e0, e1, e2, e3, emb = self.encoder(input)
    gains = self.decoder(emb, e3, e2, e1, e0)
    gains = gains.squeeze(dim=1)

    # Clamping the output
    gains = custom_clamp(
        gains,
        max_val=self.max_clamp_val,
        min_val=self.min_clamp_val
    )

    return gains


def main():
  Encoder = GainEncoder()
  input = torch.zeros((1, 3, 100, 26))
  e0, e1, e2, e3, emb = Encoder.forward(input)

  print("input:", input.shape)
  print("conv0:", e0.shape)
  print("conv1:", e1.shape)
  print("conv2:", e2.shape)
  print("conv3:", e3.shape)
  print("after linear/gru:", emb.shape)

  Decoder = GainDecoder()
  output = Decoder.forward(emb, e3, e2, e1, e0)
  print("output decoder:", output.shape)

  world_size = torch.cuda.device_count()
  print('Nb GPUs: ', world_size)
  model = DPNMM()
  model = model.to('cuda')
  print(model.masking_thresholds.spreadmatrix.device)


if __name__ == "__main__":
  main()

import os
import string
import random
import numpy as np

slurm = "SLURM_JOB_ID" in os.environ

if slurm:
  JOB_ID = os.environ["SLURM_JOB_ID"]
else:
  JOB_ID = "".join(random.choices(
      string.ascii_letters + string.digits, k=8))


root_dir = os.environ['DATA']
music_noise_dir = os.path.join(root_dir, "music_noise")
csv_file = os.path.join(music_noise_dir, "metadata.csv")


stft_parameters = {
    "nfft": 2048,
    "sr": 44100,
    "nb_bark": 26,
}

common_parameters = {
    "exp_dir": os.path.join(
        os.environ['REPO'],
        "trained_models"
    ),
    'audio': {
        'nfft': stft_parameters['nfft'],
        'sr': stft_parameters['sr'],
        'filter_order': 80,
        'envelope_order': 80,
    },
    'dataset': {
        'root_dir': os.environ['MY_DATA'],
        'csv_file': csv_file,
        #        'split': 0.8
    },
    'optim': {
        "lr": 1e-3,
        "betas": (
            0.9,
            0.999),
        "weight_decay": 0.0001,
        'batch_size': 64,
        "epochs": 50,
        "patience": 30,
        "lr_scheduler": None,  # "OneCycleLR",
        "lr_scheduler_args": {"max_lr": 1e-3, "total_steps": None,
                              "div_factor": 10, "pct_start": 0.15},
    },
    "process": {
        "num_workers": 4,
        "prefetch": 2,
        "devices": int(
            os.environ["SLURM_GPUS_ON_NODE"]) if "SLURM_GPUS_ON_NODE" in os.environ else 1,
        "num_nodes": int(
            os.environ["SLURM_NNODES"]) if slurm else 1,
    },
    "job_id": JOB_ID,
    "seed": 35,
    "loss": {
        "penalty": None,
    },
    "gains": {
        "remove_non_activated_bands": True,
        "spreadwidth": 3,
    },
}

conf = {
    "nopower": {
        "checkpoint_path": None,
        # "optim": {
        #     "lr_scheduler": "OneCycleLR",
        #     "lr_scheduler_args": {"max_lr": 1e-3, "total_steps": None,
        #                           "div_factor": 10, "pct_start": 0.15},
        # },
        "model": {
            "nfft": stft_parameters['nfft'],
            "nb_bark": stft_parameters['nb_bark'],
            "input_ch": 3,
            "conv_ch": 32,
            "conv_kernel_inp": (3, 3),
            "conv_kernel": (1, 3),
            "emb_hidden_dim": 256,
            "emb_num_layers": 1,
            "lin_groups": 32,
            "enc_lin_groups": 32,
            "rnn_type": "gru",
            "trans_conv_type": "conv_transpose",
            "gain_clamping": {
                "max_positive_clamping_value": 10. / 3.,
                "remove_high_bands": True,
                "min_negative_clamping_value": -5. / 3.,
            },
        },
        "loss": {
            "nfft": stft_parameters['nfft'],
            "sr": stft_parameters['sr'],
            "nb_bark": stft_parameters['nb_bark'],
            "thresholds_margins": 0.,
            "penalty": None,
            "penalty_thr": None,
            "loss_type": "relu",  # or "abs"
            "epsilon": 1e-12,
            "mdmm": False,
            "mean_power_constraint": False,
        },
        "gains": {
            "remove_non_activated_bands": True,
            "spreadwidth": 3,
        },
        "mdmm": {
            "multiplier_lr": 0.001,
            "constraint_thr": [2.],
            "damping_coeff": [0.],
            "initial_value": 0.,
        }
    },
    "power_2": {
        "checkpoint_path": None,
        # "optim": {
        #     "lr_scheduler": "OneCycleLR",
        #     "lr_scheduler_args": {"max_lr": 1e-3, "total_steps": None,
        #                           "div_factor": 10, "pct_start": 0.15},
        # },
        "model": {
            "nfft": stft_parameters['nfft'],
            "nb_bark": stft_parameters['nb_bark'],
            "input_ch": 3,
            "conv_ch": 32,
            "conv_kernel_inp": (3, 3),
            "conv_kernel": (1, 3),
            "emb_hidden_dim": 256,
            "emb_num_layers": 1,
            "lin_groups": 32,
            "enc_lin_groups": 32,
            "rnn_type": "gru",
            "trans_conv_type": "conv_transpose",
            "gain_clamping": {
                "max_positive_clamping_value": 10. / 3.,
                "remove_high_bands": True,
                "min_negative_clamping_value": -5. / 3.,
            },
        },
        "loss": {
            "nfft": stft_parameters['nfft'],
            "sr": stft_parameters['sr'],
            "nb_bark": stft_parameters['nb_bark'],
            "thresholds_margins": 0.,
            "loss_type": "relu",
            "epsilon": 1e-12,
            "mdmm": True,
            "mean_power_constraint": True,
        },
        "gains": {
            "remove_non_activated_bands": True,
            "spreadwidth": 3,
        },
        "mdmm": {
            "multiplier_lr": 0.001,
            "constraint_thr": [2.],
            "damping_coeff": [0.],
            "initial_value": 0.,
        }
    },
    "power_1": {
        "checkpoint_path": None,
        # "optim": {
        #     "lr_scheduler": "OneCycleLR",
        #     "lr_scheduler_args": {"max_lr": 1e-3, "total_steps": None,
        #                           "div_factor": 10, "pct_start": 0.15},
        # },
        "model": {
            "nfft": stft_parameters['nfft'],
            "nb_bark": stft_parameters['nb_bark'],
            "input_ch": 3,
            "conv_ch": 32,
            "conv_kernel_inp": (3, 3),
            "conv_kernel": (1, 3),
            "emb_hidden_dim": 256,
            "emb_num_layers": 1,
            "lin_groups": 32,
            "enc_lin_groups": 32,
            "rnn_type": "gru",
            "trans_conv_type": "conv_transpose",
            "gain_clamping": {
                "max_positive_clamping_value": 10. / 3.,
                "remove_high_bands": True,
                "min_negative_clamping_value": -5. / 3.,
            },
        },
        "loss": {
            "nfft": stft_parameters['nfft'],
            "sr": stft_parameters['sr'],
            "nb_bark": stft_parameters['nb_bark'],
            "thresholds_margins": 0.,
            "loss_type": "relu",
            "epsilon": 1e-12,
            "mdmm": True,
            "mean_power_constraint": True,
        },
        "gains": {
            "remove_non_activated_bands": True,
            "spreadwidth": 3,
        },
        "mdmm": {
            "multiplier_lr": 0.001,
            "constraint_thr": [1.],
            "damping_coeff": [0.],
            "initial_value": 0.,
        }
    },
    "power_0-5": {
        "checkpoint_path": None,
        # "optim": {
        #     "lr_scheduler": "OneCycleLR",
        #     "lr_scheduler_args": {"max_lr": 1e-3, "total_steps": None,
        #                           "div_factor": 10, "pct_start": 0.15},
        # },
        "model": {
            "nfft": stft_parameters['nfft'],
            "nb_bark": stft_parameters['nb_bark'],
            "input_ch": 3,
            "conv_ch": 32,
            "conv_kernel_inp": (3, 3),
            "conv_kernel": (1, 3),
            "emb_hidden_dim": 256,
            "emb_num_layers": 1,
            "lin_groups": 32,
            "enc_lin_groups": 32,
            "rnn_type": "gru",
            "trans_conv_type": "conv_transpose",
            "gain_clamping": {
                "max_positive_clamping_value": 10. / 3.,
                "remove_high_bands": True,
                "min_negative_clamping_value": -5. / 3.,
            },
        },
        "loss": {
            "nfft": stft_parameters['nfft'],
            "sr": stft_parameters['sr'],
            "nb_bark": stft_parameters['nb_bark'],
            "thresholds_margins": 0.,
            "loss_type": "relu",
            "epsilon": 1e-12,
            "mdmm": True,
            "mean_power_constraint": True,
        },
        "gains": {
            "remove_non_activated_bands": True,
            "spreadwidth": 3,
        },
        "mdmm": {
            "multiplier_lr": 0.001,
            "constraint_thr": [.5],
            "damping_coeff": [0.],
            "initial_value": 0.,
        }
    },
}

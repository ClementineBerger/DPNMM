# Deep Perceptual Noise Masking with Music Envelope Shaping (DPNMM)

Welcome to the **Deep Perceptual Noise Masking with Music Envelope Shaping (DPNMM)** article repository! 

This repository contains the companion code for the ICASSP 2025 paper.

## Abstract

People often listen to music in noisy environments, seeking to isolate themselves from ambient sounds. Indeed, a music signal can mask some of the noise's frequency components due to the effect of simultaneous masking. In this article, we propose a neural network based on a psychoacoustic masking model, designed to enhance the music's ability to mask ambient noise by reshaping its spectral envelope with predicted filter frequency responses. The model is trained with a perceptual loss function that balances two constraints: effectively masking the noise while preserving the original music mix and the user's chosen listening level. We evaluate our approach on simulated data replicating a user's experience of listening to music with headphones in a noisy environment. The results, based on defined objective metrics, demonstrate that our system improves the state of the art.

## Links

[:loud_sound: Audio examples](https://clementineberger.github.io/DPNMM/audio)

[:mag: Evaluation results](https://clementineberger.github.io/DPNMM/results)

[:page_facing_up:]() [Paper]() 

## Setup

### Environment variables

This repository requires environment variables that need to be added to your `.bashrc`:

```
# Path to the repository
export REPO="path/to/the/repository"

# Path to the dataset
export DATA="path/to/the/data"
```

**Note:** The dataset is not publicly available. If you wish to generate your own dataset, either following the method described in the article or using your own approach, you will need to create a `metadata.csv` file. This file must include one entry per pair of music and noise files, with at least two columns: `music_path` and `noise_path`, specifying the file paths to the corresponding audio files. Additionaly you can add a column specifying train/val/test set for training and evaluation. Ensure that the `metadata.csv` file, along with the audio files, is stored in a `music_noise` directory. This directory should be located at the path specified by the DATA environment variable.

### Requirements

Python 3.11 is recommended and the following packages are required:

```
pyyaml==6.0.1
numpy==1.26.3
librosa==0.10.1
matplotlib==3.8.0   
scipy==1.11.3
torch==2.1.2
torchaudio==2.1.2
lightning==2.2.1
tensorboard==2.16.2
pandas==2.1.4
```


## Code 

- `config.py` : training configurations.
- `custom_dataloader.py` : dataloader for the music_noise dataset.
- `losses.py` : loss functions.
- `model.py` : neural model for prediction of the gains per critical band.
- `system.py` : lighting system script.
- `train.py` : training script.
- `model_wrapper.py` : model wrapper for inference.
- `eval_model.py` : evaluation script.
- `eval_dataloader.py` : specific dataloader for evaluation.
- `example.py` : inference example.

- The `utils` folder contains useful scripts for the model and evaluation :
    - `analysis.py` : code for stft, isftf, frequency perceptual weighting.
    - `ddsp.py` : code for ddsp (bark scale, etc.)
    - `multiloss_framework.py` : implementation of the Modified Differential Multipliers method,
    - `masking_thresholds.py` : code for Johnston's masking thresholds computation,
    - `objective_metrics.py` : objective metrics for evaluation,
    - `post_process.py` : post-process utils (gains smoothing, removing non-activated bands),
    - `spectral_shaping.py` : everything related to spectral shaping (pattern for creating the filters etc.).

## Training

1. define a new configuration or use one of the predefined ones in `config.py`,
2. Train using the training script:
```
python train.py --conf_id=your_conf_name
```

## Inference

You can find an example for inference in `example.py`. You will need to specify the paths to music and noise audio signals within the script and launch it precising the `conf_id` and `job_id` of the trained model you want to use :
```
python example.py --conf_id=conf_name_of_your_model --job_id=job_name
```

## Evaluation
You can evaluate the model on a test set using the `eval_model.py` script.

```
python eval_model.py --conf_id=conf_name --job_id=job_name --remove_bands=True
```

The `remove_bands` argument indicate whether or not you want to put to zero the gains predicted in the "unactivated bands" as described in the article. 


The results are saved in `.yaml` file containing a dictionary. This file is generated in the same location as the trained model. 
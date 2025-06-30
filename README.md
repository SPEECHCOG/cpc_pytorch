# A PyTorch implementation of Contrastive Predictive Coding (CPC)

This repository contains a pipeline for training and evaluating models trained using contrastive predictive coding (CPC, see e.g. https://arxiv.org/abs/1807.03748). The code has been implemented using PyTorch.

**The present CPC implementation has been (partially) used in the following publication:**
[E. Vaaras, M. Airaksinen, and O. Räsänen, "Analysis of Self-Supervised Learning and Dimensionality Reduction Methods in Clustering-Based Active Learning for Speech Emotion Recognition," in _Proc. INTERSPEECH 2022_](https://www.isca-speech.org/archive/interspeech_2022/vaaras22_interspeech.html).

If you use the present code or its derivatives, please cite the [repository URL](https://github.com/SPEECHCOG/cpc_pytorch) and/or the [aforementioned publication](https://www.isca-speech.org/archive/interspeech_2022/vaaras22_interspeech.html).

## Repository contents
- `conf_train_cpc_model.py` and `conf_train_cpc_model_orig_implementation.py`: Example configuration files for training baseline models.
- `cpc_data_loader.py`: A file containing two example data loaders, one for raw audio and another one that converts raw audio into log-mel features.
- `cpc_loss.py`: A file containing two different variations of the InfoNCE loss used in CPC models.
- `cpc_model.py`: A file containing the actual CPC model implementation, i.e. the CPC encoder, autoregressive model, and post-net.
- `cpc_test_bench.py`: A superscript that can be used to train and evaluate different variations of CPC models and their hyperparameters at once.
- `py_conf_file_into_text.py`: A script for converting _.py_ configuration files into lists of text that can be used for printing or writing the configuration file contents into a text file.
- `train_cpc_model.py`: A script for training and evaluating a single CPC model.

**Note:** By default, the data loaders in _cpc_data_loader.py_ assume that you have a directory named _wav_files_ containing WAV files in the same directory as you have the script _cpc_data_loader.py_. Of course, you can change this option in the configuration file, e.g. with
```
params_train_dataset = {'file_dir': <your_directory>}
```
where _<your_directory>_ is the directory containing WAV files.

## Examples of how to use the code


### How to train a single CPC model with one hyperparameter configuration:
You can either use the command
```
python train_cpc_model.py
```
or
```
python train_cpc_model.py <configuration_file>
```
in order to train a single CPC model with one hyperparameter configuration. Using the former of these options requires having a configuration file named _conf_train_cpc_model.py_ in the same directory as the file _train_cpc_model.py_. In the latter option, _<configuration_file>_ is a _.py_ configuration file containing the hyperparameters you want to use.

### How to train multiple CPC models with different hyperparameter configurations:
In order to train multiple CPC models with different hyperparameter configurations, you can use the command
```
python cpc_test_bench.py <dir_to_configuration_files>
```
where _<dir_to_configuration_files>_ is a directory containing at least one _.py_ configuration file. This script runs through the CPC model training (_train_cpc_model.py_) with each of the configuration files located in _<dir_to_configuration_files>_, one at a time.

### How to train the CPC model from the original paper:
Using the command
```
python train_cpc_model.py conf_train_cpc_model_orig_implementation.py
```
you can train the CPC model from the [original CPC paper](https://arxiv.org/abs/1807.03748) with your own WAV files.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The configuration file for train_cpc_model.py using the original CPC implementation.

"""

# The hyperparameters for our training and testing process
max_epochs = 10000
patience = 100
dropout = 0.0
batch_size = 8
learning_rate = 1e-4

# The number of frames in the encoded samples z_t
num_frames_encoding = 128

# The future predicted timesteps. Giving a number n will result in timesteps [0, 1, ..., n - 1], or
# giving a list of numbers will result in discrete timesteps defined by the list. Giving one number
# inside a list (e.g. [12]) will make the model predict only one future timestep.
# NOTE: The first future timestep is 1, not 0
future_predicted_timesteps = 12

# Flags for training and testing a CPC model
train_model = 1
test_model = 1

# Flag for loading the weights for our model, i.e. flag for continuing a previous training process
load_model = 0

# Flag for saving the best model (according to validation loss) after each training epoch where the
# validation loss is lower than before
save_best_model = 1

# The name of the text file into which we log the output of the training process
name_of_log_textfile = 'cpc_trainlog_orig_implementation.txt'

# A flag for determining whether we want to print the contents of the configuration file into the
# logging file
print_conf_contents = 1

# Define our models that we want to use from the file cpc_model.py
encoder_name = 'CPC_encoder'
autoregressive_model_name = 'CPC_autoregressive_model'
postnet_name = 'CPC_postnet'

# A flag for defining if we are using RNNs in our AR model (i.e., do we need a hidden vector)
rnn_models_used_in_ar_model = 1

# Define our dataset for our data loader that we want to use from the file cpc_data_loader.py
dataset_name = 'CPC_raw_audio_dataset'

# Define our loss function that we want to use from the file cpc_loss.py
loss_name = 'CPC_loss_no_classes'

# The hyperparameters for the loss function
loss_params = {'future_predicted_timesteps': future_predicted_timesteps}

# Define the optimization algorithm we want to use from torch.optim
optimization_algorithm = 'Adam'

# The hyperparameters for our optimization algorithm
optimization_algorithm_params = {'lr': learning_rate}

# A flag to determine if we want to use a learning rate scheduler
use_lr_scheduler = 1

# Define which learning rate scheduler we want to use from torch.optim.lr_scheduler
lr_scheduler = 'ReduceLROnPlateau'

# The hyperparameters for the learning rate scheduler
lr_scheduler_params = {'mode': 'min',
                       'factor': 0.5,
                       'patience': 30}

# The names of the best models (according to validation loss) for loading/saving model weights
encoder_best_model_name = 'CPC_Encoder_best_model.pt'
ar_best_model_name = 'CPC_AR_best_model.pt'
w_best_model_name = 'W_best_model.pt'

# The hyperparameters for constructing the models. An empty dictionary will make the model to use
# only default hyperparameters, i.e., the hyperparameters of the original CPC paper
encoder_params = {'dropout': dropout}
ar_model_params = {}
w_params = {}

# The hyperparameters for our data loaders
params_train_dataset = {}
params_validation_dataset = {}
params_test_dataset = {}

# The hyperparameters for training and validation (arguments for torch.utils.data.DataLoader object)
params_train = {'batch_size': batch_size,
                'shuffle': True,
                'drop_last': True}

# The hyperparameters for testing (arguments for torch.utils.data.DataLoader object)
params_test = {'batch_size': batch_size,
               'shuffle': False,
               'drop_last': True}

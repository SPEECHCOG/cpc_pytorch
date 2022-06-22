# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for training and evaluating a CPC model.

"""

import numpy as np
import time
import sys

from importlib.machinery import SourceFileLoader
from copy import deepcopy
from torch import cuda, no_grad, save, load
from torch.utils.data import DataLoader

from py_conf_file_into_text import convert_py_conf_file_to_text


# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('Usage: \n1) python train_cpc_model.py \nOR \n2) python train_cpc_model.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
    conf_file_name = sys.argv[1]
else:
    try:
        import conf_train_cpc_model as conf
        conf_file_name = 'conf_train_cpc_model.py'
    except ModuleNotFoundError:
        sys.exit('''Usage: \n1) python train_cpc_model.py \nOR \n2) python train_cpc_model.py <configuration_file>\n\n
        By using the first option, you need to have a configuration file named "conf_train_cpc_model.py" in the same directory 
        as "train_cpc_model.py"''')


# Import our models
CPC_encoder = getattr(__import__('cpc_model', fromlist=[conf.encoder_name]), conf.encoder_name)
CPC_autoregressive_model = getattr(__import__('cpc_model', fromlist=[conf.autoregressive_model_name]), conf.autoregressive_model_name)
CPC_postnet = getattr(__import__('cpc_model', fromlist=[conf.postnet_name]), conf.postnet_name)

# Import our dataset for our data loader
CPC_dataset = getattr(__import__('cpc_data_loader', fromlist=[conf.dataset_name]), conf.dataset_name)

# Import our loss function
CPC_loss = getattr(__import__('cpc_loss', fromlist=[conf.loss_name]), conf.loss_name)

# Import our optimization algorithm
optimization_algorithm = getattr(__import__('torch.optim', fromlist=[conf.optimization_algorithm]), conf.optimization_algorithm)

# Import our learning rate scheduler
if conf.use_lr_scheduler:
    scheduler = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler]), conf.lr_scheduler)



if __name__ == '__main__':
    
    file = open(conf.name_of_log_textfile, 'w')
    file.close()
    
    # Read the text in the configuration file and add it to the logging file
    if conf.print_conf_contents:
        conf_file_lines = convert_py_conf_file_to_text(conf_file_name)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write(f'The configuration settings in the file {conf_file_name}:\n\n')
            for line in conf_file_lines:
                f.write(f'{line}\n')
            f.write('\n########################################################################################\n\n\n\n')
        
    
    # Use CUDA if it is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    with open(conf.name_of_log_textfile, 'a') as f:
        f.write(f'Process on {device}\n\n')

    # Initialize our models
    Encoder = CPC_encoder(**conf.encoder_params)
    AR_model = CPC_autoregressive_model(**conf.ar_model_params)
    W = CPC_postnet(**conf.w_params)
    
    # Pass the models to the available device
    Encoder = Encoder.to(device)
    AR_model = AR_model.to(device)
    W = W.to(device)
    
    # Give the parameters of our models to an optimizer
    model_parameters = list(Encoder.parameters()) + list(AR_model.parameters()) + list(W.parameters())
    optimizer = optimization_algorithm(params=model_parameters, **conf.optimization_algorithm_params)
    
    # Get our learning rate for later use
    learning_rate = optimizer.param_groups[0]['lr']
    
    # Give the optimizer to the learning rate scheduler
    if conf.use_lr_scheduler:
        lr_scheduler = scheduler(optimizer, **conf.lr_scheduler_params)

    # Instantiate our loss function as a class
    loss_function = CPC_loss(**conf.loss_params)

    # Variables for early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience_counter = 0
    
    if conf.load_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Loading model from file...\n')
            f.write(f'Loading model {conf.encoder_best_model_name}\n')
            f.write(f'Loading model {conf.ar_best_model_name}\n')
            f.write(f'Loading model {conf.w_best_model_name}\n')
        Encoder.load_state_dict(load(conf.encoder_best_model_name, map_location=device))
        AR_model.load_state_dict(load(conf.ar_best_model_name, map_location=device))
        W.load_state_dict(load(conf.w_best_model_name, map_location=device))
        best_model_encoder = deepcopy(Encoder.state_dict())
        best_model_ar = deepcopy(AR_model.state_dict())
        best_model_w = deepcopy(W.state_dict())
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n\n')
    else:
        best_model_encoder = None
        best_model_ar = None
        best_model_w = None
    
    
    # Initialize the data loaders
    if conf.train_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Initializing training set...\n')
        training_set = CPC_dataset(train_val_test='train', **conf.params_train_dataset)
        train_data_loader = DataLoader(training_set, **conf.params_train)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n')
            f.write('Initializing validation set...\n')
        validation_set = CPC_dataset(train_val_test='validation', **conf.params_validation_dataset)
        validation_data_loader = DataLoader(validation_set, **conf.params_train)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n')
    if conf.test_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Initializing test set...\n')
        test_set = CPC_dataset(train_val_test='test', **conf.params_test_dataset)
        test_data_loader = DataLoader(test_set, **conf.params_test)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n\n')
    
    
    # Determine our timesteps t. In the original CPC paper: "The output of the GRU at every timestep is used as the context c from which we
    # predict 12 timesteps in the future using the contrastive loss". We first determine whether our future_predicted_timesteps is a number
    # or a list of numbers.
    if isinstance(conf.future_predicted_timesteps, int):
        # future_predicted_timesteps is a number, so we have n timesteps where t in [0, 1, ..., n - 1] and our
        # n = num_frames_encoding - future_predicted_timesteps
        timesteps = np.arange(conf.num_frames_encoding - conf.future_predicted_timesteps)
        
        # Define the maximum future predicted timestep
        max_future_timestep = conf.future_predicted_timesteps
        
    elif isinstance(conf.future_predicted_timesteps, list):
        # future_predicted_timesteps is a list of numbers, so we define timesteps based on the largest number in the list
        max_future_timestep = max(conf.future_predicted_timesteps)
        
        # Then determine timesteps
        timesteps = np.arange(conf.num_frames_encoding - max_future_timestep)
        
    else:
        sys.exit('Configuration setting "future_predicted_timesteps" must be either an integer or a list of integers!')
    
    # Flag for indicating if max epochs are reached
    max_epochs_reached = 1
    
    # Start training our model
    if conf.train_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Starting training...\n')
        
        for epoch in range(1, conf.max_epochs + 1):
            
            start_time = time.time()
    
            # Lists containing the losses of each epoch
            epoch_loss_training = []
            epoch_loss_validation = []
    
            # Indicate that we are in training mode, so e.g. dropout will function
            Encoder.train()
            AR_model.train()
            W.train()
            
            if conf.rnn_models_used_in_ar_model:
                # Initialize the RNN's hidden vector
                hidden = None
            
            # Loop through every batch of our training data
            for train_data in train_data_loader:
                
                # The loss of the batch
                loss_batch = 0.0
                
                # Get the batches
                X_input, batch_labels = train_data
                X_input = X_input.to(device)
                
                # Zero the gradient of the optimizer
                optimizer.zero_grad()
                
                # Pass our data through the encoder
                Z = Encoder(X_input)
                
                # Create the output of the AR model. Note that the default AR_model flips the dimensions of Z from the
                # form [batch_size, num_features, num_frames_encoding] into [batch_size, num_frames_encoding, num_features])
                if conf.rnn_models_used_in_ar_model:
                    C, hidden = AR_model(Z, hidden)
                else:
                    C = AR_model(Z)
                
                # We go through each timestep one at a time
                for t in timesteps:
                    
                    # The encodings of the future timesteps, i.e. z_{t+k} where k in [1, 2, ..., max_future_timestep]
                    Z_future_timesteps = Z[:,(t + 1):(t + max_future_timestep + 1),:].permute(1, 0, 2)
                    
                    # c_t is the context latent representation that summarizes all encoder embeddings z_k where k <= t
                    c_t = C[:, t, :]
                    
                    # Each of the predicted future embeddings z_{t+k} where k in [1, 2, ..., max_future_timestep] (or k in
                    # future_predicted_timesteps if future_predicted_timesteps is a list) are computed using the post-net
                    predicted_future_Z = W(c_t)
                    
                    # Compute the loss of our model
                    if batch_labels != []:
                        loss = loss_function(Z_future_timesteps, predicted_future_Z, batch_labels)
                    else:
                        loss = loss_function(Z_future_timesteps, predicted_future_Z)
                    
                    # Add the loss to the total loss of the batch
                    loss_batch += loss
                    
                # Perform the backward pass
                loss_batch.backward()
                
                # Update the weights
                optimizer.step()

                # Add the loss to the total loss of the batch
                epoch_loss_training.append(loss_batch.item())
    
    
            # Indicate that we are in evaluation mode, so e.g. dropout will not function
            Encoder.eval()
            AR_model.eval()
            W.eval()
    
            # Make PyTorch not calculate the gradients, so everything will be much faster.
            with no_grad():
                
                # Loop through every batch of our validation data and perform a similar process as for the training data
                for validation_data in validation_data_loader:
                    loss_batch = 0.0
                    X_input, batch_labels = validation_data
                    X_input = X_input.to(device)
                    Z = Encoder(X_input)
                    if conf.rnn_models_used_in_ar_model:
                        C, hidden = AR_model(Z, hidden)
                    else:
                        C = AR_model(Z)
                        
                    for t in timesteps:
                        Z_future_timesteps = Z[:,(t + 1):(t + max_future_timestep + 1),:].permute(1, 0, 2)
                        c_t = C[:, t, :]
                        predicted_future_Z = W(c_t)
                        if batch_labels != []:
                            loss = loss_function(Z_future_timesteps, predicted_future_Z, batch_labels)
                        else:
                            loss = loss_function(Z_future_timesteps, predicted_future_Z)
                        loss_batch += loss
                    epoch_loss_validation.append(loss_batch.item())
    
            # Calculate mean losses
            epoch_loss_training = np.array(epoch_loss_training).mean()
            epoch_loss_validation = np.array(epoch_loss_validation).mean()
    
            # Check early stopping conditions
            if epoch_loss_validation < lowest_validation_loss:
                lowest_validation_loss = epoch_loss_validation
                patience_counter = 0
                best_model_encoder = deepcopy(Encoder.state_dict())
                best_model_ar = deepcopy(AR_model.state_dict())
                best_model_w = deepcopy(W.state_dict())
                best_validation_epoch = epoch
                if conf.save_best_model:
                    save(best_model_encoder, conf.encoder_best_model_name)
                    save(best_model_ar, conf.ar_best_model_name)
                    save(best_model_w, conf.w_best_model_name)
            else:
                patience_counter += 1
            
            end_time = time.time()
            epoch_time = end_time - start_time
            
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'Epoch: {epoch:04d} | '
                  f'Mean training loss: {epoch_loss_training:7.4f} | '
                  f'Mean validation loss: {epoch_loss_validation:7.4f} (lowest: {lowest_validation_loss:7.4f}) | '
                  f'Duration: {epoch_time:7.4f} seconds\n')
                
            # We check that do we need to update the learning rate based on the validation loss
            if conf.use_lr_scheduler:
                if conf.lr_scheduler == 'ReduceLROnPlateau':
                    lr_scheduler.step(epoch_loss_validation)
                else:
                    lr_scheduler.step()
                current_learning_rate = optimizer.param_groups[0]['lr']
                if current_learning_rate != learning_rate:
                    learning_rate = current_learning_rate
                    with open(conf.name_of_log_textfile, 'a') as f:
                        f.write(f'Updated learning rate after epoch {epoch} based on learning rate scheduler, now lr={learning_rate}\n')
            
            # If patience counter is fulfilled, stop the training
            if patience_counter >= conf.patience:
                max_epochs_reached = 0
                break
            
        if max_epochs_reached:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('\nMax number of epochs reached, stopping training\n\n')
        else:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('\nExiting due to early stopping\n\n')
        
        if best_model_encoder is None:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('\nNo best model. The criteria for the lowest acceptable validation loss not satisfied!\n\n')
            sys.exit('No best model, exiting...')
        else:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'\nBest epoch {best_validation_epoch} with validation loss {lowest_validation_loss}\n\n')
        
        
        
    # Test the model
    if conf.test_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('\n\nStarting testing... => ')
            
        # Load the best version of the model
        try:
            Encoder.load_state_dict(load(conf.encoder_best_model_name, map_location=device))
            AR_model.load_state_dict(load(conf.ar_best_model_name, map_location=device))
            W.load_state_dict(load(conf.w_best_model_name, map_location=device))
        except (FileNotFoundError, RuntimeError):
            Encoder.load_state_dict(best_model_encoder)
            AR_model.load_state_dict(best_model_ar)
            W.load_state_dict(best_model_w)
                
        testing_loss = []
        Encoder.eval()
        AR_model.eval()
        W.eval()
        with no_grad():
            
            if conf.rnn_models_used_in_ar_model:
                hidden = None
            
            for test_data in test_data_loader:
                loss_batch = 0.0
                X_input, batch_labels = test_data
                X_input = X_input.to(device)
                Z = Encoder(X_input)
                if conf.rnn_models_used_in_ar_model:
                    C, hidden = AR_model(Z, hidden)
                else:
                    C = AR_model(Z)
                    
                for t in timesteps:
                    Z_future_timesteps = Z[:,(t + 1):(t + max_future_timestep + 1),:].permute(1, 0, 2)
                    c_t = C[:, t, :]
                    predicted_future_Z = W(c_t)
                    if batch_labels != []:
                        loss = loss_function(Z_future_timesteps, predicted_future_Z, batch_labels)
                    else:
                        loss = loss_function(Z_future_timesteps, predicted_future_Z)
                    loss_batch += loss
                testing_loss.append(loss_batch.item())

            testing_loss = np.array(testing_loss).mean()
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'Testing loss: {testing_loss:7.4f}')
    
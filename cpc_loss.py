# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The loss function for the CPC model.

"""

import sys
import numpy as np
from torch import matmul, diag
from torch.nn import Module, LogSoftmax



class CPC_loss_no_classes(Module):
    """
    The CPC loss, implemented in a way that for each batch, the other samples in the same batch
    act as the negative samples. Note that in the loss calculation, a log density ratio log(f_k)
    is used as log(f_k) = matmul(Z_future_timesteps[k-1], predicted_future_Z[i].transpose(0,1)),
    w.r.t. Eq. (3) in the original CPC paper where f_k = exp(z_{t+k}^T * W_k * c_t).
    
    _____________________________________________________________________________________________
    Input parameters:
    
    future_predicted_timesteps: The future predicted timesteps (integer or a list of integers)
        
    Z_future_timesteps: The encodings of the future timesteps, i.e. z_{t+k} where k in
                        [1, 2, ..., num_future_predicted_timesteps].
    
    predicted_future_Z: The predicted future embeddings z_{t+k} where k in
                        [1, 2, ..., num_future_predicted_timesteps]
    
    _____________________________________________________________________________________________
    
    """
    
    def __init__(self, future_predicted_timesteps=12):
        super(CPC_loss_no_classes, self).__init__()
        
        # We first determine whether our future_predicted_timesteps is a number or a list of numbers.
        if isinstance(future_predicted_timesteps, int):
            # future_predicted_timesteps is a number, so we have future_predicted_timesteps loss calculations
            self.future_predicted_timesteps = np.arange(1, future_predicted_timesteps + 1)
            
        elif isinstance(future_predicted_timesteps, list):
            # future_predicted_timesteps is a list of numbers, so we have len(future_predicted_timesteps) loss calculations
            self.future_predicted_timesteps = future_predicted_timesteps
            
        else:
            sys.exit('Configuration setting "future_predicted_timesteps" must be either an integer or a list of integers!')

    def forward(self, Z_future_timesteps, predicted_future_Z):
        
        log_smax = LogSoftmax(dim=1)
        
        loss = 0
        num_future_predicted_timesteps = len(self.future_predicted_timesteps)
        batch_size = Z_future_timesteps.size()[1]
        
        # We go through each future timestep and compute the loss. Z_future_timesteps is of size
        # [future_predicted_timesteps, batch_size, num_features] or [len(future_predicted_timesteps), batch_size, num_features]
        # where num_features is the size of the encoding for each timestep produced by the encoder
        # --> with default values Z_future_timesteps.size() = torch.Size([12, 8, 512])
        i = 0
        for k in self.future_predicted_timesteps:
            loss -= diag(log_smax(matmul(Z_future_timesteps[k-1], predicted_future_Z[i].transpose(0,1)))).sum(dim=0)
            i += 1
            
        loss = loss / (batch_size * num_future_predicted_timesteps)
        
        return loss
    


class CPC_loss(Module):
    """
    The CPC loss, implemented so that for each batch, we first clean the batch from elements
    with the same label. For each occurring label in a batch, we find the first sample with the
    given label and leave out the rest. Then, we are left with a set of samples, each with a
    unique label, for which the other samples in the batch act as negative samples. Note that in
    the loss calculation, a log density ratio log(f_k) is used as
    log(f_k) = matmul(Z_future_timesteps[k-1], predicted_future_Z[i].transpose(0,1)),
    w.r.t. Eq. (3) in the original CPC paper where f_k = exp(z_{t+k}^T * W_k * c_t).
    
    _____________________________________________________________________________________________
    Input parameters:
    
    future_predicted_timesteps: The future predicted timesteps (integer or a list of integers)
        
    Z_future_timesteps: The encodings of the future timesteps, i.e. z_{t+k} where k in
                        [1, 2, ..., num_future_predicted_timesteps].
    
    predicted_future_Z: The predicted future embeddings z_{t+k} where k in
                        [1, 2, ..., num_future_predicted_timesteps]
                        
    batch_labels: The labels of the elements in each batch
    
    _____________________________________________________________________________________________
    
    """
    
    def __init__(self, future_predicted_timesteps=12):
        super(CPC_loss, self).__init__()
        
        # We first determine whether our future_predicted_timesteps is a number or a list of numbers.
        if isinstance(future_predicted_timesteps, int):
            # future_predicted_timesteps is a number, so we have future_predicted_timesteps loss calculations
            self.future_predicted_timesteps = np.arange(1, future_predicted_timesteps + 1)
            
        elif isinstance(future_predicted_timesteps, list):
            # future_predicted_timesteps is a list of numbers, so we have len(future_predicted_timesteps) loss calculations
            self.future_predicted_timesteps = future_predicted_timesteps
            
        else:
            sys.exit('Configuration setting "future_predicted_timesteps" must be either an integer or a list of integers!')

    def forward(self, Z_future_timesteps, predicted_future_Z, batch_labels):
        
        log_smax = LogSoftmax(dim=1)
        
        loss = 0
        num_future_predicted_timesteps = len(self.future_predicted_timesteps)
        batch_labels_unique = list(dict.fromkeys(batch_labels))
        indices_with_unique_label = [batch_labels.index(x) for x in batch_labels_unique]
            
        i = 0
        for k in self.future_predicted_timesteps:
            loss -= diag(log_smax(matmul(Z_future_timesteps[k-1, indices_with_unique_label], predicted_future_Z[i, indices_with_unique_label].transpose(0,1)))).sum(dim=0)
            i += 1
                
        loss = loss / (len(indices_with_unique_label) * num_future_predicted_timesteps)
        
        
        return loss


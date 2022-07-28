# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The subparts of the CPC model (encoder, autoregressive model, postnet).

"""

import numpy as np
import sys
from torch import stack
from torch.nn import Module, Linear, ReLU, Conv1d, BatchNorm1d, Dropout, GRU, ModuleList, ELU, Identity




class CPC_encoder(Module):
    """
    The CPC encoder (CNN with strided convolutions) from the original CPC paper.
    
    """
    
    def __init__(self,
                 conv_1_in_dim = 1,
                 conv_1_out_dim = 512,
                 conv_1_kernel_size = 10,
                 conv_1_stride = 5,
                 conv_1_padding = 3,
                 num_norm_features_1 = 512,
                 conv_2_in_dim = 512,
                 conv_2_out_dim = 512,
                 conv_2_kernel_size = 8,
                 conv_2_stride = 4,
                 conv_2_padding = 2,
                 num_norm_features_2 = 512,
                 conv_3_in_dim = 512,
                 conv_3_out_dim = 512,
                 conv_3_kernel_size = 4,
                 conv_3_stride = 2,
                 conv_3_padding = 1,
                 num_norm_features_3 = 512,
                 conv_4_in_dim = 512,
                 conv_4_out_dim = 512,
                 conv_4_kernel_size = 4,
                 conv_4_stride = 2,
                 conv_4_padding = 1,
                 num_norm_features_4 = 512,
                 conv_5_in_dim = 512,
                 conv_5_out_dim = 512,
                 conv_5_kernel_size = 4,
                 conv_5_stride = 2,
                 conv_5_padding = 1,
                 num_norm_features_5 = 512,
                 dropout = 0.0):

        super().__init__()

        self.conv_layer_1 = Conv1d(in_channels=conv_1_in_dim, out_channels=conv_1_out_dim, kernel_size=conv_1_kernel_size,
                                   stride=conv_1_stride, padding=conv_1_padding)
        self.batch_normalization_1 = BatchNorm1d(num_norm_features_1)
        
        self.conv_layer_2 = Conv1d(in_channels=conv_2_in_dim, out_channels=conv_2_out_dim, kernel_size=conv_2_kernel_size,
                                   stride=conv_2_stride, padding=conv_2_padding)
        self.batch_normalization_2 = BatchNorm1d(num_norm_features_2)
        
        self.conv_layer_3 = Conv1d(in_channels=conv_3_in_dim, out_channels=conv_3_out_dim, kernel_size=conv_3_kernel_size,
                                   stride=conv_3_stride, padding=conv_3_padding)
        self.batch_normalization_3 = BatchNorm1d(num_norm_features_3)
        
        self.conv_layer_4 = Conv1d(in_channels=conv_4_in_dim, out_channels=conv_4_out_dim, kernel_size=conv_4_kernel_size,
                                   stride=conv_4_stride, padding=conv_4_padding)
        self.batch_normalization_4 = BatchNorm1d(num_norm_features_4)
        
        self.conv_layer_5 = Conv1d(in_channels=conv_5_in_dim, out_channels=conv_5_out_dim, kernel_size=conv_5_kernel_size,
                                   stride=conv_5_stride, padding=conv_5_padding)
        self.batch_normalization_5 = BatchNorm1d(num_norm_features_5)
        
        self.non_linearity_relu = ReLU()
        self.dropout = Dropout(dropout)


    def forward(self, X):
        
        # Make the input X of size [batch_size, audio_window_length] into size [batch_size, 1, audio_window_length]
        # by adding a dummy dimension (number of channels)
        # --> with default values from torch.Size([8, 20480]) into torch.Size([8, 1, 20480])
        X = X.unsqueeze(1)
        X = self.dropout(self.non_linearity_relu(self.batch_normalization_1(self.conv_layer_1(X))))
        X = self.dropout(self.non_linearity_relu(self.batch_normalization_2(self.conv_layer_2(X))))
        X = self.dropout(self.non_linearity_relu(self.batch_normalization_3(self.conv_layer_3(X))))
        X = self.dropout(self.non_linearity_relu(self.batch_normalization_4(self.conv_layer_4(X))))
        X = self.dropout(self.non_linearity_relu(self.batch_normalization_5(self.conv_layer_5(X))))
        # X is now of size [batch_size, conv_5_out_dim, num_frames_encoding]
        # --> with default values torch.Size([8, 512, 128])
        
        return X
    


class CPC_encoder_mlp(Module):
    """
    A five-layer MLP encoder for 2D inputs (e.g. log-mel features).
    
    """
    
    def __init__(self,
                 linear_1_input_dim = 40,
                 linear_1_output_dim = 512,
                 num_norm_features_1 = 512,
                 linear_2_input_dim = 512,
                 linear_2_output_dim = 512,
                 num_norm_features_2 = 512,
                 linear_3_input_dim = 512,
                 linear_3_output_dim = 512,
                 num_norm_features_3 = 512,
                 normalization_type = 'batchnorm',
                 dropout = 0.2):

        super().__init__()
        
        if normalization_type == 'batchnorm':
            normalization_layer = BatchNorm1d
        elif normalization_type == None:
            normalization_layer = Identity
        else:
            sys.exit(f'Wrong value for argument "normalization_type": {normalization_type}')
        
        self.linear_layer_1 = Linear(in_features=linear_1_input_dim,
                              out_features=linear_1_output_dim)
        self.normalization_1 = normalization_layer(num_norm_features_1)
        
        self.linear_layer_2 = Linear(in_features=linear_2_input_dim,
                              out_features=linear_2_output_dim)
        self.normalization_2 = normalization_layer(num_norm_features_2)
                              
        self.linear_layer_3 = Linear(in_features=linear_3_input_dim,
                              out_features=linear_3_output_dim)
        self.normalization_3 = normalization_layer(num_norm_features_3)
        
        self.non_linearity_elu = ELU()
        self.dropout = Dropout(dropout)


    def forward(self, X):
        
        # X is now of size [batch_size, num_frames_input, num_features]
        X_output = []
        
        # We go through each timestep and produce an encoding
        for i in range(X.size()[1]):
            X_frame = X[:, i, :]
            X_frame = self.dropout(self.non_linearity_elu(self.normalization_1(self.linear_layer_1(X_frame))))
            X_frame = self.dropout(self.non_linearity_elu(self.normalization_2(self.linear_layer_2(X_frame))))
            X_frame = self.dropout(self.non_linearity_elu(self.normalization_3(self.linear_layer_3(X_frame))))
            X_output.append(X_frame)
        
        X_output = stack(X_output, dim=1)
        X_output = X_output.permute(0, 2, 1)
        # X_output is of size [batch_size, linear_3_output_dim, num_frames_encoding]
        
        return X_output



class CPC_autoregressive_model(Module):
    """
    The CPC autoregressive model (one-layer GRU) from the original CPC paper.
    
    """
    
    def __init__(self, encoding_dim = 512, output_dim = 256, num_layers = 1):

        super().__init__()
        
        self.cpc_ar = GRU(input_size=encoding_dim, hidden_size=output_dim, num_layers=num_layers, batch_first=True, bidirectional=False)


    def forward(self, X, hidden):
        
        # For the GRU, we want to reshape the data from the form [batch_size, num_features, num_frames_encoding] into
        # [batch_size, num_frames_encoding, num_features] where num_features is the size of the encoding for each
        # timestep produced by the encoder
        # --> with default values from torch.Size([8, 512, 128]) into torch.Size([8, 128, 512])
        X.transpose_(1, 2)
        
        if hidden == None:
            X, hidden = self.cpc_ar(X)
        else:
            X, hidden = self.cpc_ar(X, hidden)
        # X and hidden are now of size [batch_size, num_frames_encoding, output_dim] and [D * num_layers, batch_size, output_dim], respectively
        # (D = 2 if bidirectional=True, D = 1 otherwise)
        # --> with default values X.size() = torch.Size([8, 128, 256]) and hidden.size() = torch.Size([1, 8, 256])
        
        # We detach the hidden vector from the graph
        return X, hidden.detach()
    
    


class CPC_postnet(Module):
    """
    The CPC post-net (a linear transformation) from the original CPC paper. Alternatively, as
    mentioned by the authors, non-linear networks or recurrent neural networks could be used.
    
    """
    
    def __init__(self, encoding_dim = 512, ar_model_output_dim = 256, future_predicted_timesteps = 12):

        super().__init__()
        
        # We first determine whether our future_predicted_timesteps is a number or a list of numbers.
        if isinstance(future_predicted_timesteps, int):
            # future_predicted_timesteps is a number, so we have future_predicted_timesteps linear tranformations
            self.W = ModuleList([Linear(in_features=ar_model_output_dim, out_features=encoding_dim, bias=False) for i in np.arange(future_predicted_timesteps)])
            
        elif isinstance(future_predicted_timesteps, list):
            # future_predicted_timesteps is a list of numbers, so we have len(future_predicted_timesteps) linear transformations
            self.W = ModuleList([Linear(in_features=ar_model_output_dim, out_features=encoding_dim, bias=False) for i in range(len(future_predicted_timesteps))])
            
        else:
            sys.exit('Configuration setting "future_predicted_timesteps" must be either an integer or a list of integers!')


    def forward(self, X):
        
        predicted_future_Z = []
        for i in range(len(self.W)):
            predicted_future_Z.append(self.W[i](X))
        predicted_future_Z = stack(predicted_future_Z, dim=0)
        # predicted_future_Z is of size [future_predicted_timesteps, batch_size, num_features] or
        # [len(future_predicted_timesteps), batch_size, num_features] where num_features is the size
        # of the encoding for each timestep produced by the encoder
        # --> with default values predicted_future_Z.size() = torch.Size([12, 8, 512])
                
        return predicted_future_Z


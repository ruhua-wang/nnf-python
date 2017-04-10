# -*- coding: utf-8 -*-
# Global Imports
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from warnings import warn as warning

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D,ZeroPadding2D
#from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,RMSprop,adam


# Local Imports
from nnf.core.models.CNNModel import CNNModel
from nnf.core.models.NNModelPhase import NNModelPhase


class VGG16Model(CNNModel):
    """VGGNet16 Convolutional Neural Network Model.

    Notes
    -----
    ref: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, callbacks=None):
        """Constructs :obj:`VGG16Model` instance."""
        super().__init__(callbacks=callbacks)

    ##########################################################################
    # Protected: CNNModel Overrides
    ##########################################################################
    def _model_prefix(self):
        """Fetch the prefix for the file to be saved/loaded."""
        return "VGG16"

    def _build(self, input_shape, nb_class, data_format):
        """Build the keras VGG16."""
        assert(data_format == 'channels_last')  
        assert(input_shape == (224,224,3)) # and nb_class == 1000)                 
        inputs = Input(shape=input_shape)
       
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

        # Create model.
        self.net = Model(inputs, x, name='vgg16')
       
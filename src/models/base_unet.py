"""Script to create base Unet Architecture model"""

import tensorflow as tf
import sys
import os
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

CONF_FILE = "settings.json"

# Load configuration settings from JSON
with open(os.path.join(SRC_DIR, CONF_FILE), "r") as file:
    conf = json.load(file)


class Unet_model():
    def __init__(self):
        pass
    
    def encoder_block(self, inputs, num_filters): 
        """function to create encoder block"""
        # Convolution with 3x3 filter followed by ReLU activation 
        x = tf.keras.layers.Conv2D(num_filters,  
                                3,  
                                padding = 'same')(inputs) 
        x = tf.keras.layers.Activation('elu')(x) 
        
        # Convolution with 3x3 filter followed by ReLU activation 
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Conv2D(num_filters,  
                                3,  
                                padding = 'same')(x) 
        x = tf.keras.layers.Activation('elu')(x) 
    
        # Max Pooling with 2x2 filter 
        x = tf.keras.layers.MaxPool2D(pool_size = (2, 2), 
                                    strides = 2)(x) 
        
        return x
    
    def decoder_block(self, inputs, skip_features, num_filters): 
        """function to create decoder block"""
        # Upsampling with 2x2 filter 
        x = tf.keras.layers.Conv2DTranspose(num_filters, 
                                            (2, 2),  
                                            strides = 2,  
                                            padding = 'same')(inputs) 
        
        # Copy and crop the skip features  
        # to match the shape of the upsampled input 
        skip_features = tf.image.resize(skip_features, 
                                        size = (x.shape[1], 
                                                x.shape[2])) 
        x = tf.keras.layers.Concatenate()([x, skip_features]) 
        
        x = tf.keras.layers.Dropout(0.2)(x)

        # Convolution with 3x3 filter followed by ReLU activation 
        x = tf.keras.layers.Conv2D(num_filters, 
                                3,  
                                padding = 'same')(x) 
        x = tf.keras.layers.Activation('elu')(x) 
        
        # Convolution with 3x3 filter followed by ReLU activation 
        x = tf.keras.layers.Conv2D(num_filters, 3, padding = 'same')(x) 
        x = tf.keras.layers.Activation('elu')(x) 
        
        return x
    
    def build_architecture(self, input_shape = (conf['general']['cropped_image_size'],
                                                 conf['general']['cropped_image_size'], 3),
                                                   num_classes = 1):
        """function to build model architecture"""
        inputs = tf.keras.layers.Input(input_shape) 
        
        # Contracting Path 
        s1 = self.encoder_block(inputs, 16) 
        s2 = self.encoder_block(s1, 32) 
        s3 = self.encoder_block(s2, 64) 
        s4 = self.encoder_block(s3, 128) 
        
        # Bottleneck 
        b1 = tf.keras.layers.Conv2D(128, 3, padding = 'same')(s4) 
        b1 = tf.keras.layers.Activation('elu')(b1) 
        b1 = tf.keras.layers.Dropout(0.2)(b1)
        b1 = tf.keras.layers.Conv2D(128, 3, padding = 'same')(b1) 
        b1 = tf.keras.layers.Activation('elu')(b1) 
        
        # Expansive Path 
        s5 = self.decoder_block(b1, s4, 128) 
        s6 = self.decoder_block(s5, s3, 64) 
        s7 = self.decoder_block(s6, s2, 32) 
        s8 = self.decoder_block(s7, s1, 16) 
        
        # Output 
        outputs = tf.keras.layers.Conv2D(num_classes,  
                                        1,  
                                        padding = 'same',  
                                        activation = 'sigmoid')(s8) 
        
        model = tf.keras.models.Model(inputs = inputs,  
                                    outputs = outputs,  
                                    name = 'U-Net') 
        return model 
    

import tensorflow as tf
from tensorflow.keras.layers import Layer

class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Trainable parameters: gamma (scale) and beta (shift)
        self.gamma = self.add_weight(
            name='gamma',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta
    
    

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Concatenate, Input, LeakyReLU ,Dropout, Add
from tensorflow.keras.models import Model

def nnunet_3d(input_shape=(128, 128, 128, 4), n_classes=4, base_filters=32, dropout_rate=0.2):
    inputs = Input(input_shape)

    def conv_block_with_residual(x, filters, stage, dropout_rate=0.2):
        # Store input for residual connection
        shortcut = x

        # Convolution path
        x = Conv3D(filters, 3, padding='same', use_bias=False, name=f'conv{stage}_1')(x)
        x = InstanceNormalization(name=f'in{stage}_1')(x)
        x = LeakyReLU(alpha=0.2, name=f'leaky{stage}_1')(x)  # Increased alpha

        # Add dropout for regularization
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f'drop{stage}_1')(x)

        x = Conv3D(filters, 3, padding='same', use_bias=False, name=f'conv{stage}_2')(x)
        x = InstanceNormalization(name=f'in{stage}_2')(x)

        # Add residual connection if input has same dimensions
        if int(shortcut.shape[-1]) == filters:
            x = Add(name=f'residual{stage}')([x, shortcut])
        else:
            # If dimensions don't match, use 1x1 conv to match dimensions
            shortcut = Conv3D(filters, 1, padding='same', name=f'shortcut{stage}')(shortcut)
            x = Add(name=f'residual{stage}')([x, shortcut])

        x = LeakyReLU(alpha=0.2, name=f'leaky{stage}_out')(x)

        # Add dropout for the entire block
        if dropout_rate > 0:
            x = Dropout(dropout_rate/2, name=f'drop{stage}_out')(x)

        return x

    # Encoder with increased capacity
    c1 = conv_block_with_residual(inputs, base_filters, '1', dropout_rate)            # 128x128x128x32
    p1 = MaxPooling3D(2, name='pool1')(c1)                                            # 64x64x64x32

    c2 = conv_block_with_residual(p1, base_filters * 2, '2', dropout_rate)            # 64x64x64x64
    p2 = MaxPooling3D(2, name='pool2')(c2)                                            # 32x32x32x64

    c3 = conv_block_with_residual(p2, base_filters * 4, '3', dropout_rate)            # 32x32x32x128
    p3 = MaxPooling3D(2, name='pool3')(c3)                                            # 16x16x16x128

    c4 = conv_block_with_residual(p3, base_filters * 8, '4', dropout_rate)            # 16x16x16x256
    p4 = MaxPooling3D(2, name='pool4')(c4)                                            # 8x8x8x256

    # Bottleneck with increased capacity
    c5 = conv_block_with_residual(p4, base_filters * 16, '5', dropout_rate)           # 8x8x8x512

    # Decoder with Deep Supervision
    u4 = UpSampling3D(2, name='up4')(c5)                                              # 16x16x16x512
    u4 = Conv3D(base_filters * 8, 2, padding='same', name='upconv4')(u4)              # 16x16x16x256

    # Attention mechanism for skip connection
    att4 = tf.keras.layers.Multiply()([u4, c4])                                       # Element-wise multiplication
    concat4 = Concatenate(name='concat4')([u4, att4])                                 # Enhanced skip connection
    c6 = conv_block_with_residual(concat4, base_filters * 8, '6', dropout_rate)       # 16x16x16x256
    out4 = Conv3D(n_classes, 1, activation='softmax', name='out4')(c6)

    u3 = UpSampling3D(2, name='up3')(c6)                                              # 32x32x32x256
    u3 = Conv3D(base_filters * 4, 2, padding='same', name='upconv3')(u3)              # 32x32x32x128

    # Attention for skip connection
    att3 = tf.keras.layers.Multiply()([u3, c3])                                       # Element-wise multiplication
    concat3 = Concatenate(name='concat3')([u3, att3])                                 # Enhanced skip connection
    c7 = conv_block_with_residual(concat3, base_filters * 4, '7', dropout_rate)       # 32x32x32x128
    out3 = Conv3D(n_classes, 1, activation='softmax', name='out3')(c7)

    u2 = UpSampling3D(2, name='up2')(c7)                                              # 64x64x64x128
    u2 = Conv3D(base_filters * 2, 2, padding='same', name='upconv2')(u2)              # 64x64x64x64

    #Attention for skip connection
    att2 = tf.keras.layers.Multiply()([u2, c2])                                       # Element-wise multiplication
    concat2 = Concatenate(name='concat2')([u2, att2])                                 # Enhanced skip connection
    c8 = conv_block_with_residual(concat2, base_filters * 2, '8', dropout_rate)       # 64x64x64x64
    out2 = Conv3D(n_classes, 1, activation='softmax', name='out2')(c8)

    u1 = UpSampling3D(2, name='up1')(c8)                                              # 128x128x128x64
    u1 = Conv3D(base_filters, 2, padding='same', name='upconv1')(u1)                  # 128x128x128x32

    # Attention for skip connection
    att1 = tf.keras.layers.Multiply()([u1, c1])                                       # Element-wise multiplication
    concat1 = Concatenate(name='concat1')([u1, att1])                                 # Enhanced skip connection
    c9 = conv_block_with_residual(concat1, base_filters, '9', dropout_rate/2)         # 128x128x128x32
    outputs = Conv3D(n_classes, 1, activation='softmax', name='out_final')(c9)

    return Model(inputs=inputs, outputs=[outputs, out2, out3, out4])
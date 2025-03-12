from tensorflow import keras
from keras import layers
import cv2 as cv
import numpy as np


def convolution_block(x, filters, kernel_size=(3, 3), activation='relu'):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def transconv_block(x, filters, kernel_size=(3, 3), strides=(2, 2)):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def TransUNet(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Encoder
    conv1 = convolution_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = convolution_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = convolution_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bridge
    conv4 = convolution_block(pool3, 512)

    # Decoder
    upconv3 = transconv_block(conv4, 256)
    concat3 = layers.concatenate([upconv3, conv3], axis=-1)
    upconv3 = convolution_block(concat3, 256)

    upconv2 = transconv_block(upconv3, 128)
    concat2 = layers.concatenate([upconv2, conv2], axis=-1)
    upconv2 = convolution_block(concat2, 128)

    upconv1 = transconv_block(upconv2, 64)
    concat1 = layers.concatenate([upconv1, conv1], axis=-1)
    upconv1 = convolution_block(concat1, 64)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(upconv1)

    model = keras.Model(inputs, outputs)

    return model


def Model_Trans_Unet(Data, Target):
    input_shape = (256, 256, 3)  # Adjust input shape as needed
    num_classes = 3  # Modify the number of classes
    tar = []
    for i in range(len(Target)):
        targ = cv.cvtColor(Target[i], cv.COLOR_GRAY2RGB)
        tar.append(targ)
    tar = np.asarray(tar)
    model = TransUNet(input_shape, num_classes)
    model.summary()
    # Train the model with your data
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Data, tar, epochs=10)
    pred = model.predict(Data)
    return pred

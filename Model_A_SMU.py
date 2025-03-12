import numpy as np
import tensorflow as tf
from keras import Input
from keras.src.layers import MultiHeadAttention, Add, LayerNormalization, UpSampling2D, Conv2D, MaxPooling2D


# Swin Transformer Block
def swin_transformer_block(x, num_heads=3, ff_dim=32):
    # Multi-Head Self Attention
    att = MultiHeadAttention(key_dim=x.shape[-1], num_heads=num_heads)(x, x, x)
    att = Add()([x, att])
    att = LayerNormalization(epsilon=1e-6)(att)

    # Feed-Forward Networks
    ff = Conv2D(filters=ff_dim, kernel_size=1, activation='relu')(att)
    ff = Conv2D(filters=x.shape[-1], kernel_size=1)(ff)
    ff = Add()([att, ff])
    ff = LayerNormalization(epsilon=1e-6)(ff)

    return ff


# MobileUNet Architecture
def build_mobileunet(input_shape):
    # input_layer = Input(shape=input_shape)
    input_layer = Input(input_shape)
    # Encoder
    encoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    encoder_output = MaxPooling2D((2, 2))(encoder_output)

    # Swin Transformer Blocks
    for _ in range(3):
        encoder_output = swin_transformer_block(encoder_output)

    # Decoder
    decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    # Output
    output_layer = Conv2D(1, (1, 1), activation='sigmoid')(decoder_output)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def Model_A_SMU(image_data, tar, sol=None):
    if sol is None:
        sol = [5, 5, 5]

    input_shape = (image_data.shape[1], image_data.shape[2], 3)  # Adjust dimensions as per your input data

    Target = np.zeros((tar.shape[0], input_shape[0], input_shape[1], 1))
    for i in range(tar.shape[0]):
        temp_1 = np.resize(tar[i], (input_shape[0] * input_shape[1], 1))
        Target[i] = np.reshape(temp_1, (input_shape[0], input_shape[1], 1))

    model = build_mobileunet(input_shape)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # Train the model
    model.fit(image_data, Target, epochs=int(sol[1]), batch_size=2, steps_per_epoch=int(sol[2]))
    # Predict using the trained model
    predicted_images = model.predict(image_data)
    return predicted_images



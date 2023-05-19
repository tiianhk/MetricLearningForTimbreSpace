from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D,
                                     BatchNormalization,
                                     Activation,
                                     MaxPooling2D,
                                     Input,
                                     Dense,
                                     Flatten,
                                     Dot,
                                     Concatenate)


def ConvBlock(x, num_filters=64, kernel_size=(3,3), activation='relu'):
    x = Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling2D()(x)
    return x


def create_backbone_model(input_shape):

    # tensor shape (batch, frequency, time, channels)
    melspec = Input(shape=input_shape)
    x = ConvBlock(melspec)
    x = ConvBlock(x)
    x = ConvBlock(x)
    x = ConvBlock(x)

    # maxpooling over the time dimension
    x = MaxPooling2D(pool_size=(1, x.shape[-2]))(x)

    # project the flattened tensor to a 64d embedding
    x = Flatten()(x)
    x = Dense(64)(x)

    return Model(inputs=[melspec], outputs=[x], name='backbone')


def create_triplet_model(input_shape):

    # triplet input
    anchor = Input(shape = input_shape, name='anchor_input')
    positive = Input(shape = input_shape, name='positive_input')
    negative = Input(shape = input_shape, name='negative_input')

    # get backbone model, compute embeddings
    backbone_model = create_backbone_model(input_shape)
    anchor_embedding = backbone_model(anchor)
    positive_embedding = backbone_model(positive)
    negative_embedding = backbone_model(negative)

    # cosine similarity for normalized embeddings
    dist_anchor_positive = Dot(axes=1, normalize=True)([anchor_embedding, positive_embedding])
    dist_anchor_negative = Dot(axes=1, normalize=True)([anchor_embedding, negative_embedding])
    similarity_scores = Concatenate()([dist_anchor_positive, dist_anchor_negative])

    return Model(inputs=[anchor, positive, negative], outputs=similarity_scores, name='triplet')


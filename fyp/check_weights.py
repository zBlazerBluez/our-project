import numpy as numpy
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras import backend as K
NUM_ACTIONS = 128


def _build_model():

    l_input = Input(batch_shape=(None, 8, 8, 3))

    l1 = Conv2D(32, (2, 2), activation='relu', padding='same')(l_input)
    l2 = Conv2D(64, (2, 2), activation='relu', padding='same')(l1)
    l3 = Flatten()(l2)
    l4 = Flatten()(l_input)
    l5 = Dense(NUM_ACTIONS * 2, activation='relu')(l3)
    l6 = Dense(64, activation='relu')(l4)

    out_actions = Dense(NUM_ACTIONS, activation='softmax')(l5)
    out_value = Dense(1, activation='linear')(l6)

    model = Model(inputs=[l_input], outputs=[out_actions, out_value])
    model._make_predict_function()  # have to initialize before threading

    return model


model = _build_model()
# model.load_weights('trained_models/weights_a3c5_v2.h5')
for layer in model.layers:
    weights = layer.get_weights()
    # print(weights)
    if weights:
        print(weights[0].shape)
        # print(weights[1].shape)
        # print(weights[2].shape)

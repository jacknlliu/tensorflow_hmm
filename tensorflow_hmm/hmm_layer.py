from keras.layers import Lambda, Activation
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from tensorflow_hmm import HMMTensorflow

class HMMLayer(Layer):
    def __init__(self, states, length=None):
        # todo: perhaps states should just be inferred by the input shape
        # todo: create a few utility functions for generating transition matrices
        self.states = states
        self.P = np.ones((states, states), dtype=np.float32) * (0.01 / (states - 1))
        for i in range(states):
            self.P[i, i] = 0.99

        self.hmm = HMMTensorflow(self.P)

        super(HMMLayer, self).__init__()

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('input_shape must be 3, found {}'.format(len(input_shape)))

        super(HMMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # todo: only optionally apply sigmoid
        # todo: apply viterbi during inference
        x = Activation(K.sigmoid)(x)
        x = Lambda(lambda x: self.hmm.forward_backward(x)[0])(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

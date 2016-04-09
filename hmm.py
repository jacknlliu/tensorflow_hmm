import tensorflow as tf
import numpy as np


class HMM(object):
    """
    A class for Hidden Markov Models.

    The model attributes are:
    - K :: the number of states
    - P :: the K by K transition matrix (from state i to state j,
        (i, j) in [1..K])
    - p0 :: the initial distribution (defaults to starting in state 0)
    """

    def __init__(self, P, p0=None):
        self.K = P.shape[0]

        self.P = P
        self.logP = np.log(self.P)

        if p0 is None:
            self.p0 = np.ones(self.K)
            self.p0 /= sum(self.p0)
        elif len(p0) != self.K:
            raise ValueError(
                'dimensions of p0 {} must match P[0] {}'.format(
                    p0.shape, P.shape[0]))
        else:
            self.p0 = p0
        self.logp0 = np.log(self.p0)

    def lik(self, y):
        # if y == 0: return [1, 0]
        # if y == 1: return [0, 1]
        return y * np.array([0.0, 1.0]) + (1.0 - y) * np.array([1.0, 0.0])

    def log_lik(self, y):
        return np.log(self.lik(y))


class HMMNumpy(HMM):

    def forward_backward(self, y):
        # set up
        nT = y.size
        posterior = np.zeros((nT, self.K))
        forward = np.zeros((nT + 1, self.K))
        backward = np.zeros((nT + 1, self.K))

        # forward pass
        forward[0, :] = 1.0 / self.K
        for t in xrange(nT):
            tmp = np.multiply(
                np.matmul(forward[t, :], self.P),
                self.lik(y[t])
            )

            forward[t + 1, :] = tmp / np.sum(tmp)

        # backward pass
        backward[-1, :] = 1.0 / self.K
        for t in xrange(nT, 0, -1):
            tmp = np.matmul(
                np.matmul(
                    self.P, np.diag(self.lik(y[t - 1]))
                ),
                backward[t, :].transpose()
            ).transpose()

            backward[t - 1, :] = tmp / np.sum(tmp)

        # remove initial/final probabilities
        forward = forward[1:, :]
        backward = backward[:-1, :]

        # combine and normalize
        posterior = np.array(forward) * np.array(backward)
        # [:,None] expands sum to be correct size
        posterior = posterior / np.sum(posterior, 1)[:, None]

        return posterior, forward, backward

    def _viterbi_partial_forward(self, scores):
        tmpMat = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                tmpMat[i, j] = scores[i] + self.logP[i, j]
        return tmpMat

    def viterbi_decode(self, y):
        y = np.array(y)

        nT = y.shape[0]

        pathStates = np.zeros((nT, self.K), dtype=np.int)
        pathScores = np.zeros((nT, self.K))

        # initialize
        pathScores[0] = self.logp0 + self.log_lik(y[0])

        for t, yy in enumerate(y[1:]):
            # propagate forward
            tmpMat = self._viterbi_partial_forward(pathScores[t])

            # the inferred state
            pathStates[t + 1] = np.argmax(tmpMat, 0)
            pathScores[t + 1] = np.max(tmpMat, 0) + self.log_lik(yy)

        # now backtrack viterbi to find states
        s = np.zeros(nT, dtype=np.int)
        s[-1] = np.argmax(pathScores[-1])
        for t in range(nT - 1, 0, -1):
            s[t - 1] = pathStates[t, s[t]]

        return s, pathScores


class HMMTensorflow(HMM):

    def forward_backward(self, y):
        # set up
        if isinstance(y, np.ndarray):
            nT = y.size
        else:
            nT = len(y)

        posterior = np.zeros((nT, self.K))
        forward = []
        backward = np.zeros((nT + 1, self.K))

        # forward pass
        forward.append(
            tf.ones((1, self.K), dtype=tf.float64) * (1.0 / self.K)
        )
        for t in xrange(nT):
            # NOTE: np.matrix expands forward[t, :] into 2d and causes * to be
            # matrix multiplies instead of element wise that an array would be
            tmp = tf.mul(
                tf.matmul(forward[t], self.P),
                self.lik(y[t])
            )

            forward.append(tmp / tf.reduce_sum(tmp))

        # backward pass
        backward = [None] * (nT + 1)
        backward[-1] = tf.ones((1, self.K), dtype=tf.float64) * (1.0 / self.K)
        for t in xrange(nT, 0, -1):
            tmp = tf.transpose(
                tf.matmul(
                    tf.matmul(self.P, tf.diag(self.lik(y[t - 1]))),
                    tf.transpose(backward[t])
                )
            )
            backward[t - 1] = tmp / tf.reduce_sum(tmp)

        # remove initial/final probabilities
        forward = forward[1:]
        backward = backward[:-1]

        # combine and normalize
        posterior = [f * b for f, b in zip(forward, backward)]
        posterior = [p / tf.reduce_sum(p) for p in posterior]

        return posterior, forward, backward

    def _viterbi_partial_forward(self, scores):
        # first convert scores into shape [K, 1]
        # then concatenate K of them into shape [K, K]
        expanded_scores = tf.concat(
            1, [tf.expand_dims(scores, 1)] * self.K
        )
        return expanded_scores + self.logP

    def viterbi_decode(self, y, nT):
        # pathStates and pathScores wil be of type tf.Tensor.  They
        # are lists since tensorflow doesn't allow indexing, and the
        # list and order are only really necessary to build the unrolled
        # graph.  We never do any computation across all of time at once
        pathStates = []
        pathScores = []

        # initialize
        pathStates.append(None)
        pathScores.append(self.logp0 + self.log_lik(y[0]))

        for t, yy in enumerate(y[1:]):
            # propagate forward
            tmpMat = self._viterbi_partial_forward(pathScores[t])

            # the inferred state
            pathStates.append(tf.argmax(tmpMat, 0))
            pathScores.append(tf.reduce_max(tmpMat, 0) + self.log_lik(yy))

        # now backtrack viterbi to find states
        s = [0] * nT
        s[-1] = tf.argmax(pathScores[-1], 0)
        for t in range(nT - 1, 0, -1):
            s[t - 1] = tf.gather(pathStates[t], s[t])

        return s, pathScores

from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_hmm import HMMNumpy, HMMTensorflow


@pytest.fixture
def latch_P():
    P = np.array([[0.5, 0.5], [0.0, 1.0]])
    # P = np.array([[0.5, 0.5], [0.5, 0.5]])
    # P = np.array([[0.5, 0.5], [0.0000000001, 0.9999999999]])
    # P = np.array([[0.5, 0.5], [1e-50, 1 - 1e-50]])

    for i in range(2):
        for j in range(2):
            print('from', i, 'to', j, P[i, j])
    return P


@pytest.fixture
def hmm_latch(latch_P):
    return HMMNumpy(latch_P)


@pytest.fixture
def fair_P():
    return np.array([[0.5, 0.5], [0.5, 0.5]])


@pytest.fixture
def hmm_fair(fair_P):
    return HMMNumpy(fair_P)


@pytest.fixture
def hmm_tf_fair(fair_P):
    return HMMTensorflow(fair_P)


@pytest.fixture
def hmm_tf_latch(latch_P):
    return HMMTensorflow(latch_P)


def lik(y):
    """ given 1d vector of likliehoods length N, return matrix with
    shape (N, 2) where (N, 0) is 1 - y and (N, 1) is y.

    This makes it easy to convert a time series of probabilities
    into 2 states, off/on, for a simple HMM.
    """

    liklihood = np.array([y, y], float).T
    liklihood[:, 0] = 1 - liklihood[:, 0]
    return liklihood


def test_hmm_tf_fair_forward_backward(hmm_tf_fair, hmm_fair):
    y = lik(np.array([0, 0, 1, 1]))

    np_posterior, _, _ = hmm_fair.forward_backward(y)
    print('tf')
    g_posterior, _, _ = hmm_tf_fair.forward_backward(y)
    tf_posterior = np.concatenate(tf.Session().run(g_posterior))

    print('np_posterior', np_posterior)
    print('tf_posterior', tf_posterior)
    assert np.isclose(np_posterior, tf_posterior).all()


def test_hmm_tf_fair_forward_backward_multiple_batch(hmm_tf_fair, hmm_fair):
    y = lik(np.array([0, 0, 1, 1]))

    np_posterior, _, _ = hmm_fair.forward_backward(y)
    print('tf')
    y = np.stack([y] * 3)
    g_posterior, _, _ = hmm_tf_fair.forward_backward(y)
    tf_posterior = tf.Session().run(g_posterior)

    print('np_posterior', np_posterior)
    print('tf_posterior', tf_posterior)
    assert np.isclose(np_posterior, tf_posterior).all()
    # assert False


def test_hmm_tf_latch_forward_backward_multiple_batch(hmm_tf_latch, hmm_latch):
    y = lik(np.array([0, 0, 1, 1]))

    np_posterior, np_forward, _ = hmm_latch.forward_backward(y)
    y = np.stack([y] * 3)
    print('tf')
    g_posterior, g_forward, _ = hmm_tf_latch.forward_backward(y)
    tf_posterior = tf.Session().run(g_posterior)

    print('np_posterior', np_posterior)
    print('tf_posterior', tf_posterior)
    assert np.isclose(np_posterior, tf_posterior).all()


def test_lik():
    yin = np.array([0, 0.25, 0.5, 0.75, 1])
    y = lik(yin)

    assert np.all(y == np.array([
        [1.00, 0.00],
        [0.75, 0.25],
        [0.50, 0.50],
        [0.25, 0.75],
        [0.00, 1.00],
    ]))


def test_hmm_fair_forward_backward(hmm_fair):
    y = lik(np.array([0, 0, 1, 1]))

    posterior, f, b = hmm_fair.forward_backward(y)

    # if P is filled with 0.5, the only thing that matters is the emission
    # liklihood.  assert that the posterior is = the liklihood of y
    for i, yi in enumerate(y):
        liklihood = yi / np.sum(yi)
        assert np.isclose(posterior[i, :], liklihood).all()

    # assert that posterior for any given t sums to 1
    assert np.isclose(np.sum(posterior, 1), 1).all()


def test_hmm_latch_two_step_no_noise(hmm_latch):
    for i in range(2):
        for j in range(2):
            y = [i, i, j, j]
            # y = [i, j]

            if i == 1 and j == 0:
                continue

            print('*'*80)
            print(y)
            states, scores = hmm_latch.viterbi_decode(lik(y))

            assert all(states == y)


def test_hmm_tf_partial_forward(hmm_tf_latch, hmm_latch):
    scoress = [
        np.log(np.array([0, 1])),
        np.log(np.array([1, 0])),
        np.log(np.array([0.25, 0.75])),
        np.log(np.array([0.5, 0.5])),
    ]

    for scores in scoress:
        tf_ret = tf.Session().run(
            hmm_tf_latch._viterbi_partial_forward(scores)
        )
        np_ret = hmm_latch._viterbi_partial_forward(scores)

        assert (tf_ret == np_ret).all()


def test_hmm_tf_viterbi_decode(hmm_tf_latch, hmm_latch):
    ys = [
        lik(np.array([0, 0])),
        lik(np.array([1, 1])),
        lik(np.array([0, 1])),
        lik(np.array([0, 0.25, 0.5, 0.75, 1])),
    ]

    for y in ys:
        print(y)

        tf_s_graph, tf_scores_graph = hmm_tf_latch.viterbi_decode(y)
        tf_s = tf.Session().run(tf_s_graph)
        tf_scores = [tf_scores_graph[0]]
        tf_scores.extend([tf.Session().run(g) for g in tf_scores_graph[1:]])
        print(np.array(tf_scores))

        np_s, np_scores = hmm_latch.viterbi_decode(y)
        print(np_scores)

        assert (tf_s == np_s).all()
        print()


def test_hmm_tf_viterbi_decode_wrong_shape(hmm_tf_latch, hmm_latch):
    with pytest.raises(ValueError):
        hmm_tf_latch.viterbi_decode([0, 1, 1, 0])

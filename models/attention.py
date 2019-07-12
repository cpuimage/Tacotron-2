from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops, random_ops
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauMonotonicAttention


# From https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(
        cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context = array_ops.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(array_ops.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, next_attention_state


# Implementation for https://arxiv.org/abs/1906.00672

def monotonic_stepwise_attention(p_choose_i, previous_attention, mode):
    # p_choose_i, previous_alignments, previous_score: [batch_size, memory_size]
    # p_choose_i: probability to keep attended to the last attended entry i
    if mode == "parallel":
        pad = tf.zeros([tf.shape(p_choose_i)[0], 1], dtype=p_choose_i.dtype)
        attention = previous_attention * p_choose_i + tf.concat(
            [pad, previous_attention[:, :-1] * (1.0 - p_choose_i[:, :-1])], axis=1)
    elif mode == "hard":
        # Given that previous_alignments is one_hot
        move_next_mask = tf.concat([tf.zeros_like(previous_attention[:, :1]), previous_attention[:, :-1]], axis=1)
        stay_prob = tf.reduce_sum(p_choose_i * previous_attention, axis=1)  # [B]
        attention = tf.where(stay_prob > 0.5, previous_attention, move_next_mask)
    else:
        raise ValueError("mode must be 'parallel', or 'hard'.")
    return attention


def _stepwise_monotonic_probability_fn(score, previous_alignments, sigmoid_noise, mode, seed=None):
    if sigmoid_noise > 0:
        noise = random_ops.random_normal(array_ops.shape(score), dtype=score.dtype,
                                         seed=seed)
        score += sigmoid_noise * noise
    if mode == "hard":
        # When mode is hard, use a hard sigmoid
        p_choose_i = math_ops.cast(score > 0, score.dtype)
    else:
        p_choose_i = math_ops.sigmoid(score)
    alignments = monotonic_stepwise_attention(p_choose_i, previous_alignments, mode)
    return alignments


class BahdanauStepwiseMonotonicAttention(BahdanauMonotonicAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=True,
                 score_mask_value=0,
                 sigmoid_noise=2.0,
                 sigmoid_noise_seed=None,
                 score_bias_init=3.5,
                 mode="parallel",
                 dtype=None,
                 name="BahdanauStepwiseMonotonicAttention"):
        if dtype is None:
            dtype = tf.float32
        wrapped_probability_fn = functools.partial(
            _stepwise_monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
            seed=sigmoid_noise_seed)
        super(BahdanauMonotonicAttention, self).__init__(
            query_layer=tf.layers.Dense(
                num_units, name="query_layer", use_bias=False, dtype=dtype),
            memory_layer=tf.layers.Dense(
                num_units, name="memory_layer", use_bias=False, dtype=dtype),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name
        self._score_bias_init = score_bias_init

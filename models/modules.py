import tensorflow as tf
from functools import reduce


class LSTMImpl:
    LSTMCell = "tf.nn.rnn_cell.LSTMCell"
    LSTMBlockCell = "tf.contrib.rnn.LSTMBlockCell"

    all_list = [LSTMCell, LSTMBlockCell]


class GRUImpl:
    GRUCell = "tf.contrib.rnn.GRUCell"
    GRUBlockCellV2 = "tf.contrib.rnn.GRUBlockCellV2"

    all_list = [GRUCell, GRUBlockCellV2]


def lstm_cell_factory(lstm_impl, num_units):
    if lstm_impl == LSTMImpl.LSTMCell:
        cell = tf.nn.rnn_cell.LSTMCell(num_units)
        return cell
    elif lstm_impl == LSTMImpl.LSTMBlockCell:
        cell = tf.contrib.rnn.LSTMBlockCell(num_units)
        return cell
    else:
        raise ValueError(f"Unknown LSTM cell implementation: {lstm_impl}. Supported: {', '.join(LSTMImpl.all_list)}")


def gru_cell_factory(gru_impl, num_units):
    if gru_impl == GRUImpl.GRUCell:
        cell = tf.nn.rnn_cell.GRUCell(num_units)
        return cell
    elif gru_impl == GRUImpl.GRUBlockCellV2:
        cell = tf.contrib.rnn.GRUBlockCellV2(num_units)
        return cell
    else:
        raise ValueError(f"Unknown GRU cell implementation: {gru_impl}. Supported: {', '.join(GRUImpl.all_list)}")


class HighwayNet(tf.layers.Layer):

    def __init__(self, out_units,
                 h_kernel_initializer=None,
                 h_bias_initializer=None,
                 t_kernel_initializer=None,
                 t_bias_initializer=tf.constant_initializer(-1.0),
                 trainable=True, name=None, **kwargs):
        super(HighwayNet, self).__init__(name=name, trainable=trainable, **kwargs)
        self.out_units = out_units
        self.H = tf.layers.Dense(out_units, activation=tf.nn.relu, name="H",
                                 kernel_initializer=h_kernel_initializer,
                                 bias_initializer=h_bias_initializer)
        self.T = tf.layers.Dense(out_units, activation=tf.nn.sigmoid, name="T",
                                 kernel_initializer=t_kernel_initializer,
                                 bias_initializer=t_bias_initializer)

    def build(self, input_shape):
        with tf.control_dependencies([tf.assert_equal(self.out_units, input_shape[-1])]):
            self.built = True

    def call(self, inputs, **kwargs):
        h = self.H(inputs)
        t = self.T(inputs)
        return inputs - (h + inputs) * t

    def compute_output_shape(self, input_shape):
        return input_shape


class CBHG(tf.layers.Layer):

    def __init__(self, out_units, conv_channels, max_filter_width, projection1_out_channels, projection2_out_channels,
                 num_highway, is_training,
                 trainable=True, name=None, **kwargs):
        half_out_units = out_units // 2
        assert out_units % 2 == 0
        super(CBHG, self).__init__(name=name, trainable=trainable, **kwargs)

        self.out_units = out_units

        self.convolution_banks = [
            Conv1d(kernel_size,
                   conv_channels,
                   activation=tf.nn.relu,
                   is_training=is_training,
                   name=f"conv1d_K{kernel_size}")
            for kernel_size in range(1, max_filter_width + 1)]
        self.maxpool = tf.layers.MaxPooling1D(pool_size=2, strides=1, padding="SAME")

        self.projection1 = Conv1d(kernel_size=3,
                                  out_channels=projection1_out_channels,
                                  activation=tf.nn.relu,
                                  is_training=is_training,
                                  name="proj1")

        self.projection2 = Conv1d(kernel_size=3,
                                  out_channels=projection2_out_channels,
                                  activation=tf.identity,
                                  is_training=is_training,
                                  name="proj2")

        self.adjustment_layer = tf.layers.Dense(half_out_units)

        self.highway_nets = [HighwayNet(half_out_units) for _ in range(1, num_highway + 1)]

    def build(self, _):
        self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        conv_outputs = tf.concat([conv1d(inputs) for conv1d in self.convolution_banks], axis=-1)

        maxpool_output = self.maxpool(conv_outputs)

        proj1_output = self.projection1(maxpool_output)
        proj2_output = self.projection2(proj1_output)

        # residual connection
        highway_input = proj2_output + inputs

        if highway_input.shape[2] != self.out_units // 2:
            highway_input = self.adjustment_layer(highway_input)

        highway_output = reduce(lambda acc, hw: hw(acc), self.highway_nets, highway_input)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.GRUCell(self.out_units // 2),
            tf.nn.rnn_cell.GRUCell(self.out_units // 2),
            highway_output,
            sequence_length=input_lengths,
            dtype=highway_output.dtype)

        return tf.concat(outputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.out_units])


class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper for tf LSTM to create Zoneout LSTM Cell

    inspired by:
    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py

    Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.

    Many thanks to @Ondal90 for pointing this out. You sir are a hero!
    """

    def __init__(self, num_units, is_training, zoneout_factor_cell=0.0, zoneout_factor_output=0.0, state_is_tuple=True,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, **kwargs):
        super(ZoneoutLSTMCell, self).__init__(name=name, trainable=trainable, **kwargs)
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)

        if zm < 0. or zs > 1.:
            raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

        self._cell = lstm_cell_factory(lstm_impl, num_units)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Runs vanilla LSTM Cell and applies zoneout.
        """
        # Apply vanilla LSTM
        output, new_state = self._cell(inputs, state, scope)

        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
            new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])

        # Apply zoneout
        if self.is_training:
            keep_rate_cell = 1.0 - self._zoneout_cell
            keep_rate_output = 1.0 - self._zoneout_outputs
            c = keep_rate_cell * tf.nn.dropout(new_c - prev_c, keep_prob=keep_rate_cell) + prev_c
            h = keep_rate_output * tf.nn.dropout(new_h - prev_h, keep_prob=keep_rate_output) + prev_h
        else:
            c = new_c - self._zoneout_cell * (new_c + prev_c)
            h = new_h - self._zoneout_outputs * (new_h + prev_h)

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat([c, h], axis=1)

        return output, new_state


class EncoderConvolutions:
    """Encoder convolutional layers used to find local dependencies in inputs characters.
    """

    def __init__(self, is_training, kernel_size=(5,), channels=128, num_layers=3, drop_rate=0.5,
                 activation=tf.nn.relu,
                 name=None):
        """
        Args:
            is_training: Boolean, determines if the model is training or in inference to control dropout
            kernel_size: tuple or integer, The size of convolution kernels
            channels: integer, number of convolutional kernels
            activation: callable, postnet activation function for each convolutional layer
            name: Postnet scope.
        """
        super(EncoderConvolutions, self).__init__()
        self.is_training = is_training

        self.kernel_size = kernel_size
        self.channels = channels
        self.activation = activation
        self.name = 'enc_conv_layers' if name is None else name
        self.drop_rate = drop_rate
        self.num_layers = num_layers

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            x = inputs
            for i in range(self.num_layers):
                x = Conv1d(self.kernel_size, self.channels, self.activation,
                           self.is_training, self.drop_rate, 'conv_layer_{}_'.format(i + 1) + self.name)(x)
        return x


class EncoderRNN:
    """Encoder bidirectional one layer LSTM
    """

    def __init__(self, is_training, size=256, zoneout=0.1, scope=None):
        """
        Args:
            is_training: Boolean, determines if the model is training or in inference to control zoneout
            size: integer, the number of LSTM units for each direction
            zoneout: the zoneout factor
            scope: EncoderRNN scope.
        """
        super(EncoderRNN, self).__init__()
        self.is_training = is_training

        self.size = size
        self.zoneout = zoneout
        self.scope = 'encoder_lstm' if scope is None else scope

        # Create forward LSTM Cell
        self._fw_cell = ZoneoutLSTMCell(size, is_training,
                                        zoneout_factor_cell=zoneout,
                                        zoneout_factor_output=zoneout,
                                        name='encoder_fw_lstm')

        # Create backward LSTM Cell
        self._bw_cell = ZoneoutLSTMCell(size, is_training,
                                        zoneout_factor_cell=zoneout,
                                        zoneout_factor_output=zoneout,
                                        name='encoder_bw_lstm')

    def __call__(self, inputs, input_lengths):
        with tf.variable_scope(self.scope):
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                self._fw_cell,
                self._bw_cell,
                inputs,
                sequence_length=input_lengths,
                dtype=tf.float32,
                swap_memory=True)

            return tf.concat(outputs, axis=2)  # Concat and return forward + backward outputs


class PreNet(tf.layers.Layer):

    def __init__(self, out_units, is_training, drop_rate=0.5,
                 apply_dropout_on_inference=False,
                 trainable=True, name=None, **kwargs):
        super(PreNet, self).__init__(name=name, trainable=trainable, **kwargs)
        self.out_units = out_units
        self.drop_rate = drop_rate
        self.is_training = is_training
        self.apply_dropout_on_inference = apply_dropout_on_inference
        self.dense = tf.layers.Dense(out_units, activation=tf.nn.relu)

    def build(self, _):
        self.built = True

    def call(self, inputs, **kwargs):
        dense = self.dense(inputs)
        dropout = tf.layers.dropout(dense, rate=self.drop_rate, training=self.dropout_enabled)
        return dropout

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)

    @property
    def dropout_enabled(self):
        return self.is_training or self.apply_dropout_on_inference


class DecoderRNN:
    """Decoder two uni directional LSTM Cells
    """

    def __init__(self, is_training, layers=2, size=1024, zoneout=0.1, scope=None):
        """
        Args:
            is_training: Boolean, determines if the model is in training or inference to control zoneout
            layers: integer, the number of LSTM layers in the decoder
            size: integer, the number of LSTM units in each layer
            zoneout: the zoneout factor
        """
        super(DecoderRNN, self).__init__()
        self.is_training = is_training

        self.layers = layers
        self.size = size
        self.zoneout = zoneout
        self.scope = 'decoder_rnn' if scope is None else scope

        # Create a set of LSTM layers
        self.rnn_layers = [ZoneoutLSTMCell(size, is_training,
                                           zoneout_factor_cell=zoneout,
                                           zoneout_factor_output=zoneout,
                                           name='decoder_LSTM_{}'.format(i + 1)) for i in range(layers)]

        self._cell = tf.contrib.rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=True)

    def __call__(self, inputs, states):
        with tf.variable_scope(self.scope):
            return self._cell(inputs, states)


class FrameProjection:
    """Projection layer to r * num_mels dimensions or num_mels dimensions
    """

    def __init__(self, shape=80, activation=None, scope=None):
        """
        Args:
            shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
            activation: callable, activation function
            scope: FrameProjection scope.
        """
        super(FrameProjection, self).__init__()

        self.shape = shape
        self.activation = activation

        self.scope = 'Linear_projection' if scope is None else scope
        self.dense = tf.layers.Dense(units=shape, activation=activation, name='projection_{}'.format(self.scope))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            # If activation==None, this returns a simple Linear projection
            # else the projection will be passed through an activation function
            # output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
            # 	name='projection_{}'.format(self.scope))
            output = self.dense(inputs)

            return output


class StopProjection:
    """Projection to a scalar and through a sigmoid activation
    """

    def __init__(self, is_training, shape=1, activation=tf.nn.sigmoid, scope=None):
        """
        Args:
            is_training: Boolean, to control the use of sigmoid function as it is useless to use it
                during training since it is integrate inside the sigmoid_crossentropy loss
            shape: integer, dimensionality of output space. Defaults to 1 (scalar)
            activation: callable, activation function. only used during inference
            scope: StopProjection scope.
        """
        super(StopProjection, self).__init__()
        self.is_training = is_training

        self.shape = shape
        self.activation = activation
        self.scope = 'stop_token_projection' if scope is None else scope

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output = tf.layers.dense(inputs, units=self.shape,
                                     activation=None, name='projection_{}'.format(self.scope))

            # During training, don't use activation as it is integrated inside the sigmoid_cross_entropy loss function
            if self.is_training:
                return output
            return self.activation(output)


class PostNetLinear(tf.layers.Layer):
    def __init__(self, is_training,
                 num_freq,
                 cbhg_out_units=256, conv_channels=128, max_filter_width=8,
                 projection1_out_channels=256,
                 projection2_out_channels=80,
                 num_highway=4,
                 trainable=True, name=None, **kwargs):
        super(PostNetLinear, self).__init__(name=name, trainable=trainable, **kwargs)
        self.cbhg = CBHG(
            cbhg_out_units, conv_channels, max_filter_width, projection1_out_channels, projection2_out_channels,
            num_highway, is_training=is_training)
        self.linear = tf.layers.Dense(num_freq)

    def build(self, _):
        self.built = True

    def call(self, inputs, **kwargs):
        cbhg_output = self.cbhg(inputs)
        dense_output = self.linear(cbhg_output)
        return dense_output

    def compute_output_shape(self, input_shape):
        return self.linear.compute_output_shape(input_shape)


class PostNetResidual(tf.layers.Layer):
    """Postnet that takes final decoder output and fine tunes it (using vision on past and future frames)
       """

    def __init__(self, out_units, num_postnet_layers, kernel_size, out_channels, is_training, drop_rate=0.5,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(PostNetResidual, self).__init__(name=name, trainable=trainable, **kwargs)

        final_conv_layer = Conv1d(kernel_size, out_channels, activation=None, is_training=is_training,
                                  drop_rate=drop_rate,
                                  name=f"conv1d_{num_postnet_layers}",
                                  dtype=dtype)

        self.convolutions = [Conv1d(kernel_size, out_channels, activation=tf.nn.tanh, is_training=is_training,
                                    drop_rate=drop_rate,
                                    name=f"conv1d_{i}",
                                    dtype=dtype) for i in
                             range(1, num_postnet_layers)] + [final_conv_layer]

        self.projection_layer = tf.layers.Dense(out_units, dtype=dtype)

    def call(self, inputs, **kwargs):
        output = reduce(lambda acc, conv: conv(acc), self.convolutions, inputs)
        # Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
        projected = self.projection_layer(output)
        # Project residual to same dimension as mel spectrogram
        # ==> [batch_size, decoder_steps * r, num_mels]
        summed = inputs + projected
        return summed


class Conv1d(tf.layers.Layer):

    def __init__(self, kernel_size, out_channels, activation, is_training, drop_rate=0.0, name=None,
                 use_bias=False,
                 strides=1,
                 trainable=True, **kwargs):
        super(Conv1d, self).__init__(name=name, trainable=trainable, **kwargs)
        self.is_training = is_training
        self.activation = activation
        self.drop_rate = drop_rate
        self.conv1d = tf.layers.Conv1D(out_channels, kernel_size, strides=strides, use_bias=use_bias, activation=None,
                                       padding="SAME")

    def build(self, _):
        self.built = True

    def call(self, inputs, **kwargs):
        conv1d = self.conv1d(inputs)
        batch_normalization = tf.layers.batch_normalization(conv1d, training=self.is_training)
        # fused_batch_norm (and 16bit precision) is only supported for 4D tensor
        # conv1d_rank4 = tf.expand_dims(conv1d, axis=2)
        # batch_normalization_rank4 = tf.layers.batch_normalization(conv1d_rank4, training=self.is_training)
        # batch_normalization = tf.squeeze(batch_normalization_rank4, axis=2)
        output = self.activation(batch_normalization) if self.activation is not None else batch_normalization
        output = tf.layers.dropout(output, self.drop_rate, training=self.is_training)
        return output

    def compute_output_shape(self, input_shape):
        return self.conv1d.compute_output_shape(input_shape)


def _round_up_tf(x, multiple):
    # Tf version of remainder = x % multiple
    remainder = tf.mod(x, multiple)
    # Tf version of return x if remainder == 0 else x + multiple - remainder
    x_round = tf.cond(tf.equal(remainder, tf.zeros(tf.shape(remainder), dtype=tf.int32)),
                      lambda: x,
                      lambda: x + multiple - remainder)

    return x_round


def sequence_mask(lengths, r, expand=True):
    """Returns a 2-D or 3-D tensorflow sequence mask depending on the argument 'expand'
    """
    max_len = tf.reduce_max(lengths)
    max_len = _round_up_tf(max_len, tf.convert_to_tensor(r))
    if expand:
        return tf.expand_dims(tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32), axis=-1)
    return tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)


def masked_mse(targets, outputs, targets_lengths, hparams, mask=None):
    """Computes a masked Mean Squared Error
    """

    # [batch_size, time_dimension, 1]
    # example:
    # sequence_mask([1, 3, 2], 5) = [[[1., 0., 0., 0., 0.]],
    #							    [[1., 1., 1., 0., 0.]],
    #							    [[1., 1., 0., 0., 0.]]]
    # Note the maxlen argument that ensures mask shape is compatible with r>1
    # This will by default mask the extra paddings caused by r>1
    if mask is None:
        mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)

    # [batch_size, time_dimension, channel_dimension(mels)]
    ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
    mask_ = mask * ones

    with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
        return tf.losses.mean_squared_error(labels=targets, predictions=outputs, weights=mask_)


def masked_sigmoid_cross_entropy(targets, outputs, targets_lengths, hparams, mask=None):
    """Computes a masked SigmoidCrossEntropy with logits
    """

    # [batch_size, time_dimension]
    # example:
    # sequence_mask([1, 3, 2], 5) = [[1., 0., 0., 0., 0.],
    #							    [1., 1., 1., 0., 0.],
    #							    [1., 1., 0., 0., 0.]]
    # Note the maxlen argument that ensures mask shape is compatible with r>1
    # This will by default mask the extra paddings caused by r>1
    if mask is None:
        mask = sequence_mask(targets_lengths, hparams.outputs_per_step, False)

    with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask))]):
        # Use a weighted sigmoid cross entropy to measure the <stop_token> loss. Set hparams.cross_entropy_pos_weight to 1
        # will have the same effect as  vanilla tf.nn.sigmoid_cross_entropy_with_logits.
        losses = tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=outputs,
                                                          pos_weight=hparams.cross_entropy_pos_weight)

    with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
        masked_loss = losses * mask

    return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)


def masked_linear_loss(targets, outputs, targets_lengths, hparams, mask=None):
    """Computes a masked MAE loss with priority to low frequencies
    """

    # [batch_size, time_dimension, 1]
    # example:
    # sequence_mask([1, 3, 2], 5) = [[[1., 0., 0., 0., 0.]],
    #							    [[1., 1., 1., 0., 0.]],
    #							    [[1., 1., 0., 0., 0.]]]
    # Note the maxlen argument that ensures mask shape is compatible with r>1
    # This will by default mask the extra paddings caused by r>1
    if mask is None:
        mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)

    # [batch_size, time_dimension, channel_dimension(freq)]
    ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
    mask_ = mask * ones

    l1 = tf.abs(targets - outputs)
    n_priority_freq = int(2000 / (hparams.sample_rate * 0.5) * hparams.num_freq)

    with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
        masked_l1 = l1 * mask_
        masked_l1_low = masked_l1[:, :, 0:n_priority_freq]

    mean_l1 = tf.reduce_sum(masked_l1) / tf.reduce_sum(mask_)
    mean_l1_low = tf.reduce_sum(masked_l1_low) / tf.reduce_sum(mask_)

    return 0.5 * mean_l1 + 0.5 * mean_l1_low

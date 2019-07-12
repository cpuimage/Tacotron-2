from utils.text_sequence import symbols
from infolog import log
from models.helpers import TacoTrainingHelper, TacoTestHelper
from models.modules import *
from tensorflow.contrib.seq2seq import dynamic_decode
from models.Architecture_wrappers import TacotronEncoderCell, TacotronDecoderCell
from models.custom_decoder import CustomDecoder
from models.attention import BahdanauStepwiseMonotonicAttention

import numpy as np


class Tacotron2:
    """Tacotron-2 Feature prediction Model.
    """

    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, mel_targets=None, stop_token_targets=None, linear_targets=None,
                   targets_lengths=None, gta=False,
                   global_step=None, is_training=False, is_evaluating=False):
        """
        Initializes the model for inference
        sets "mel_outputs" and "alignments" fields.
        Args:
            - inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
              steps in the input time series, and values are character IDs
            - input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
            of each sequence in inputs.
            - mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
            of steps in the output time series, M is num_mels, and values are entries in the mel
            spectrogram. Only needed for training.
        """
        if mel_targets is None and stop_token_targets is not None:
            raise ValueError('no multi targets were provided but token_targets were given')
        if mel_targets is not None and stop_token_targets is None and not gta:
            raise ValueError('Mel targets are provided without corresponding token_targets')
        if not gta and self._hparams.predict_linear and linear_targets is None and is_training:
            raise ValueError(
                'Model is set to use post processing to predict linear spectrograms in training but no linear targets given!')
        if gta and linear_targets is not None:
            raise ValueError('Linear spectrogram prediction is not supported in GTA mode!')
        if is_training and self._hparams.mask_decoder and targets_lengths is None:
            raise RuntimeError('Model set to mask paddings but no targets lengths provided for the mask!')
        if is_training and is_evaluating:
            raise RuntimeError('Model can not be in training and evaluation modes at the same time!')

        params = self._hparams

        batch_size = tf.shape(inputs)[0]
        t2_output_range = (-params.max_abs_value, params.max_abs_value) if params.symmetric_mels else (
            0, params.max_abs_value)

        with tf.variable_scope('inference'):
            assert params.teacher_forcing_mode in ('constant', 'scheduled')
            if params.teacher_forcing_mode == 'scheduled' and is_training:
                assert global_step is not None

            # GTA is only used for predicting mels to train Wavenet vocoder, so we ommit post processing when doing GTA synthesis
            post_condition = params.predict_linear and not gta

            # Embeddings ==> [batch_size, sequence_length, embedding_dim]
            self.embedding_table = tf.get_variable(
                'inputs_embedding', [len(symbols), params.embedding_dim], dtype=tf.float32)
            embedded_inputs = tf.nn.embedding_lookup(self.embedding_table, inputs)

            # Encoder Cell ==> [batch_size, encoder_steps, encoder_lstm_units]
            encoder_cell = TacotronEncoderCell(
                EncoderConvolutions(is_training, kernel_size=params.enc_conv_kernel_size,
                                    channels=params.enc_conv_channels, drop_rate=params.dropout_rate,
                                    num_layers=params.enc_conv_num_layers
                                    , name='encoder_convolutions'),
                EncoderRNN(is_training, size=params.encoder_lstm_units,
                           zoneout=params.zoneout_rate, scope='encoder_lstm'))

            encoder_outputs = encoder_cell(embedded_inputs, input_lengths)

            # For shape visualization purpose
            enc_conv_output_shape = encoder_cell.conv_output_shape

            # Decoder Parts
            # Attention Decoder Prenet
            prenet = [PreNet(out_unit, is_training, params.dropout_rate, name="decoder_prenet_{}".format(out_unit)) for
                      out_unit in params.prenet_layers]

            # Attention Mechanism
            attention_mechanism = BahdanauStepwiseMonotonicAttention(num_units=params.attention_dim,
                                                                     memory=encoder_outputs,
                                                                     memory_sequence_length=input_lengths)
            # Decoder LSTM Cells
            decoder_lstm = DecoderRNN(is_training, layers=params.decoder_layers,
                                      size=params.decoder_lstm_units, zoneout=params.zoneout_rate,
                                      scope='decoder_lstm')
            # Frames Projection layer
            frame_projection = FrameProjection(params.num_mels * params.outputs_per_step,
                                               scope='linear_transform_projection')
            # <stop_token> projection layer
            stop_projection = StopProjection(is_training or is_evaluating, shape=params.outputs_per_step,
                                             scope='stop_token_projection')

            # Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
            decoder_cell = TacotronDecoderCell(
                prenet,
                attention_mechanism,
                decoder_lstm,
                frame_projection,
                stop_projection)

            # Define the helper for our decoder
            if is_training or is_evaluating or gta:
                self.helper = TacoTrainingHelper(batch_size, mel_targets, params, gta, is_evaluating,
                                                 global_step)
            else:
                self.helper = TacoTestHelper(batch_size, params)

            # initial decoder state
            decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            # Only use max iterations at synthesis time
            max_iters = tf.reduce_max(targets_lengths) // params.outputs_per_step + 1 if (
                    is_training or is_evaluating) else tf.reduce_max(input_lengths) * params.outputs_per_step
            # Decode
            (frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
                CustomDecoder(decoder_cell, self.helper, decoder_init_state),
                impute_finished=False,
                maximum_iterations=max_iters,
                swap_memory=params.swap_with_cpu)

            # Reshape outputs to be one output per entry
            # ==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
            decoder_output = tf.reshape(frames_prediction, [batch_size, -1, params.num_mels])
            stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

            if params.clip_outputs:
                decoder_output = tf.minimum(
                    tf.maximum(decoder_output, t2_output_range[0] - params.lower_bound_decay), t2_output_range[1])

            postnet_output = PostNetResidual(out_units=params.num_mels, num_postnet_layers=params.postnet_num_layers,
                                             kernel_size=params.postnet_kernel_size,
                                             out_channels=params.postnet_channels,
                                             is_training=is_training, drop_rate=params.dropout_rate)(decoder_output)

            # Compute the mel spectrogram
            mel_outputs = decoder_output + postnet_output

            if params.clip_outputs:
                mel_outputs = tf.minimum(tf.maximum(mel_outputs, t2_output_range[0] - params.lower_bound_decay),
                                         t2_output_range[1])

            if post_condition:
                # Add post-processing CBHG. This does a great job at extracting features from mels before projection to Linear specs.
                linear_outputs = PostNetLinear(is_training,
                                               params.num_freq,
                                               params.post_net_cbhg_out_units,
                                               params.post_net_conv_channels,
                                               params.post_net_max_filter_width,
                                               params.post_net_projection1_out_channels,
                                               params.post_net_projection2_out_channels,
                                               params.post_net_num_highway)(mel_outputs)

                if params.clip_outputs:
                    linear_outputs = tf.minimum(
                        tf.maximum(linear_outputs, t2_output_range[0] - params.lower_bound_decay),
                        t2_output_range[1])

            # Grab alignments from the final decoder state
            alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

            self.decoder_output = decoder_output
            self.alignments = alignments
            self.stop_token_prediction = stop_token_prediction
            self.mel_outputs = mel_outputs

            if post_condition:
                self.linear_outputs = linear_outputs
        log('initialisation done')

        if is_training:
            self.ratio = self.helper._ratio
        self.inputs = inputs
        self.input_lengths = input_lengths
        self.mel_targets = mel_targets
        self.linear_targets = linear_targets
        self.targets_lengths = targets_lengths
        self.stop_token_targets = stop_token_targets

        self.all_vars = tf.trainable_variables()

        log('Initialized Tacotron2 model. Dimensions (? = dynamic shape): ')
        log('  Train mode:               {}'.format(is_training))
        log('  Eval mode:                {}'.format(is_evaluating))
        log('  GTA mode:                 {}'.format(gta))
        log('  Synthesis mode:           {}'.format(not (is_training or is_evaluating)))
        log('  Input:                    {}'.format(inputs.shape))
        log('  embedding:                {}'.format(embedded_inputs.shape))
        log('  enc conv out:             {}'.format(enc_conv_output_shape))
        log('  encoder out:              {}'.format(encoder_outputs.shape))
        log('  decoder out:              {}'.format(self.decoder_output.shape))
        log('  postnet out:             {}'.format(postnet_output.shape))
        log('  mel out:                  {}'.format(self.mel_outputs.shape))
        if post_condition:
            log('  linear out:               {}'.format(self.linear_outputs.shape))
        log('  <stop_token> out:         {}'.format(self.stop_token_prediction.shape))

        # 1_000_000 is causing syntax problems for some people?! Python please :)
        log('  Tacotron2 Parameters       {:.3f} Million.'.format(
            np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))

    def add_loss(self):

        """Adds loss to the model. Sets "loss" field. initialize must have been called."""

        hp = self._hparams

        with tf.variable_scope('loss'):
            if hp.mask_decoder:
                # Compute loss of predictions before postnet
                before = masked_mse(self.mel_targets, self.decoder_output,
                                    self.targets_lengths,
                                    hparams=self._hparams)
                # Compute loss after postnet
                after = masked_mse(self.mel_targets, self.mel_outputs,
                                   self.targets_lengths,
                                   hparams=self._hparams)
                # Compute <stop_token> loss (for learning dynamic generation stop)
                stop_token_loss = masked_sigmoid_cross_entropy(self.stop_token_targets,
                                                               self.stop_token_prediction,
                                                               self.targets_lengths,
                                                               hparams=self._hparams)
                # Compute masked linear loss
                if hp.predict_linear:
                    # Compute Linear L1 mask loss (priority to low frequencies)
                    linear_loss = masked_linear_loss(self.linear_targets, self.linear_outputs,
                                                     self.targets_lengths, hparams=self._hparams)
                else:
                    linear_loss = 0.
            else:
                # Compute loss of predictions before postnet
                before = tf.losses.mean_squared_error(self.mel_targets, self.decoder_output)
                # Compute loss after postnet
                after = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)
                # Compute <stop_token> loss (for learning dynamic generation stop)
                stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.stop_token_targets,
                    logits=self.stop_token_prediction))

                if hp.predict_linear:
                    # Compute linear loss
                    # From https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
                    # Prioritize loss for frequencies under 2000 Hz.
                    l1 = tf.abs(self.linear_targets - self.linear_outputs)
                    n_priority_freq = int(2000 / (hp.sample_rate * 0.5) * hp.num_freq)
                    linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:, :, 0:n_priority_freq])
                else:
                    linear_loss = 0.

            # Compute the regularization weight
            if hp.scale_regularization:
                reg_weight_scaler = 1. / (2 * hp.max_abs_value) if hp.symmetric_mels else 1. / hp.max_abs_value
                reg_weight = hp.reg_weight * reg_weight_scaler
            else:
                reg_weight = hp.reg_weight

            # Regularize variables
            # Exclude all types of bias, RNN (Bengio et al. On the difficulty of training recurrent neural networks), embeddings and prediction projection layers.
            # Note that we consider attention mechanism v_a weights as a prediction projection layer and we don't regularize it. (This gave better stability)
            regularization = tf.add_n([tf.nn.l2_loss(v) for v in self.all_vars
                                       if not (
                        'bias' in str(v.name).lower() or '_projection' in str(
                    v.name).lower() or 'inputs_embedding' in str(v.name).lower()
                        or 'rnn' in str(v.name).lower() or 'lstm' in str(v.name).lower())]) * reg_weight

            # Compute final loss term
            self.before_loss = before
            self.after_loss = after
            self.stop_token_loss = stop_token_loss
            self.regularization_loss = regularization
            self.linear_loss = linear_loss

            loss = before + after + stop_token_loss + regularization + linear_loss
            self.loss = loss

    def add_optimizer(self, global_step, hvd=None):
        """Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
        Args:
            global_step: int32 scalar Tensor representing current global step in training
            hvd: horovod
        """

        hp = self._hparams
        with tf.variable_scope('optimizer'):
            hvd_size = 1.0
            if hvd:
                hvd_size = hvd.size()
            if hp.decay_learning_rate:
                self.decay_steps = hp.decay_steps
                self.decay_rate = hp.decay_rate
                self.learning_rate = self._learning_rate_decay(hp.initial_learning_rate,
                                                               global_step) * hvd_size
            else:
                self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate) * hvd_size

            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1,
                                               hp.adam_beta2, hp.adam_epsilon)
            if hvd:
                optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.none)
            #  Compute Gradient
            update_vars = [v for v in self.all_vars if not (
                    'inputs_embedding' in v.name or 'encoder_' in v.name)] if hp.fine_tuning else None
            gradients, variables = zip(*optimizer.compute_gradients(self.loss, var_list=update_vars))
            self.gradients = gradients
            # Just for caution
            # https://github.com/Rayhane-mamah/Tacotron-2/issues/11
            if hp.clip_gradients:
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)  # __mark 0.5 refer
            else:
                clipped_gradients = gradients
            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step, name='train_op')

    def _learning_rate_decay(self, init_lr, global_step):
        #################################################################
        # Narrow Exponential Decay:

        # Phase 1: lr = 1e-3
        # We only start learning rate decay after 50k steps

        # Phase 2: lr in ]1e-5, 1e-3[
        # decay reach minimal value at step 310k

        # Phase 3: lr = 1e-5
        # clip by minimal learning rate value (step > 310k)
        #################################################################
        hp = self._hparams

        # Compute natural exponential decay
        lr = tf.train.exponential_decay(init_lr,
                                        global_step - hp.start_decay,  # lr = 1e-3 at step 50k
                                        self.decay_steps,
                                        self.decay_rate,  # lr = 1e-5 around step 310k
                                        name='lr_exponential_decay')

        # clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, hp.final_learning_rate), init_lr)

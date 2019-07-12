import os
import platform
import numpy as np
import tensorflow as tf
from datasets import audio
from infolog import log
from models import create_model
from utils import plot
from utils.text_sequence import text_to_sequence
from datasets.audio import inv_linear_spectrogram_tensorflow


class Synthesizer:
    def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron2', freezer=False):
        log('Constructing model: %s' % model_name)
        if freezer:
            try:
                checkpoint_path = tf.train.get_checkpoint_state(checkpoint_path).model_checkpoint_path
            except:
                raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint_path))
        # Force the batch size to be known in order to use attention masking in batch synthesis
        inputs = tf.placeholder(tf.int32, (None, None), name='inputs')
        input_lengths = tf.placeholder(tf.int32, (None,), name='input_lengths')
        targets = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.model = create_model(hparams)
            if gta:
                self.model.initialize(inputs, input_lengths, targets, gta=gta)
            else:
                self.model.initialize(inputs, input_lengths)

            self.mel_outputs = self.model.mel_outputs
            self.linear_outputs = self.model.linear_outputs if (hparams.predict_linear and not gta) else None
            if freezer:
                self.alignments = tf.identity(self.model.alignments, name="alignments")[0]
                self.linear_outputs = inv_linear_spectrogram_tensorflow(self.model.linear_outputs[0], hparams=hparams)
            else:
                self.alignments = tf.identity(self.model.alignments, name="alignments")
            self.stop_token_prediction = self.model.stop_token_prediction
            self.targets = targets

        self.gta = gta
        self._hparams = hparams
        # pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        # explicitely setting the padding to a value that doesn't originally exist in the spectogram
        # to avoid any possible conflicts, without affecting the output range of the model too much
        if hparams.symmetric_mels:
            self._target_pad = -hparams.max_abs_value
        else:
            self._target_pad = 0.

        self.inputs = inputs
        self.input_lengths = input_lengths
        self.targets = targets

        log('Loading checkpoint: %s' % checkpoint_path)
        # Memory allocation on the GPUs as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, texts, basenames, out_dir, log_dir, mel_filenames):
        hparams = self._hparams
        # [-max, max] or [0,max]
        t2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (
            0, hparams.max_abs_value)

        # Repeat last sample until number of samples is dividable by the number of GPUs (last run scenario)
        while len(texts) % hparams.synthesis_batch_size != 0:
            texts.append(texts[-1])
            basenames.append(basenames[-1])
            if mel_filenames is not None:
                mel_filenames.append(mel_filenames[-1])

        seqs = [np.asarray(text_to_sequence(text)) for text in texts]
        input_lengths = [len(seq) for seq in seqs]
        input_seqs, max_seq_len = self._prepare_inputs(seqs)

        feed_dict = {
            self.inputs: input_seqs,
            self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
        }

        if self.gta:
            np_targets = [np.load(mel_filename) for mel_filename in mel_filenames]
            target_lengths = [len(np_target) for np_target in np_targets]
            target_seqs, max_target_len = self._prepare_targets(np_targets, self._hparams.outputs_per_step)
            feed_dict[self.targets] = target_seqs
            assert len(np_targets) == len(texts)
        linears = None
        if self.gta or not hparams.predict_linear:
            mels, alignments, stop_tokens = self.session.run(
                [self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict)

            # Natural batch synthesis
            # Get Mel lengths for the entire batch from stop_tokens predictions
            target_lengths = self._get_output_lengths(stop_tokens)

            # Take off the batch wise padding
            mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
            assert len(mels) == len(texts)

        else:
            linears, mels, alignments, stop_tokens = self.session.run(
                [self.linear_outputs, self.mel_outputs, self.alignments, self.stop_token_prediction],
                feed_dict=feed_dict)

            # Natural batch synthesis
            # Get Mel/Linear lengths for the entire batch from stop_tokens predictions
            target_lengths = self._get_output_lengths(stop_tokens)

            # Take off the batch wise padding
            mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
            linears = [linear[:target_length, :] for linear, target_length in zip(linears, target_lengths)]
            linears = np.clip(linears, t2_output_range[0], t2_output_range[1])
            assert len(mels) == len(linears) == len(texts)

        mels = np.clip(mels, t2_output_range[0], t2_output_range[1])

        if basenames is None:
            # Generate wav and read it
            wav = audio.inv_mel_spectrogram(mels[0].T, hparams)
            audio.save_wav(wav, 'temp.wav', sr=hparams.sample_rate)  # Find a better way

            if platform.system() == 'Linux':
                # Linux wav reader
                os.system('aplay temp.wav')

            elif platform.system() == 'Windows':
                # windows wav reader
                os.system('start /min mplay32 /play /close temp.wav')

            else:
                raise RuntimeError(
                    'Your OS type is not supported yet, please add it to "synthesizer.py, line-165" and feel free to make a Pull Request ;) Thanks!')

            return

        saved_mels_paths = []
        for i, mel in enumerate(mels):
            # Write the spectrogram to disk
            # Note: outputs mel-spectrogram files and target ones have same names, just different folders
            mel_filename = os.path.join(out_dir, 'mel-{}.npy'.format(basenames[i]))
            np.save(mel_filename, mel, allow_pickle=False)
            saved_mels_paths.append(mel_filename)

            if log_dir is not None:
                # save wav (mel -> wav)
                wav = audio.inv_mel_spectrogram(mel.T, hparams)
                audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-mel.wav'.format(basenames[i])),
                               sr=hparams.sample_rate)

                # save alignments
                plot.plot_alignment(alignments[i], os.path.join(log_dir, 'plots/alignment-{}.png'.format(basenames[i])),
                                    title='{}'.format(texts[i]), split_title=True, max_len=target_lengths[i])

                # save mel spectrogram plot
                plot.plot_spectrogram(mel, os.path.join(log_dir, 'plots/mel-{}.png'.format(basenames[i])),
                                      title='{}'.format(texts[i]), split_title=True)

                if linears:
                    # save wav (linear -> wav)
                    wav = audio.inv_linear_spectrogram(linears[i].T, hparams)
                    audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-linear.wav'.format(basenames[i])),
                                   sr=hparams.sample_rate)

                    # save linear spectrogram plot
                    plot.plot_spectrogram(linears[i], os.path.join(log_dir, 'plots/linear-{}.png'.format(basenames[i])),
                                          title='{}'.format(texts[i]), split_title=True, auto_aspect=True)

        return saved_mels_paths

    @staticmethod
    def _round_up(x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    @staticmethod
    def _get_output_lengths(stop_tokens):
        # Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
        output_lengths = [row.index(1) if 1 in row else len(row) for row in np.round(stop_tokens).tolist()]
        return output_lengths

    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

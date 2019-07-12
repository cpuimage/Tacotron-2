import os
import threading
import time

import numpy as np
import tensorflow as tf
from infolog import log
from sklearn.model_selection import train_test_split

_batches_per_group = 32

import msgpack
import msgpack_numpy

msgpack_numpy.patch()


class Feeder:
    """
        Feeds batches of data into queue on a background thread.
    """

    def __init__(self, coordinator, metadata_filename, hparams):
        super(Feeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        self._train_offset = 0
        self._test_offset = 0
        # Load metadata
        self._out_dir = os.path.dirname(metadata_filename)
        with open(metadata_filename, encoding='utf-8') as f:
            self._metadata = []
            for line in f:
                npz_filename, time_steps, mel_frames, text = line.strip().split('|')
                self._metadata.append(
                    [os.path.join(self._out_dir, os.path.basename(npz_filename)), time_steps, mel_frames, text])
            frame_shift_ms = hparams.hop_size / hparams.sample_rate
            hours = sum([int(x[2]) for x in self._metadata]) * frame_shift_ms / 3600
            log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(self._metadata), hours))

        # Train test split
        if hparams.test_size is None:
            assert hparams.test_batches is not None

        test_size = (hparams.test_size if hparams.test_size is not None
                     else hparams.test_batches * hparams.batch_size)
        indices = np.arange(len(self._metadata))
        train_indices, test_indices = train_test_split(indices,
                                                       test_size=test_size,
                                                       random_state=hparams.data_random_state)

        # Make sure test_indices is a multiple of batch_size else round down
        len_test_indices = self._round_down(len(test_indices), hparams.batch_size)
        extra_test = test_indices[len_test_indices:]
        test_indices = test_indices[:len_test_indices]
        train_indices = np.concatenate([train_indices, extra_test])

        self._train_meta = list(np.array(self._metadata)[train_indices])
        self._test_meta = list(np.array(self._metadata)[test_indices])
        self.test_steps = len(self._test_meta) // hparams.batch_size

        if hparams.test_size is None:
            assert hparams.test_batches == self.test_steps

        # pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        # explicitely setting the padding to a value that doesn't originally exist in the spectogram
        # to avoid any possible conflicts, without affecting the output range of the model too much
        if hparams.symmetric_mels:
            self._target_pad = -hparams.max_abs_value
        else:
            self._target_pad = 0.
        # Mark finished sequences with 1s
        self._token_pad = 1.

        with tf.device('/cpu:0'):
            # Create placeholders for inputs and targets. Don't specify batch size because we want
            # to be able to feed different batch sizes at eval time.
            self._placeholders = [
                tf.placeholder(tf.int32, shape=(None, None), name='inputs'),
                tf.placeholder(tf.int32, shape=(None,), name='input_lengths'),
                tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='mel_targets'),
                tf.placeholder(tf.float32, shape=(None, None), name='token_targets'),
                tf.placeholder(tf.float32, shape=(None, None, hparams.num_freq), name='linear_targets'),
                tf.placeholder(tf.int32, shape=(None,), name='targets_lengths'),
            ]

            # Create queue for buffering data
            queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32],
                                 name='input_queue')
            self._enqueue_op = queue.enqueue(self._placeholders)
            self.inputs, self.input_lengths, self.mel_targets, self.token_targets, self.linear_targets, self.targets_lengths = queue.dequeue()

            self.inputs.set_shape(self._placeholders[0].shape)
            self.input_lengths.set_shape(self._placeholders[1].shape)
            self.mel_targets.set_shape(self._placeholders[2].shape)
            self.token_targets.set_shape(self._placeholders[3].shape)
            self.linear_targets.set_shape(self._placeholders[4].shape)
            self.targets_lengths.set_shape(self._placeholders[5].shape)

            # Create eval queue for buffering eval data
            eval_queue = tf.FIFOQueue(1, [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32],
                                      name='eval_queue')
            self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
            self.eval_inputs, self.eval_input_lengths, self.eval_mel_targets, self.eval_token_targets, \
            self.eval_linear_targets, self.eval_targets_lengths = eval_queue.dequeue()

            self.eval_inputs.set_shape(self._placeholders[0].shape)
            self.eval_input_lengths.set_shape(self._placeholders[1].shape)
            self.eval_mel_targets.set_shape(self._placeholders[2].shape)
            self.eval_token_targets.set_shape(self._placeholders[3].shape)
            self.eval_linear_targets.set_shape(self._placeholders[4].shape)
            self.eval_targets_lengths.set_shape(self._placeholders[5].shape)

    def start_threads(self, session):
        self._session = session
        thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
        thread.daemon = True  # Thread will close when parent quits
        thread.start()

        thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
        thread.daemon = True  # Thread will close when parent quits
        thread.start()

    @staticmethod
    def loads_msgpack(path):
        """
        Args: 
            :param path:  the output of `dumps`.
        """
        with open(path, 'rb') as f:
            bin_data = f.read()
        return msgpack.loads(bin_data, raw=False)

    def _get_test_groups(self):
        npz_filename, time_steps, mel_frames, text = self._test_meta[self._test_offset]
        self._test_offset += 1
        npz_data = self.loads_msgpack(npz_filename)
        input_data = npz_data['input_data']
        mel_target = npz_data['mel']
        linear_target = npz_data['linear']
        target_length = npz_data['mel_frames']
        token_target = npz_data['stop_token']
        return input_data, mel_target, token_target, linear_target, target_length

    def make_test_batches(self):
        start = time.time()

        # Read a group of examples
        n = self._hparams.batch_size
        r = self._hparams.outputs_per_step

        # Test on entire test set
        examples = [self._get_test_groups() for _ in range(len(self._test_meta))]

        # Bucket examples based on similar output sequence length for efficiency
        examples.sort(key=lambda x: x[-1])
        batches = [examples[i: i + n] for i in range(0, len(examples), n)]
        np.random.shuffle(batches)

        log('\nGenerated {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
        return batches, r

    def _enqueue_next_train_group(self):
        while not self._coord.should_stop():
            start = time.time()

            # Read a group of examples
            n = self._hparams.batch_size
            r = self._hparams.outputs_per_step
            examples = [self._get_next_example() for _ in range(n * _batches_per_group)]

            # Bucket examples based on similar output sequence length for efficiency
            examples.sort(key=lambda x: x[-1])
            batches = [examples[i: i + n] for i in range(0, len(examples), n)]
            np.random.shuffle(batches)

            log('\nGenerated {} train batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
            for batch in batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
                self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def _enqueue_next_test_group(self):
        # Create test batches once and evaluate on them for all test steps
        test_batches, r = self.make_test_batches()
        while not self._coord.should_stop():
            for batch in test_batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
                self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self):
        """Gets a single example (input, mel_target, token_target, linear_target, mel_length) from_ disk
        """
        if self._train_offset >= len(self._train_meta):
            self._train_offset = 0
            np.random.shuffle(self._train_meta)

        npz_filename, time_steps, mel_frames, text = self._train_meta[self._train_offset]
        self._train_offset += 1

        npz_data = self.loads_msgpack(npz_filename)
        input_data = npz_data['input_data']
        mel_target = npz_data['mel']
        linear_target = npz_data['linear']
        target_length = npz_data['mel_frames']
        token_target = npz_data['stop_token']
        return input_data, mel_target, token_target, linear_target, target_length

    def _prepare_batch(self, batches, outputs_per_step):
        np.random.shuffle(batches)
        targets_lengths = np.asarray([x[-1] for x in batches], dtype=np.int32)  # Used to mask loss
        input_lengths = np.asarray([len(x[0]) for x in batches], dtype=np.int32)
        inputs, input_max_len = self._prepare_inputs([x[0] for x in batches])
        mel_targets, mel_target_max_len = self._prepare_targets([x[1] for x in batches], outputs_per_step)
        token_targets, token_target_max_len = self._prepare_token_targets([x[2] for x in batches], outputs_per_step)
        linear_targets, linear_target_max_len = self._prepare_targets([x[3] for x in batches], outputs_per_step)
        return inputs, input_lengths, mel_targets, token_targets, linear_targets, targets_lengths

    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

    def _prepare_token_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_token_target(t, data_len) for t in targets]), data_len

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

    def _pad_token_target(self, t, length):
        return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=self._token_pad)

    @staticmethod
    def _round_up(x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    @staticmethod
    def _round_down(x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder

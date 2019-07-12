import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from datasets import audio
from utils.text_sequence import text_to_sequence
import msgpack
import msgpack_numpy

msgpack_numpy.patch()


def dumps_msgpack(obj, path):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object.
    """
    bin_data = msgpack.dumps(obj, use_bin_type=True)

    if path:
        with open(path, 'wb') as f:
            f.write(bin_data)
    else:
        return bin_data


def build_from_path(hparams, input_dir, out_dir, n_jobs=12, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dir: input directory that contains the files to prerocess
        - out_dir: output directory of the preprocessed speech dataset
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    if isinstance(input_dir, str):
        sub_dirs = []
        if not os.path.exists(os.path.join(input_dir, 'transcript.txt')):
            sub_names = os.listdir(input_dir)
            for name in sub_names:
                sub_dirs.append(os.path.join(input_dir, name))
        else:
            sub_dirs = [input_dir]
    else:
        sub_dirs = input_dir
    for sub_dir in sub_dirs:
        with open(os.path.join(sub_dir, 'transcript.txt'), encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) < hparams.batch_size:
                continue
            for line in lines:
                wav_path, seq_text = line.strip().split('|')
                wav_path = os.path.join(sub_dir, wav_path)
                if not os.path.exists(wav_path):
                    continue
                futures.append(executor.submit(
                    partial(_process_utterance, out_dir, index, wav_path, seq_text, hparams)))
                index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(out_dir, index, wav_path, text, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - out_dir: the directory to write the msgpack into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - text: text spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
    """
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError:  # catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    # Trim lead/trail silences
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)

    # Pre-emphasize
    preem_wav = audio.preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

    # rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
        preem_wav = preem_wav / np.abs(preem_wav).max() * hparams.rescaling_max

        # Assert all audio is in [-1, 1]
        if (wav > 1.).any() or (wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))
        if (preem_wav > 1.).any() or (preem_wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))

    # [-1, 1]
    out = wav
    constant_values = 0.
    out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(preem_wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Compute the linear scale spectrogram from the wav
    linear_spectrogram = audio.linearspectrogram(preem_wav, hparams).astype(np.float32)
    linear_frames = linear_spectrogram.shape[1]

    # sanity check
    assert linear_frames == mel_frames

    # Ensure time resolution adjustement between audio and mel-spectrogram
    l_pad, r_pad = audio.librosa_pad_lr(wav, audio.get_hop_size(hparams), hparams.pad_sides)

    # Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
    out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0
    time_steps = len(out)
    npz_filename = '{}.npz'.format(index)
    r = hparams.outputs_per_step
    if hparams.symmetric_mels:
        _pad_value = -hparams.max_abs_value
    else:
        _pad_value = 0.
    # +2r for head and tail silence
    mel_spec = np.pad(mel_spectrogram.T, [[r, r], [0, 0]], 'constant', constant_values=_pad_value)
    linear_spec = np.pad(linear_spectrogram.T, [[r, r], [0, 0]], 'constant', constant_values=_pad_value)
    target_length = len(linear_spec)
    target_frames = (target_length // r + 1) * r
    num_pad = target_frames - target_length
    if num_pad != 0:
        linear_spec = np.pad(linear_spec, ((0, num_pad), (0, 0)), "constant", constant_values=_pad_value)
        mel_spec = np.pad(mel_spec, ((0, num_pad), (0, 0)), "constant", constant_values=_pad_value)
    stop_token = np.concatenate(
        [np.zeros(target_frames - 1, dtype=np.float32), np.ones(1, dtype=np.float32)],
        axis=0)
    data = {
        'mel': mel_spec,
        'linear': linear_spec,
        'audio': out.astype(out_dtype),
        'input_data': np.asarray(text_to_sequence(text)),
        'time_steps': time_steps,
        'mel_frames': target_frames,
        'text': text,
        'stop_token': stop_token,
    }
    dumps_msgpack(data, os.path.join(out_dir, npz_filename))
    # Return a tuple describing this training example
    return npz_filename, time_steps, mel_frames, text

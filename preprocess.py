import argparse
import os
from multiprocessing import cpu_count
from datasets import preprocessor
from hparams import hparams
from tqdm import tqdm


def preprocess(args, input_folder, out_dir, hparams):
    os.makedirs(out_dir, exist_ok=True)
    metadata = preprocessor.build_from_path(hparams, input_folder, out_dir, args.n_jobs, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        sum_mel_frames = 0.0
        sum_time_steps = 0.0
        max_text = 0.0
        max_mel_frames = 0.0
        max_time_steps = 0.0
        for npz_filename, time_steps, mel_frames, text in metadata:
            len_text = len(text)
            lines = [npz_filename, time_steps, mel_frames, text]
            f.write('|'.join([str(x) for x in lines]) + '\n')
            sum_mel_frames += mel_frames
            sum_time_steps += time_steps
            max_text = max(max_text, len_text)
            max_time_steps = max(max_time_steps, time_steps)
            max_mel_frames = max(max_mel_frames, mel_frames)
    sr = hparams.sample_rate
    hours = sum_time_steps / sr / 3600
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
        len(metadata), sum_mel_frames, sum_time_steps, hours))
    print('Max input length (text chars): {}'.format(int(max_text)))
    print('Max mel frames length: {}'.format(int(max_mel_frames)))
    print('Max audio timesteps length: {}'.format(max_time_steps))


def run_preprocess(args, hparams):
    input_folder = os.path.join(args.base_dir, args.dataset)
    output_folder = os.path.join(args.base_dir, args.output)
    preprocess(args, input_folder, output_folder, hparams)


def main():
    print('initializing preprocessing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--dataset', type=str, default='./train_datasets/')
    parser.add_argument('--output', type=str, default='./train_data/')
    parser.add_argument('--n_jobs', type=int, default=cpu_count())
    args = parser.parse_args()

    modified_hp = hparams.parse(args.hparams)

    run_preprocess(args, modified_hp)


if __name__ == '__main__':
    main()

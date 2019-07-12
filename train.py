from hparams import hparams
import argparse
import os
import time
import traceback
from datetime import datetime
import infolog
import numpy as np
import tensorflow as tf
from datasets import audio
from hparams import hparams_debug_string
from feeder import Feeder
from models import create_model
from utils import ValueWindow, plot
from utils.text_sequence import sequence_to_text
from utils.text_sequence import symbols
from tqdm import tqdm

log = infolog.log


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def add_embedding_stats(summary_writer, embedding_names, paths_to_meta, checkpoint_path):
    # Create tensorboard projector
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    config.model_checkpoint_path = checkpoint_path

    for embedding_name, path_to_meta in zip(embedding_names, paths_to_meta):
        # Initialize config
        embedding = config.embeddings.add()
        # Specifiy the embedding variable and the metadata
        embedding.tensor_name = embedding_name
        embedding.metadata_path = path_to_meta

    # Project the embeddings to space dimensions for visualization
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)


def add_train_stats(model, hparams):
    with tf.variable_scope('stats'):
        tf.summary.histogram('mel_outputs', model.mel_outputs)
        tf.summary.histogram('mel_targets', model.mel_targets)
        tf.summary.scalar('before_loss', model.before_loss)
        tf.summary.scalar('after_loss', model.after_loss)
        if hparams.predict_linear:
            tf.summary.scalar('linear_loss', model.linear_loss)
            tf.summary.histogram('linear_outputs', model.linear_outputs)
            tf.summary.histogram('linear_targets', model.linear_targets)
        tf.summary.scalar('regularization_loss', model.regularization_loss)
        tf.summary.scalar('stop_token_loss', model.stop_token_loss)
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('learning_rate', model.learning_rate)  # Control learning rate decay speed
        if hparams.teacher_forcing_mode == 'scheduled':
            tf.summary.scalar('teacher_forcing_ratio',
                              model.ratio)  # Control teacher forcing ratio decay when mode = 'scheduled'
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram('gradient_norm', gradient_norms)
        tf.summary.scalar('max_gradient_norm',
                          tf.reduce_max(gradient_norms))  # visualize gradients (in case of explosion)
        return tf.summary.merge_all()


def add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, stop_token_loss, loss):
    values = [
        tf.Summary.Value(tag='eval_model/eval_stats/eval_before_loss', simple_value=before_loss),
        tf.Summary.Value(tag='eval_model/eval_stats/eval_after_loss', simple_value=after_loss),
        tf.Summary.Value(tag='eval_model/eval_stats/stop_token_loss', simple_value=stop_token_loss),
        tf.Summary.Value(tag='eval_model/eval_stats/eval_loss', simple_value=loss),
    ]
    if linear_loss is not None:
        values.append(tf.Summary.Value(tag='eval_model/eval_stats/eval_linear_loss', simple_value=linear_loss))
    test_summary = tf.Summary(value=values)
    summary_writer.add_summary(test_summary, step)


def model_train_mode(feeder, hparams, global_step, hvd=None):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        model = create_model(hparams)
        if hparams.predict_linear:
            model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets,
                             linear_targets=feeder.linear_targets,
                             targets_lengths=feeder.targets_lengths, global_step=global_step,
                             is_training=True)
        else:
            model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets,
                             targets_lengths=feeder.targets_lengths, global_step=global_step,
                             is_training=True)
        model.add_loss()
        model.add_optimizer(global_step, hvd)
        stats = add_train_stats(model, hparams)
        return model, stats


def model_test_mode(feeder, hparams, global_step):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        model = create_model(hparams)
        if hparams.predict_linear:
            model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets,
                             feeder.eval_token_targets,
                             linear_targets=feeder.eval_linear_targets, targets_lengths=feeder.eval_targets_lengths,
                             global_step=global_step,
                             is_training=False, is_evaluating=True)
        else:
            model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets,
                             feeder.eval_token_targets,
                             targets_lengths=feeder.eval_targets_lengths, global_step=global_step, is_training=False,
                             is_evaluating=True)
        model.add_loss()
        return model


def train(log_dir, args, hparams, use_hvd=False):
    if use_hvd:
        import horovod.tensorflow as hvd
        # Initialize Horovod.
        hvd.init()
    else:
        hvd = None
    save_dir = os.path.join(log_dir, 'pretrained')
    plot_dir = os.path.join(log_dir, 'plots')
    wav_dir = os.path.join(log_dir, 'wavs')
    mel_dir = os.path.join(log_dir, 'mel-spectrograms')
    eval_dir = os.path.join(log_dir, 'eval-dir')
    eval_plot_dir = os.path.join(eval_dir, 'plots')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    tensorboard_dir = os.path.join(log_dir, 'events')
    meta_folder = os.path.join(log_dir, 'metas')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(meta_folder, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
    input_path = os.path.join(args.base_dir, args.train_input)
    linear_dir = os.path.join(log_dir, 'linear-spectrograms')
    os.makedirs(linear_dir, exist_ok=True)

    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Loading training data from: {}'.format(input_path))
    log('Using model: {}'.format(args.model))
    log(hparams_debug_string())

    # Start by setting a seed for repeatability
    tf.set_random_seed(hparams.random_seed)

    # Set up data feeder
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder'):
        feeder = Feeder(coord, input_path, hparams)

    # Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    model, stats = model_train_mode(feeder, hparams, global_step, hvd=hvd)
    eval_model = model_test_mode(feeder, hparams, global_step)

    # Embeddings metadata
    char_embedding_meta = os.path.join(meta_folder, 'CharacterEmbeddings.tsv')
    if not os.path.isfile(char_embedding_meta):
        with open(char_embedding_meta, 'w', encoding='utf-8') as f:
            for symbol in symbols:
                if symbol == ' ':
                    symbol = '\\s'  # For visual purposes, swap space with \s

                f.write('{}\n'.format(symbol))

    char_embedding_meta = char_embedding_meta.replace(log_dir, '..')
    # Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=20)

    log('Tacotron2 training set to a maximum of {} steps'.format(args.train_steps))

    # Memory allocation on the GPU as needed
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    if use_hvd:
        config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Train
    with tf.Session(config=config) as sess:
        try:
            sess.run(tf.global_variables_initializer())

            # saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, default = True
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)

                    if checkpoint_state and checkpoint_state.model_checkpoint_path:
                        log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
                        saver.restore(sess, checkpoint_state.model_checkpoint_path)

                    else:
                        log('No model to load at {}'.format(save_dir), slack=True)
                        saver.save(sess, checkpoint_path, global_step=global_step)

                except tf.errors.OutOfRangeError as e:
                    log('Cannot restore checkpoint: {}'.format(e), slack=True)
            else:
                log('Starting new training!', slack=True)
                saver.save(sess, checkpoint_path, global_step=global_step)

            # initializing feeder
            feeder.start_threads(sess)
            # Horovod bcast vars across workers
            if use_hvd:
                # Horovod: broadcast initial variable states from rank 0 to all other processes.
                # This is necessary to ensure consistent initialization of all workers when
                # training is started with random weights or restored from a checkpoint.
                bcast = hvd.broadcast_global_variables(0)
                bcast.run()
                log('Worker{}: Initialized'.format(hvd.rank()))
            # Training loop
            summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
            # Training loop
            while not coord.should_stop() and step < args.train_steps:
                start_time = time.time()
                step, loss, opt = sess.run([global_step, model.loss, model.train_op])
                main_process = True
                if use_hvd:
                    main_process = hvd.rank() == 0
                if main_process:
                    time_window.append(time.time() - start_time)
                    loss_window.append(loss)
                    message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
                        step, time_window.average, loss, loss_window.average)
                    log(message, end='\r', slack=(step % args.checkpoint_interval == 0))

                    if np.isnan(loss) or loss > 100.:
                        log('Loss exploded to {:.5f} at step {}'.format(loss, step))
                        raise Exception('Loss exploded')

                    if step % args.summary_interval == 0:
                        log('\nWriting summary at step {}'.format(step))
                        summary_writer.add_summary(sess.run(stats), step)

                    if step % args.eval_interval == 0:
                        # Run eval and save eval stats
                        log('\nRunning evaluation at step {}'.format(step))

                        eval_losses = []
                        before_losses = []
                        after_losses = []
                        stop_token_losses = []
                        linear_losses = []
                        linear_loss = None
                        if hparams.predict_linear:
                            for _ in tqdm(range(feeder.test_steps)):
                                eloss, before_loss, after_loss, stop_token_loss, linear_loss, mel_p, mel_t, t_len, align, lin_p, lin_t = sess.run(
                                    [
                                        eval_model.loss, eval_model.before_loss,
                                        eval_model.after_loss,
                                        eval_model.stop_token_loss, eval_model.linear_loss,
                                        eval_model.mel_outputs[0],
                                        eval_model.mel_targets[0], eval_model.targets_lengths[0],
                                        eval_model.alignments[0], eval_model.linear_outputs[0],
                                        eval_model.linear_targets[0],
                                    ])
                                eval_losses.append(eloss)
                                before_losses.append(before_loss)
                                after_losses.append(after_loss)
                                stop_token_losses.append(stop_token_loss)
                                linear_losses.append(linear_loss)
                            linear_loss = sum(linear_losses) / len(linear_losses)

                            wav = audio.inv_linear_spectrogram(lin_p.T, hparams)
                            audio.save_wav(wav,
                                           os.path.join(eval_wav_dir, 'step-{}-eval-wave-from-linear.wav'.format(step)),
                                           sr=hparams.sample_rate)

                        else:
                            for i in tqdm(range(feeder.test_steps)):
                                eloss, before_loss, after_loss, stop_token_loss, mel_p, mel_t, t_len, align = sess.run([
                                    eval_model.loss, eval_model.before_loss,
                                    eval_model.after_loss,
                                    eval_model.stop_token_loss, eval_model.mel_outputs[0],
                                    eval_model.mel_targets[0],
                                    eval_model.targets_lengths[0], eval_model.alignments[0]
                                ])
                                eval_losses.append(eloss)
                                before_losses.append(before_loss)
                                after_losses.append(after_loss)
                                stop_token_losses.append(stop_token_loss)

                        eval_loss = sum(eval_losses) / len(eval_losses)
                        before_loss = sum(before_losses) / len(before_losses)
                        after_loss = sum(after_losses) / len(after_losses)
                        stop_token_loss = sum(stop_token_losses) / len(stop_token_losses)

                        log('Saving eval log to {}..'.format(eval_dir))
                        # Save some log to monitor model improvement on same unseen sequence
                        wav = audio.inv_mel_spectrogram(mel_p.T, hparams)
                        audio.save_wav(wav, os.path.join(eval_wav_dir, 'step-{}-eval-wave-from-mel.wav'.format(step)),
                                       sr=hparams.sample_rate)

                        plot.plot_alignment(align, os.path.join(eval_plot_dir, 'step-{}-eval-align.png'.format(step)),
                                            title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step,
                                                                                        eval_loss),
                                            max_len=t_len // hparams.outputs_per_step)
                        plot.plot_spectrogram(mel_p,
                                              os.path.join(eval_plot_dir,
                                                           'step-{}-eval-mel-spectrogram.png'.format(step)),
                                              title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(),
                                                                                          step,
                                                                                          eval_loss),
                                              target_spectrogram=mel_t,
                                              max_len=t_len)

                        if hparams.predict_linear:
                            plot.plot_spectrogram(lin_p, os.path.join(eval_plot_dir,
                                                                      'step-{}-eval-linear-spectrogram.png'.format(
                                                                          step)),
                                                  title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(),
                                                                                              step, eval_loss),
                                                  target_spectrogram=lin_t,
                                                  max_len=t_len, auto_aspect=True)

                        log('Eval loss for global step {}: {:.3f}'.format(step, eval_loss))
                        log('Writing eval summary!')
                        add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, stop_token_loss,
                                       eval_loss)

                    if step % args.checkpoint_interval == 0 or step == args.train_steps or step == 300:
                        # Save model and current global step
                        saver.save(sess, checkpoint_path, global_step=global_step)

                        log('\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..')
                        if hparams.predict_linear:
                            input_seq, mel_prediction, linear_prediction, alignment, target, target_length, linear_target = sess.run(
                                [
                                    model.inputs[0],
                                    model.mel_outputs[0],
                                    model.linear_outputs[0],
                                    model.alignments[0],
                                    model.mel_targets[0],
                                    model.targets_lengths[0],
                                    model.linear_targets[0],
                                ])

                            # save predicted linear spectrogram to disk (debug)
                            linear_filename = 'linear-prediction-step-{}.npy'.format(step)
                            np.save(os.path.join(linear_dir, linear_filename), linear_prediction.T, allow_pickle=False)

                            # save griffin lim inverted wav for debug (linear -> wav)
                            wav = audio.inv_linear_spectrogram(linear_prediction.T, hparams)
                            audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-linear.wav'.format(step)),
                                           sr=hparams.sample_rate)

                            # Save real and predicted linear-spectrogram plot to disk (control purposes)
                            plot.plot_spectrogram(linear_prediction,
                                                  os.path.join(plot_dir, 'step-{}-linear-spectrogram.png'.format(step)),
                                                  title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(),
                                                                                              step, loss),
                                                  target_spectrogram=linear_target,
                                                  max_len=target_length, auto_aspect=True)

                        else:
                            input_seq, mel_prediction, alignment, target, target_length = sess.run([
                                model.inputs[0],
                                model.mel_outputs[0],
                                model.alignments[0],
                                model.mel_targets[0],
                                model.targets_lengths[0],
                            ])

                        # save predicted mel spectrogram to disk (debug)
                        mel_filename = 'mel-prediction-step-{}.npy'.format(step)
                        np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T, allow_pickle=False)

                        # save griffin lim inverted wav for debug (mel -> wav)
                        wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
                        audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-mel.wav'.format(step)),
                                       sr=hparams.sample_rate)

                        # save alignment plot to disk (control purposes)
                        plot.plot_alignment(alignment, os.path.join(plot_dir, 'step-{}-align.png'.format(step)),
                                            title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step,
                                                                                        loss),
                                            max_len=target_length // hparams.outputs_per_step)
                        # save real and predicted mel-spectrogram plot to disk (control purposes)
                        plot.plot_spectrogram(mel_prediction,
                                              os.path.join(plot_dir, 'step-{}-mel-spectrogram.png'.format(step)),
                                              title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(),
                                                                                          step,
                                                                                          loss),
                                              target_spectrogram=target,
                                              max_len=target_length)
                        log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))

                    if step % args.embedding_interval == 0 or step == args.train_steps or step == 1:
                        # Get current checkpoint state
                        checkpoint_state = tf.train.get_checkpoint_state(save_dir)

                        # Update Projector
                        log('\nSaving Model Character Embeddings visualization..')
                        add_embedding_stats(summary_writer, [model.embedding_table.name], [char_embedding_meta],
                                            checkpoint_state.model_checkpoint_path)
                        log('Tacotron2 Character embeddings have been updated on tensorboard!')

            log('Tacotron2 training complete after {} global steps!'.format(args.train_steps), slack=True)
            return save_dir

        except Exception as e:
            log('Exiting due to exception: {}'.format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name, args.slack_url)
    return log_dir, modified_hp


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--train_input', default='../train_data/train.txt')
    parser.add_argument('--name', help='Name of logging directory.')
    parser.add_argument('--model', default='Tacotron2')
    parser.add_argument('--input_dir', default='../train_data/', help='folder to contain inputs sentences/targets')
    parser.add_argument('--output_dir', default='output', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=250,
                        help='Steps between running summary ops')
    parser.add_argument('--embedding_interval', type=int, default=5000,
                        help='Steps between updating embeddings projection visualization')
    parser.add_argument('--checkpoint_interval', type=int, default=2500,
                        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5000,
                        help='Steps between eval on test data')
    parser.add_argument('--train_steps', type=int, default=100000,
                        help='total number of tacotron2 training steps')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--slack_url', default=None, help='slack webhook notification destination link')
    parser.add_argument('--gpu_devices', default='0', help='Set the gpu the model should run on.(eg:hvd,0,1,2...)')
    args = parser.parse_args(argv[1:])
    use_hvd = args.gpu_devices == 'hvd'
    if not use_hvd:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    log_dir, hparams = prepare_run(args)
    train(log_dir, args, hparams, use_hvd=use_hvd)


if __name__ == '__main__':
    tf.app.run(main=main)

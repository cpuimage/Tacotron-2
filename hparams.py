import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
    # Audio
    # Audio parameters are the most important parameters to tune when using this work on your personal data. Below are the beginner steps to adapt
    # this work to your personal data:
    #	1- Determine my data sample rate: First you need to determine your audio sample_rate (how many samples are in a second of audio). This can be done using sox: "sox --i <filename>"
    #		(For this small tuto, I will consider 24kHz (24000 Hz), and defaults are 22050Hz, so there are plenty of examples to refer to)
    #	2- set sample_rate parameter to your data correct sample rate
    #	3- Fix win_size and and hop_size accordingly: (Supposing you will follow our advice: 50ms window_size, and 12.5ms frame_shift(hop_size))
    #		a- win_size = 0.05 * sample_rate. In the tuto example, 0.05 * 24000 = 1200
    #		b- hop_size = 0.25 * win_size. Also equal to 0.0125 * sample_rate. In the tuto example, 0.25 * 1200 = 0.0125 * 24000 = 300 (Can set frame_shift_ms=12.5 instead)
    #	4- Fix n_fft, num_freq and upsample_scales parameters accordingly.
    #		a- n_fft can be either equal to win_size or the first power of 2 that comes after win_size. I usually recommend using the latter
    #			to be more consistent with signal processing friends. No big difference to be seen however. For the tuto example: n_fft = 2048 = 2**11
    #		b- num_freq = (n_fft / 2) + 1. For the tuto example: num_freq = 2048 / 2 + 1 = 1024 + 1 = 1025.
    #		c- For WaveNet, upsample_scales products must be equal to hop_size. For the tuto example: upsample_scales=[15, 20] where 15 * 20 = 300
    #			it is also possible to use upsample_scales=[3, 4, 5, 5] instead. One must only keep in mind that upsample_kernel_size[0] = 2*upsample_scales[0]
    #			so the training segments should be long enough (2.8~3x upsample_scales[0] * hop_size or longer) so that the first kernel size can see the middle
    #			of the samples efficiently. The length of WaveNet training segments is under the parameter "max_time_steps".
    #	5- Finally comes the silence trimming. This very much data dependent, so I suggest trying preprocessing (or part of it, ctrl-C to stop), then use the
    #		.ipynb provided in the repo to listen to some inverted mel/linear spectrograms. That will first give you some idea about your above parameters, and
    #		it will also give you an idea about trimming. If silences persist, try reducing trim_top_db slowly. If samples are trimmed mid words, try increasing it.
    #	6- If audio quality is too metallic or fragmented (or if linear spectrogram plots are showing black silent regions on top), then restart from step 2.
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
    num_freq=513,  # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.999,  # Rescaling value

    # train samples of lengths between 3sec and 14sec are more than enough to make a model capable of generating consistent speech.
    clip_mels_length=True,
    # For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
    max_mel_frames=1000,
    # Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.

    # Mel spectrogram
    n_fft=1024,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
    magnitude_power=2.,  # The power of the spectrogram magnitude (1. for energy, 2. for power)

    # M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
    trim_silence=True,  # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    trim_fft_size=512,  # Trimming window size
    trim_hop_size=128,  # Trimmin hop length
    trim_top_db=23,  # Trimming db difference from reference db (smaller==harder trim.)

    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion,
    # not too small for fast convergence)
    pad_sides=1,  # Can be 1 or 2. 1 for pad right only, 2 for both sides padding.

    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.

    # Limits
    min_level_db=-100,
    ref_level_db=20,

    # Griffin Lim
    power=1.5,  # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    griffin_lim_iters=60,  # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.

    # Tacotron
    # Model general type
    outputs_per_step=3,
    # number of frames to generate at each decoding step (increase to speed up computation and allows for higher batch size, decreases G&L audio quality)
    stop_at_any=True,
    # Determines whether the decoder should stop when predicting <stop> to any frame or to all of them (True works pretty well)
    clip_outputs=True,
    # Whether to clip spectrograms to T2_output_range (even in loss computation). ie: Don't penalize model for exceeding output range and bring back to borders.
    lower_bound_decay=0.1,
    # Small regularizer for noise synthesis by adding small range of penalty for silence regions. Set to 0 to clip in Tacotron2 range.

    # Input parameters
    embedding_dim=256,  # dimension of embedding space

    # Encoder parameters
    enc_conv_num_layers=3,  # number of encoder convolutional layers
    enc_conv_kernel_size=(5,),  # size of encoder convolution filters for each layer
    enc_conv_channels=128,  # number of encoder convolutions filters for each layer
    encoder_lstm_units=128,  # number of lstm units for each direction (forward and backward)

    # Attention mechanism
    attention_dim=128,  # dimension of attention space

    # Decoder
    prenet_layers=[256, 128],  # number of layers and number of units of prenet [256, 256]
    decoder_layers=2,  # number of decoder lstm layers
    decoder_lstm_units=512,  # number of decoder lstm units on each layer
    max_iters=2000,  # Max decoder steps during inference (Just for safety from infinite loop cases)

    # Residual postnet
    postnet_num_layers=5,  # number of postnet convolutional layers
    postnet_kernel_size=(5,),  # size of postnet convolution filters for each layer
    postnet_channels=128,  # number of postnet convolution filters for each layer

    # CBHG mel->linear postnet
    post_net_cbhg_out_units=128,
    post_net_conv_channels=128,
    post_net_max_filter_width=8,
    post_net_projection1_out_channels=256,
    post_net_projection2_out_channels=80,
    post_net_num_highway=4,
    # Loss params
    mask_encoder=True,
    # whether to mask encoder padding while computing attention. Set to True for better prosody but slower convergence.
    mask_decoder=False,
    # Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)
    cross_entropy_pos_weight=20,
    # Use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1 = disabled)
    predict_linear=True,
    # Whether to add a post-processing network to the Tacotron2 to predict linear spectrograms (True mode Not tested!!)
    ###########################################################################################################################################

    # Tacotron2 Training
    # Reproduction seeds
    random_seed=5339,
    # Determines initial graph and operations (i.e: model) random state for reproducibility
    data_random_state=1234,  # random state for train test split repeatability

    # performance parameters
    swap_with_cpu=False,
    # Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

    # train/test split ratios, mini-batches sizes
    batch_size=32,  # number of training samples on each training steps
    # Tacotron2 Batch synthesis supports ~16x the training batch size (no gradients during testing).
    # Training Tacotron2 with unmasked paddings makes it aware of them, which makes synthesis times different from training. We thus recommend masking the encoder.
    synthesis_batch_size=1,
    # DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN'T TRAIN TACOTRON WITH "mask_encoder=True"!!
    test_size=0.05,
    # % of data to keep as test data, if None, test_batches must be not None. (5% is enough to have a good idea about overfit)
    test_batches=None,  # number of test batches.

    # Learning rate schedule
    decay_learning_rate=True,  # boolean, determines if the learning rate will follow an exponential decay
    start_decay=50000,  # Step at which learning decay starts
    decay_steps=50000,  # Determines the learning rate decay slope (UNDER TEST)
    decay_rate=0.5,  # learning rate decay rate (UNDER TEST)
    initial_learning_rate=1e-3,  # starting learning rate
    final_learning_rate=1e-5,  # minimal learning rate

    # Optimization parameters
    adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    adam_epsilon=1e-6,  # AdamOptimizer Epsilon parameter

    # Regularization parameters
    reg_weight=1e-7,  # regularization weight (for L2 regularization)
    scale_regularization=False,
    # Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)
    zoneout_rate=0.1,  # zoneout rate for all LSTM cells in the network
    dropout_rate=0.5,  # dropout rate for all convolutional layers + prenet
    clip_gradients=True,  # whether to clip gradients

    # Evaluation parameters
    natural_eval=False,
    # Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same teacher-forcing ratio as in training (just for overfit)

    # Decoder RNN learning can take be done in one of two ways:
    #	Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
    #	Scheduled Sampling Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
    # The second approach is inspired by:
    # Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
    # Can be found under: https://arxiv.org/pdf/1506.03099.pdf
    teacher_forcing_mode='constant',
    # Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
    teacher_forcing_ratio=1.,
    # Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
    teacher_forcing_init_ratio=1.,  # initial teacher forcing ratio. Relevant if mode='scheduled'
    teacher_forcing_final_ratio=0.,
    # final teacher forcing ratio. (Set None to use alpha instead) Relevant if mode='scheduled'
    teacher_forcing_start_decay=10000,
    # starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
    teacher_forcing_decay_steps=280000,
    # Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
    teacher_forcing_decay_alpha=None,
    # teacher forcing ratio decay rate. Defines the final tfr as a ratio of initial tfr. Relevant if mode='scheduled'

    # Speaker adaptation parameters
    fine_tuning=False,
    # Set to True to freeze encoder and only keep training pretrained decoder. Used for speaker adaptation with small data.
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
    return 'Hyperparameters:\n' + '\n'.join(hp)

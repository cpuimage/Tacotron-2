# Tacotron-2
Tensorflow implementation of DeepMind's Tacotron-2 (without wavenet). A deep neural network architecture described in this paper: [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)


# Repository Structure:
	Tacotron-2
	├── datasets
	├── logs-Tacotron2	(2)
	│   ├── eval-dir
	│   │ 	├── plots
	│   │	└── wavs
	│   ├── mel-spectrograms
	│   ├── plots
	│   ├── pretrained
	│   ├── metas
	│   └── wavs
	├── papers
	├─|
	│ ├── models
	│ └── utils
	├── synth_output	(3)
	│   ├── eval
	│   ├── gta
	│   ├── logs-eval
	│   │   ├── plots
	│   │   └── wavs
	│   └── natural
	├── train_data	(1)
	│   ├── 1.npy
	│   ├── 2.npy
	│   ├── train.txt


The previous tree shows the current state of the repository (separate training, one step at a time).

- Step **(0)**: Get your dataset, here I have set the examples of **Ljspeech**, **en_US** and **en_UK** (from **M-AILABS**).
- Step **(1)**: Preprocess your data. This will give you the **train_data** folder.
- Step **(2)**: Train your Tacotron model. Yields the **logs-Tacotron2** folder.
- Step **(3)**: Synthesize/Evaluate the Tacotron model. Gives the **Synth_output** folder.
 
# Model Architecture:
<p align="center">
  <img src="https://preview.ibb.co/bU8sLS/Tacotron_2_Architecture.png"/>
</p>
 
# How to start
- **Machine Setup:**

First, you need to have python 3 installed along with [Tensorflow](https://www.tensorflow.org/install/).

Next, you need to install some Linux dependencies to ensure audio libraries work properly:

> apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools

Finally, you can install the requirements. If you are an Anaconda user: (else replace **pip** with **pip3** and **python** with **python3**)

> pip install -r requirements.txt

# Hparams setting:
Before proceeding, you must pick the hyperparameters that suit best your needs. While it is possible to change the hyper parameters from command line during preprocessing/training.


# Preprocessing
Before running the following steps, please make sure you are inside **Tacotron-2 folder**

> cd Tacotron-2

Preprocessing can then be started using: 

> python preprocess.py 

# Training:
To **train both models** sequentially (one after the other):

> python train.py 

checkpoints will be made each **5000 steps** and stored under **logs-Tacotron2 folder.**
  
# Synthesis
To **synthesize audio** in an **End-to-End** (text to audio):

> python synthesize.py
 

# References and Resources:
- [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
- [nii-yamagishilab/tacotron2](https://github.com/nii-yamagishilab/tacotron2)
- [Robust Sequence-to-Sequence Acoustic Modeling with Stepwise Monotonic Attention for Neural TTS](https://arxiv.org/pdf/1906.00672.pdf)
- [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)
- [Original tacotron paper](https://arxiv.org/pdf/1703.10135.pdf)
- [Attention-Based Models for Speech Recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Wavenet: A generative model for raw audio](https://arxiv.org/pdf/1609.03499.pdf)
- [Fast Wavenet](https://arxiv.org/pdf/1611.09482.pdf)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- [keithito/tacotron](https://github.com/keithito/tacotron)

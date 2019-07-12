from .tacotron2 import Tacotron2


def create_model(hparams):
    return Tacotron2(hparams)

from types import NoneType
from autocorrect import Speller
from keras.engine.functional import Functional
import config as cfg


class SpeechAnalyzer:
  def __init__(self, model: Functional, stft: cfg.FourierTransformParams,
               decoder: any = None, speller: (Speller|NoneType) = None):
    pass


  @staticmethod
  def from_config(config: cfg.Config):
    raise NotImplementedError


  def from_file(self, audio_path: str) -> str:
    raise NotImplementedError
  

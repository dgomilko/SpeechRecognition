from types import NoneType
from keras.models import load_model
from autocorrect import Speller
from keras.engine.functional import Functional
import config as cfg
import utils


class SpeechAnalyzer:
  def __init__(self, model: Functional, stft: cfg.FourierTransformParams,
               decoder: utils.Decoder = None, speller: (Speller|NoneType) = None):
    self.__model = model
    self.__stft = stft
    self.__decoder = decoder if decoder else utils.make_batch_decoder()
    self.__speller = speller


  @staticmethod
  def from_config(config: cfg.Config):
    print('Loading model...')
    return SpeechAnalyzer(
      model=load_model(config.model, custom_objects=dict(ctc_loss=utils.ctc_loss)),
      stft=config.stft,
      decoder=utils.make_batch_decoder(),
      speller=Speller(config.speller) if config.speller else None
    )


  def from_file(self, audio_path: str) -> str:
    spectrogram = utils.read_as_spectrogram(audio_path, **self.__stft.__dict__)
    predicted = self.__model.predict(spectrogram)
    decoded = ''.join(self.__decoder(predicted))
    return self.__speller(decoded) if callable(self.__speller) else decoded
  

  def __call__(self, audio_path: str) -> str:
    return self.from_file(audio_path)

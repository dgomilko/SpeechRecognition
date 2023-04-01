from typing import Callable
import numpy as np
import tensorflow as tf
from keras.layers import StringLookup
from keras.backend import ctc_batch_cost, ctc_decode
from tensorflow.python.framework.ops import EagerTensor


Analyzable = Callable[[str], str]
Decoder = Callable[[np.ndarray], tuple[str, ...]]

def ctc_loss(y_true, y_pred):
  batch_len = tf.shape(y_true)[0]
  get_length = lambda x: tf.shape(x)[1] * tf.ones(shape=(batch_len, 1), dtype=tf.int32)
  input_len, label_len = [get_length(x) for x in [y_pred, y_true]]
  return ctc_batch_cost(y_true, y_pred, input_len, label_len)


def read_as_spectrogram(path: str, frame_length: int, frame_step: int, fft_length: int) -> EagerTensor:
  audio_file = tf.io.read_file(path)
  audio = tf.audio.decode_wav(audio_file, desired_channels=1)[0]
  audio = tf.squeeze(audio, axis=-1).numpy()
  spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
  magnitude = tf.math.pow(tf.abs(spectrogram), 0.5)
  means = tf.math.reduce_mean(magnitude, 1, keepdims=True)
  stddevs = tf.math.reduce_std(magnitude, 1, keepdims=True)
  normalized = (magnitude - means) / (stddevs + 1e-10)
  return normalized


def make_batch_decoder() -> Decoder:
  characters = tuple('абвгдеёжзийклмнопрстуфхцчшщъыьэюя ')
  char_to_num = StringLookup(vocabulary=characters, oov_token='')
  num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token='', invert=True)
  decode_to_utf = lambda chars: tf.strings.reduce_join(num_to_char(chars)).numpy().decode('utf-8')
  def batch_predictions(data: np.ndarray) -> tuple[str, ...]:
    input_len = np.ones(data.shape[0]) * data.shape[1]
    results = ctc_decode(data, input_length=input_len, greedy=True)[0][0]
    return tuple(map(decode_to_utf, results))
  return batch_predictions


def mark_found(target: str, pieces: tuple[str, ...], wrap: tuple[str, ...], anycase: bool = True) -> tuple[str, dict[str, int]]:
  start, end = wrap
  if anycase:
    target = target.lower()
    start = start.lower()
    end = end.lower()
  matches: dict[str, int] = dict()
  for piece in pieces:
    found = target.count(piece)
    matches[piece] = found
    if found:
      target = target.replace(piece, start + piece + end)
  return target, matches

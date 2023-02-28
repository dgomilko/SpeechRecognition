import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from autocorrect import Speller
from tensorflow.keras.models import load_model
from keras.layers import StringLookup
from keras.backend import ctc_batch_cost, ctc_decode

MODEL_PATH = './speech_recognition_model'
FRAME_LENGTH = 256
FRAME_STEP = 160
FFT_LENGTH = 384

def ctc_loss(y_true, y_pred):
  batch_len = tf.shape(y_true)[0]
  get_length = lambda x: \
    tf.shape(x)[1] * tf.ones(shape=(batch_len, 1), dtype=tf.int32)
  input_len, label_len = [get_length(x) for x in [y_pred, y_true]]
  return ctc_batch_cost(y_true, y_pred, input_len, label_len)

def wav_to_spectrogram(wav_path):
  audio_file = tf.io.read_file(wav_path)
  audio = tf.audio.decode_wav(audio_file, desired_channels=1)[0]
  audio = tf.squeeze(audio, axis=-1).numpy()
  spectrogram = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH)
  magnitude = tf.math.pow(tf.abs(spectrogram), 0.5)
  means = tf.math.reduce_mean(magnitude, 1, keepdims=True)
  stddevs = tf.math.reduce_std(magnitude, 1, keepdims=True)
  normalized = (magnitude - means) / (stddevs + 1e-10)
  return normalized

characters = [*"абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = StringLookup(vocabulary=characters, oov_token='')
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token='', invert=True)

def decode_batch_predictions(pred):
  decode_str = lambda chars: \
    tf.strings.reduce_join(num_to_char(chars)).numpy().decode('utf-8')
  input_len = np.ones(pred.shape[0]) * pred.shape[1]
  results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
  return [decode_str(result) for result in results]

def keyword_search(text):
  BOLD = '\033[1m'
  END = '\033[0m'
  text =   'The impresario of the King’s Theatre, Pierre-Fran^ois Laporte, had assembled a company of Italian singers for a season in 1830, and Jules-Prosper Meric, Meric-Lalande’s husband, had suggested that he engage Bellini with it.'
  keywords = input('Enter space-separated keywords: ')
  keyword_list = keywords.split()
  if any(keyword in text for keyword in keywords):
    result = text
    for i in keyword_list:
      result = result.replace(i, BOLD + i + END)
    print(f'Result:\n{result}')
  else:
    print('No matches found')
  more_keywords = input('Search for more keywords? (y/n): ')
  while not more_keywords in ['y', 'n']:
    print('There is no such option')
    more_keywords = input('Enter your answer (y/n): ')
  if more_keywords == 'y': keyword_search(text)

model = load_model(MODEL_PATH, custom_objects={'ctc_loss': ctc_loss})
speller = Speller('ru')

try:
  while True:
    audio_path = input('Enter audio file path: ')
    if not os.path.isfile(audio_path):
      print(f'There is no file named {audio_path}')
      continue
    spectrogram = wav_to_spectrogram(audio_path)
    result = model.predict(spectrogram)
    decoded = ''.join(decode_batch_predictions(result))
    corrected = speller(decoded)
    with open('result.txt', 'w+') as f:
      f.write(corrected)
    print(f'Result:\n{corrected}')
    print('What would you like to do next?\n1 - keyword search\n2 - convert another audio to text')
    mode = input('mode: ')
    while not mode in ['1', '2']:
      print('There is no such option')
      mode = input('mode: ')
    if mode == '1': keyword_search(corrected)
except KeyboardInterrupt:
  print('\nExiting...')

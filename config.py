from dataclasses import dataclass
import json


dataclass
class FourierTransformParams:
  frame_length: int = 0
  frame_step: int = 0
  fft_length: int = 0


@dataclass
class Config:
  model: str
  input_dir: str
  output_dir: str
  speller: (str | None)
  anycase_search: bool
  stft: FourierTransformParams


def get_config(path: str = 'config.json') -> Config:
  with open(path, 'r') as f:
    entries = json.loads(f.read())
  stft = FourierTransformParams()
  fourier_params = entries['fourier_transform']
  for key in fourier_params:
    if hasattr(stft, key): setattr(stft, key, fourier_params[key])
  speller = entries['speller'] if 'speller' in entries else None
  anycase_search = entries['anycase_search'] if 'anycase_search' in entries else False
  return Config(
    model=entries['model'],
    input_dir=entries['input'],
    output_dir=entries['output'],
    speller=speller,
    anycase_search=anycase_search,
    stft=stft,
  )

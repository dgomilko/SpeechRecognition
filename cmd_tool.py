from enum import Enum
from config import Config
import utils


class Command(Enum):
  HELP = 'h'
  EXIT = 'e'
  ANALYZE = 'a'
  READ = 'r'
  SEARCH = 's'


class CommandTool:
  def __init__(self, speech_analyzer: utils.Analyzable, config: Config):
    pass

  def setup(self):
    raise NotImplementedError
  

  def take_input(self):
    raise NotImplementedError


  def run(self):
    raise NotImplementedError


  def on_exit(self):
    raise NotImplementedError


  def search_matches(self):
    raise NotImplementedError


  def analyze_audio(self):
    raise NotImplementedError


  def read_text_file(self):
    raise NotImplementedError

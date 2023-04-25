from types import NoneType
from enum import Enum
import os
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
    self.__speech_analyzer = speech_analyzer
    self.__config = config
    gray_bkg = '\x1b[48;5;240m'
    bkg = gray_bkg
    bold = '\033[1m'
    reset_format = '\x1b[0m'
    cmd_list = (
      'cmd speech analyzer tool:',
      ':h - help cmd list',
      ':e - exit',
      ':a <filename> - read and analyze an audio file from input dir',
      ':r <filename> - read a text file from output dir',
      ':s <...keywords> - search matches in red text'
    )
    self.__search_wrap = (bold + bkg, reset_format)
    self.__help_msg = '\n  '.join(cmd_list)
    self.__cmd = ':' + Command.HELP.value
    self.__text: str = None


  def setup(self):
    for path in self.__config.input_dir, self.__config.output_dir:
      if not os.path.isdir(path):
        if os.path.isfile(path): os.remove(path)
        os.mkdir(path)
  

  def take_input(self):
    self.__cmd = input('> ')


  def run(self):
    match self.__cmd:
      case cmd if not cmd:
        return
      case _ if self.__is_cmd_prefix(Command.HELP):
        print(self.__help_msg)
      case _ if self.__is_cmd_prefix(Command.EXIT):
        raise KeyboardInterrupt()
      case _ if self.__is_cmd_prefix(Command.SEARCH):
        self.search_matches()
      case _ if self.__is_cmd_prefix(Command.ANALYZE):
        self.analyze_audio()
      case _ if self.__is_cmd_prefix(Command.READ):
        self.read_text_file()
      case _:
        self.__show_error('no such command, use help (:h) to list existing commands')


  def on_exit(self):
    print(' Exiting')


  def search_matches(self):
    if self.__text is None:
      return self.__show_error('no text to search')
    pieces = self.__cmd[len(':' + Command.SEARCH.value):].strip().split()
    if not pieces:
      return self.__show_error('no keywords to search given')
    result, matches = utils.mark_found(self.__text, pieces, self.__search_wrap, self.__config.anycase_search)
    if not matches:
      return print('No matches found')
    matches = self.__sort_matches(matches, ascending=False)
    print(result)
    print('\nFound:')
    for key in matches: print(f'"{key}": {matches[key]} time(s)')


  def analyze_audio(self):
    file = self.__file_check(Command.ANALYZE, self.__config.input_dir)
    if file is None: return
    filename, file_path = file
    print('Analyzing...')
    self.__text = self.__speech_analyzer(file_path)
    text_path = os.path.join(self.__config.output_dir, filename[:-4] + '.txt')
    with open(text_path, 'w+') as f: f.write(self.__text)
    print('Speech recognition result:')
    print(self.__text)


  def read_text_file(self):
    file = self.__file_check(Command.READ, self.__config.output_dir)
    if file is None: return
    _, file_path = file
    with open(file_path, 'r') as f: self.__text = f.read()
    print('Red file:')
    print(self.__text)


  def  __file_check(self, command: Command, outer_dir: str) -> (tuple[str ,str] | NoneType):
    filename = self.__cmd[len(':' + command.value):].strip()
    if not filename:
      return self.__show_error('no filename given')
    file_path = os.path.join(outer_dir, filename)
    if not os.path.isfile(file_path):
      return self.__show_error('the given file does not exist')
    return filename, file_path


  def __is_cmd_prefix(self, cmd: Command) -> bool:
    return self.__cmd.startswith(':' + cmd.value)


  def __show_error(self, message: str):
    print('Error:', message)


  def __sort_matches(self, matches: dict[str, int], ascending: bool = True):
    return dict(sorted(matches.items(), key=lambda item: item[1], reverse=(not ascending)))

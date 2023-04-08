import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import config as cfg
from speech_analyzer import SpeechAnalyzer
from cmd_tool import CommandTool


CONFIG_PATH = 'config.json'

if __name__ == '__main__':
  config = cfg.get_config(CONFIG_PATH)
  speech_analyzer = SpeechAnalyzer.from_config(config)
  cmd_tool = CommandTool(speech_analyzer, config)
  cmd_tool.setup()
  try:
    while True:
      cmd_tool.run()
      cmd_tool.take_input()
  except (KeyboardInterrupt, EOFError):
    cmd_tool.on_exit()

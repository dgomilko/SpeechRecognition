# Console application for speech recognition

The app runs on python3.11. It works with **DeepSpeech** neural network model for speech recognition, defined in **deepSpeech.ipynb**.

In order to configure the environment, define **config.json** file similar to **config.example.json**, with paths to desired input and output folders and directory with trained model, language to be detected and case sensitivity for search. Then run the following commands.

**Create venv:**
```bash
python3.11 -m venv venv
```

**Activate venv (linux):**
```bash
source venv/bin/activate
```

**Dectivate venv:**
```bash
deactivate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the script:**
```bash
python main.py
```

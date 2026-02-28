# NLPTranslation

## Central Repository for Project/LASER experimentation

Set Up instructions that for LASER example that worked for me (Mac Intel Chip)

1. Set up Python virtual environment with python 3.10
brew install python@3.10

/usr/local/bin/python3.10 -m venv ~/venvs/laser310
source ~/venvs/laser310/bin/activate
python -V

2. Install the dependencies that didn't run automatically
python -m pip install --upgrade "pip<24.1" setuptools wheel
python -m pip install "numpy==1.26.4"
python -m pip install torch torchvision torchaudio
python -m pip install requests
python -m pip install laser-encoders

3. run exmaple
python callie-laser-poc.py

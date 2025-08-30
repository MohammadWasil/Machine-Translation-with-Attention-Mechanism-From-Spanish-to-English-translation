# Machine Translation with Attention Mechanism (From Spanish to English translation)

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMohammadWasil%2FMachine-Translation-with-Attention-Mechanism-From-Spanish-to-English-translation&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

The research project was associated with "[INF-DSAM1B] Advanced Machine Learning B: Deep Learning for NLP", Summer Semester 2021, for my Masters of Science: Data Science, University of Potsdam, Germany.

You can find the Technical Report on [ResearchGate](https://www.researchgate.net/publication/355917108_Neural_Machine_Translation_with_Attention).

## Version:
```
Pytroch version : 1.7.1+cu101
torchtext version: 0.8.0
spacy version: 3.1.1
```

## To install Pytorch (Conda version):
### Steps
1. Create an environment <Br/>
```conda create -n pytorch python=3.7```

2. Activate the environment <Br/>
```conda activate pytorch```

3. Install jupyter<Br/>
```conda install -c anaconda jupyter```

4. Install the ipykernel<Br/>
```pip install ipykernel```

5. Register your environment<Br/>
```python -m ipykernel install --user --name pytorch --display-name "pytorch"```

6. Install [pytorch](https://pytorch.org/get-started/locally/)<Br/>
GPU Version: ```conda install pytorch cudatoolkit -c pytorch``` <Br/>
CPU Version: ```conda install pytorch cpuonly -c pytorch```

## Installation Steps:

1. Install Python 3.6–3.8
Ensure Python 3.6–3.8 is installed on your system. For WSL/Ubuntu, you can use the `deadsnakes` PPA:

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt install python3.7.17
sudo apt install python3.7-venv python3.7-distutils -y
```

2. Create a virtual env in python and activate it:

```
python -m venv .venv
source .venv/bin/activate
```

2. Install Poetry

```
pip install poetry
```

3. Install runtime and development dependencies:
```
poetry install --no-root
```

"Model was trained for 20 Epochs and achieved a BLEU score of 25.37 (with model size of roughly 510 mb). <Br/>"

Make sure to have an empty folder, "Data"

The files can be run from cmd prompt, using the the following cmd lines:
1) cd to "/Machine_translation_with_attention_DL4NLP". Then run the below code line in cmd.
2) To train and validate the model, type:<Br/> ```python run.py --RUN_MODE train_val```
3) To evaluate the loss on test data, type:<Br/> ```python run.py --RUN_MODE test --MODEL _number_``` (```_number_``` is the model number until which you trained your model)
4) To calculate the BLEU score, type:<Br/> ```python run.py --RUN_MODE bleu --MODEL _number_``` (```_number_``` is the model number until which you trained your model)

# Machine Translation with Attention Mechanism (From Spanish to English translation)

[![Repo Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMohammadWasil%2FMachine-Translation-with-Attention-Mechanism-From-Spanish-to-English-translation&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

The research project was associated with "[INF-DSAM1B] Advanced Machine Learning B: Deep Learning for NLP", Summer Semester 2021, for my Masters of Science: Data Science, University of Potsdam, Germany.

You can find the Technical Report on [ResearchGate](https://www.researchgate.net/publication/355917108_Neural_Machine_Translation_with_Attention).

## Version:
```
Pytroch version : 1.7.1+cu101
torchtext version: 0.8.0
spacy version: 3.1.1
```

## To install Pytorch:
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

## Model was trained for 20 Epochs and achieved a BLEU score of 25.37 (with model size of roughly 510 mb). <Br/>

Different version of pytorch, torchtext and spacy might cause the program to break.

The structure of the files should looks like this:
```
.
├── Data                    
│   ├──                     # initialy this would be empty.
│   └──                     # initialy this would be empty.
├── config.yml
├── data_utils.py
├── download_data.py
├── mymodel.py
├── README.txt
└── run.py
```
Make sure to have an empty folder, "Data"

The files can be run from cmd prompt, using the the following cmd lines:
1) cd to "/Machine_translation_with_attention_DL4NLP". Then run the below code line in cmd.
2) To train and validate the model, type:<Br/> ```python run.py --RUN_MODE train_val```
3) To evaluate the loss on test data, type:<Br/> ```python run.py --RUN_MODE test --MODEL _number_``` (```_number_``` is the model number until which you trained your model)
4) To calculate the BLEU score, type:<Br/> ```python run.py --RUN_MODE bleu --MODEL _number_``` (```_number_``` is the model number until which you trained your model)

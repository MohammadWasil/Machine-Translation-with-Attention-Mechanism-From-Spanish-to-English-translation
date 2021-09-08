# Machine Translation with Attention Mechanism (From Spanish to English translation)

## Version:
```
Pytroch version : 1.7.1+cu101 <Br/>
torchtext version: 0.8.0 <Br/>
spacy version: 3.1.1 <Br/>
```

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
2) To train and validate the model, run "python run.py --RUN_MODE train_val"
3) To evaluate the loss on test data,  "python run.py --RUN_MODE test --MODEL _number_" (_number_ is the model number until which you trained your model)
4) To calculate the BLEU score, type,  "python run.py --RUN_MODE bleu --MODEL _number_" (_number_ is the model number until which you trained your model)

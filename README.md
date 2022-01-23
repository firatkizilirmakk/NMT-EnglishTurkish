# Neural Machine Translation: English - Turkish Case

This repository provides code basline for machine translation between English-Turkish.


File descriptions:

***
**Dataframes**:
* wmt16.csv: SETimes dataset
* iwstl14.csv: HuggingFace IWSLT 14
* iwslt14_.csv: IWSLT 14 downloaded from official website, parsed subset.
* root-iwslt14_.csv: the same previous dataset where Turkish words are reduced to their root form using Morse.
***
**Code**
* create_datasets.py : script to download datasets from HuggingFace, or process the datasets downloaded from HuggingFace and OPUS.
* models.py: contains the NMT architecture code.
***
* train.ipynb: consist of vocabulary creation, training loop of RNN enc-dec, Attention enc-dec and Transformer
* train.html: html output of the previous file
***
* mbart.ipynb: contains the mBart train and evaluation code.
* mbart.html: html output of the previous file
***
* Julia-Text.ipynb: applies Morse tool to create a new dataframe consisting of Turkish words with only root forms.
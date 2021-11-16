import pandas as pd
from datasets import load_dataset
from opustools import opus_read

def create_iwslt14_df():
    dataset = load_dataset("ted_talks_iwslt", language_pair = ('en', 'tr'), year = '2014', split = 'train')

    en_list, tr_list = [], []

    for i in range(len(dataset)):
        data = dataset[i]['translation']
        en = data['en']
        tr = data['tr']

        en_list.append(en)
        tr_list.append(tr)

    df = pd.DataFrame({'en': en_list, 'tr': tr_list})
    df.to_csv('iwslt14.csv', index = False)

def create_wmt16_df():
    df = pd.DataFrame()
    for split in ['train', 'validation', 'test']:
        dataset = load_dataset("wmt16", 'tr-en', split = split)
        en_list, tr_list = [], []

        for i in range(len(dataset)):
            data = dataset[i]['translation']
            en = data['en']
            tr = data['tr']

            en_list.append(en)
            tr_list.append(tr)

        tmp_df = pd.DataFrame({'en': en_list, 'tr': tr_list, 'split': split})
        df = pd.concat([df, tmp_df])

    df.to_csv('wmt16.csv', index = False)

def create_wmtnews_df():
    df = pd.DataFrame()
    length_of_txt = 30044
    line_counter = 0
    en_list, tr_list = [], []
    partition = None
    with open('wmt-news.txt', 'r') as f:
        while line_counter < length_of_txt:
            line = f.readline()

            # partition start lines
            if str(line).startswith('# en/news'):
                if partition is not None:
                    # save to df
                    tmp_df = pd.DataFrame({'en': en_list, 'tr': tr_list, 'partition': partition})
                    df = pd.concat([df, tmp_df])
                    en_list, tr_list = [], []

                partition = str(line)[5:].split('-')[0]
            elif str(line).startswith('(src)='):
                # src and trg line
                trg_line = f.readline()
                src = str(line).split('>')[1]
                trg = str(trg_line).split('>')[1]

                en_list.append(src)
                tr_list.append(trg)

                line_counter += 1

            line_counter +=1

    tmp_df = pd.DataFrame({'en': en_list, 'tr': tr_list, 'partition': partition})
    df = pd.concat([df, tmp_df])
    df.to_csv('wmt-news.csv', index = False)

def create_globalvoice_df():
    df = pd.DataFrame()
    length_of_txt = 24509
    line_counter = 0
    en_list, tr_list = [], []
    partition = None
    with open('globalvoice.txt', 'r') as f:
        while line_counter < length_of_txt:
            line = f.readline()

            # partition start lines
            if str(line).startswith('# en/20'):
                if partition is not None:
                    # save to df
                    tmp_df = pd.DataFrame({'en': en_list, 'tr': tr_list, 'partition': partition})
                    df = pd.concat([df, tmp_df])
                    en_list, tr_list = [], []

                partition = str(line)[5:].split('.')[0]
            elif str(line).startswith('(src)='):
                # src and trg line
                trg_line = f.readline()
                src = str(line).split('>')[1]
                trg = str(trg_line).split('>')[1]

                en_list.append(src)
                tr_list.append(trg)

                line_counter += 1

            line_counter +=1

    tmp_df = pd.DataFrame({'en': en_list, 'tr': tr_list, 'partition': partition})
    df = pd.concat([df, tmp_df])
    df.to_csv('globalvoice.csv', index = False)


## ORPUS other datasets info ##
# Wikimatrix, ccmatrix etc. not good. two turkish sentences as src and trg sometimes. not very good translations
# OpenSubtitles. transltions are good. Unmatched number of translations.
# CCMatrix seems good. 47 GB
# TED2020. translations are not bad. Some wrong translations. Unmatched number of translations
# EUbookshop. translations are not bad. Unmatched number of translations
# Tildemodel. translations are not bad. Some wrong translations.
# Ubuntu. it seems good. not parsing correctly!!!

opus_reader = opus_read.OpusRead(directory = 'GoURMET', source='en', target='tr', download_dir='./datasets/GoURMET', preprocess='raw',)
opus_reader.printPairs()

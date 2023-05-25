import pandas as pd
import numpy as np
import os
import pickle
from keras.preprocessing.text import Tokenizer
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


class Synonym:

    def __init__(self, text, score):
        self.text = text
        self.score = score


class Word:

    def __init__(self, index, synonym):
        self.index = index
        self.synonym = synonym


def give_paraphrase_text(path):
    filename_path = os.path.join(path, 'parafraze.csv')
    file = pd.read_csv(filename_path)
    # _, test = train_test_split(file, test_size=0.20, shuffle=False, random_state=10)
    return file.original, file.paraphrase


def merge_csv_into_one(path):
    is_first = True
    para = None
    if os.path.exists(os.path.join(path, 'parafraze.csv')):
        os.remove(os.path.join(path, 'parafraze.csv'))
    for filename in os.listdir(path):
        filename_path = os.path.join(path, filename)
        if is_first:
            para = pd.read_csv(filename_path)
            is_first = False
        else:
            para2 = pd.read_csv(filename_path)
            para = pd.concat([para, para2])
    para.to_csv(os.path.join(path, 'parafraze.csv'), index=False)


def deserialize_xml(xml_path_):
    with open(xml_path_, 'r') as file_:
        xml_string = file_.read()
    root = ET.fromstring(xml_string)

    dict = {}
    for element in root.iter('entry'):
        word = element.find('headword').text
        try:
            text = element.find('groups_core/group[1]/candidate[1]/s').text
            score = element.find('groups_core/group[1]/candidate[1]').attrib['score']
            dict[word] = Synonym(text, score)
        except Exception as e:
            print('An exception while reading synonyms:', str(e))

    return dict


def detokenize(seq, tokenizer_):
    words = list(tokenizer_.word_index.keys())
    words.insert(0, '')
    detokenized_seq = [words[round(i)] for i in seq]

    print(''.join(['' if (i == '') else i + ' ' for i in detokenized_seq]))
    return detokenized_seq


def detokenize_seq(seq, tokenizer_, best_syn_):
    words = list(tokenizer_.word_index.keys())
    words.insert(0, '')

    word = ''
    for i, seq_num in enumerate(seq):
        if i in best_syn_:
            word += best_syn_[i].synonym.text + ' '
        else:
            word += words[seq_num] + ' '

    return word


def get_best_synonyms(seq, tokenizer_, xml_dict_, num):
    # get back a text
    # [158, 258, 65, 2] -> "Danes je sončen dan"

    # loop through all words and select the one with best fit synonym
    detokenized_seq = detokenize(seq, tokenizer_)
    list_synonym = []
    for i, word in enumerate(detokenized_seq):
        if word in xml_dict_:
            list_synonym.append(Word(i, xml_dict_[word]))

    best_n = sorted(list_synonym, key=lambda word_element: word_element.synonym.score, reverse=True)[:min(num, len(list_synonym))]
    best_n_dict = {}
    for b in best_n:
        best_n_dict[b.index] = b

    return best_n_dict


# naredimo shranjevanje treh stolpcev - fraza, parafraza, nasa fraza
def return_in_csv(x_texts_, y_texts_, para_, path):
    results = pd.DataFrame(columns=['original', 'paraphrase', 'our_paraphrase'])
    length = min(len(x_texts_), min(len(y_texts_), len(para_)))

    for index in range(length):
        results = results.append({'original': x_texts_[index], 'paraphrase': y_texts_[index], 'our_paraphrase': para_[index]}, ignore_index=True)

    results.to_csv(os.path.join(path, 'results.csv'), index=False)


if __name__ == '__main__':
    print('heeeeyooo')

    paraphrase_path_directory = '../data'

    merge_csv_into_one(paraphrase_path_directory)
    x_text, y_text = give_paraphrase_text(paraphrase_path_directory)

    if os.path.exists('xml_dict.pkl'):
        with open('xml_dict.pkl', 'rb') as file:
            xml_dict = pickle.load(file)
    else:
        xml_path = os.path.join(os.getcwd(), 'CJVT_Thesaurus-v1.0.xml')
        # xml_obj = pd.read_xml(xml_path)
        # xml_dict = xml_to_dict(xml_obj)
        xml_dict = deserialize_xml('CJVT_Thesaurus-v1.0.xml')
        with open('xml_dict.pkl', 'wb') as file:
            pickle.dump(xml_dict, file)

    tokenizer = Tokenizer()

    if os.path.exists('lstmTokenizer.pickle'):
        with open('lstmTokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        tokenizer.fit_on_texts(x_text)

    # print('before:', len(tokenizer.word_index), list(tokenizer.word_index)[:2], list(tokenizer.word_index)[-2:])
    tokenizer.fit_on_texts(list(xml_dict.keys()))
    # print('after:', len(tokenizer.word_index), list(tokenizer.word_index)[:2], list(tokenizer.word_index)[-2:])
    sequences = tokenizer.texts_to_sequences(x_text)

    # num_of_seq = 100
    num_of_seq = len(x_text)
    i = 0
    paraphrases = [""] * len(x_text)
    for text, sequence in zip(x_text, sequences):
        best_syn = get_best_synonyms(sequence, tokenizer, xml_dict,
                                     round(len(sequence) / 5) if (len(sequence) > 5) else 1)
        paraphrase = detokenize_seq(sequence, tokenizer, best_syn)
        # print(text, ' || ', paraphrase, ' || ', [best_syn[x].synonym.text for x in best_syn.keys()])
        paraphrases[i] = paraphrase
        i += 1
        if i > num_of_seq:
            break

    if not os.path.exists('Results/'):
        os.makedirs('Results')
    return_in_csv(x_text, y_text, paraphrases, 'Results')
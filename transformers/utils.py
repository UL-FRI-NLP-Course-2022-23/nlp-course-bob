from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from keras.utils import pad_sequences
import pandas as pd
import os

MAX_LENGTH = 150
def tensor_datasets(x, y, mask, batch_size):
    data = TensorDataset(x, mask, y)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    
    return dataloader

def convert_to_input(data, tokenizer: AutoTokenizer):
    sentences, paraphrases = data.loc[:,"original"], data.loc[:,"paraphrase"]
    input_id_list = []
    output_id_list = []
    attention_mask_list = []
    
    for x,y in tqdm(zip(sentences,paraphrases), total=len(paraphrases), desc='Tokenizing data:'):
        input_id_list.extend([tokenizer.encode(x, max_length = MAX_LENGTH)])
        output_id_list.extend([tokenizer.encode(y, max_length = MAX_LENGTH)])

    input_id_list = pad_sequences(input_id_list,
                          maxlen=MAX_LENGTH, dtype="long", value=0.0,
                          truncating="post", padding="post")
    output_id_list = pad_sequences(output_id_list,
                          maxlen=MAX_LENGTH, dtype="long", value=0.0,
                          truncating="post", padding="post")
    
    attention_mask_list = [[float(i != 0.0) for i in ii] for ii in input_id_list]

    return torch.tensor(input_id_list).type(torch.LongTensor), torch.tensor(output_id_list).type(torch.LongTensor), torch.tensor(attention_mask_list).type(torch.LongTensor)
   
def load_data(path):
    data = pd.DataFrame()
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        data_new = pd.read_csv(filepath)
        
        if not data.empty:
            data = pd.concat([data, data_new])
        else:
            data = data_new
    
    #data.to_csv('paraphrases_combined.csv')
    return data


def load_models(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    return tokenizer, model
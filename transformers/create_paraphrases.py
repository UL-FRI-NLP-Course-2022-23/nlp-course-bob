import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm
import os

model_path = 'models/t5_final.pt'
tokenizer_path = 'models/small'
phrases_path = "data/paraphrases_30k_filtered.csv"
destination = 'results/bigger data'

def main():
    #Load model
    model = torch.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    df = pd.read_csv(phrases_path,header=None, skip_blank_lines=True)
    _, test = train_test_split(df, test_size=0.20, shuffle=False, random_state = 10)
    results = pd.DataFrame(columns = ['original', 'paraphrase', 'our_paraphrase'])

    for _, row in tqdm(test.iterrows(), total=len(test)):
        sentence = row[1]
        tokens = tokenizer.encode(sentence, max_length=150)
        input = torch.tensor([tokens]).to('cuda')
        with torch.no_grad():
            out = model.generate(input)
        out = out[0].to('cpu')
        out = tokenizer.decode(out)
        out = out.strip('<pad> ')
        out = out.strip('</s')
        results = results.append({'original': row[1], 'paraphrase':row[2], 'our_paraphrase': out}, ignore_index=True)
        
    results.to_csv(os.path.join(destination, 'results.csv'))

if __name__ == '__main__':
    main()
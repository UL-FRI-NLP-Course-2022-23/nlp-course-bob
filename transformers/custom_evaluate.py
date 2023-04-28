import pandas as pd
import torch
from transformers import AutoTokenizer

model = torch.load('test_00_t5.pt')
tokenizer = AutoTokenizer.from_pretrained("models/small")

df = pd.read_csv("parafraze_eval.txt", sep="delimiter",header=None, skip_blank_lines=True)

results = pd.DataFrame()
for index, row in df.iterrows():
    sentence = row[0]
    tokens = tokenizer.encode(sentence, max_length=128)
    input = torch.tensor([tokens]).to('cuda')
    with torch.no_grad():
        out = model.generate(input)
    out = out[0].to('cpu')
    out = tokenizer.decode(out)
    print(out)
    
    


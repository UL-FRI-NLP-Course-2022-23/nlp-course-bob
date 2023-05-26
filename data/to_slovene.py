import os
import requests
import csv
import pandas as pd
absolute_path = os.path.dirname(__file__)
url = "http://localhost:4000/api/translate"
body = {"src_language": "en", "tgt_language": "sl", "text":"Please, let this be translated."}
path = absolute_path + "/translated"
destination = absolute_path + "/paraphrases"
header = ['original', 'paraphrase']
dir_list = os.listdir(path)
starting_ix = 0
num_files = 20000
seznam = dir_list[starting_ix : starting_ix + num_files]
for i, name in enumerate(seznam):
    if name.endswith('.txt') or name.endswith("parafraze.csv"):
        continue
    full_name = f"{path}//{name}"
    #print(full_name)
    df = pd.read_csv(full_name, encoding='utf-8')
    rows = df.to_numpy()
    new_rows = []
    dolzina = len(rows)
    for j, value in enumerate(rows):
        body["text"] = value[1]
        if len(value[1]) >= 5000:
            body['text'] = value[1][0:4999]
        #print(body)
        parafraza = requests.post(url, json=body).json()['result']
        new_rows.append([value[0], parafraza])
        #print(parafraza)
        print(f"{i}/{num_files}, {j}/{dolzina}")

    csv_file = full_name.replace(".csv", "_parafraze.csv")
    csv_file = destination + "//" + name.replace(".csv", "_parafraze.csv")
    with open(csv_file, 'w', encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_rows)
        

import os
import requests
import csv
absolute_path = os.path.dirname(__file__)
url = "http://localhost:4000/api/translate"
body = {"src_language": "sl", "tgt_language": "en", "text":"Prosim, naj bo to prevedeno"}
path = absolute_path + "/gigafidaCleaned"
destination = absolute_path + "/translated"
header = ['original', 'translation']
dir_list = os.listdir(path)
starting_ix = 0
num_files = 20000
seznam = dir_list[starting_ix : starting_ix + num_files]
for i, name in enumerate(seznam):
    if not name.endswith('.txt'):
        continue
    full_name = f"{path}//{name}"
    text = []
    rows = []
    with open(full_name, 'r', encoding="utf-8") as file:
        data = file.read().strip().split('\n')
        text = [s.replace('\n', '') for s in data]

    dolzina = len(text)
    for j,line in enumerate(text):
        body["text"] = line
        if len(line) >= 5000:
            body["text"] = line[0:4999]

        translation = requests.post(url, json=body).json()["result"]
        rows.append([body["text"], translation])
        print(f"{i}/{num_files}, {j}/{dolzina}")
    
    csv_file = destination + "//" + name.replace(".txt", ".csv")
    with open(csv_file, 'w', encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


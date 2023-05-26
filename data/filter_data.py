import classla
import os
absolute_path = os.path.dirname(__file__)
path = absolute_path + "/ccGigafida"
destPath = absolute_path + "/gigafidaCleaned"
classla.download('sl')
nlp = classla.Pipeline('sl', processors='tokenize,pos')
dir_list = os.listdir(path)
starting_ix = 0
num_files = 20000
seznam = dir_list[starting_ix : starting_ix + num_files]
st_povedi = 0
for i, name in enumerate(seznam):
    if not name.endswith('.txt'):
        continue
    full_name = f"{path}//{name}"
    text = []
    rows = []
    with open(full_name, 'r', encoding="UTF8") as file:
        data = file.read().strip().split('\n\n')
        text = [s.replace('\n', '') for s in data]
    good_sentences = []
    bad_sentences = []
    dolzina = len(text)
    for j,line in enumerate(text):
        doc = nlp(line)
        st_povedi += 1
        for sentence in doc.sentences:
            jePovedek = False
            st_nezazelenih = 0
            for word in sentence.words:
                if(word.upos == 'NUM' or word.upos == 'PUNCT' or word.upos == 'SYM'):
                    st_nezazelenih += 1
                if (word.upos == 'VERB' or word.upos == 'AUX'):
                    jePovedek = True
            if st_nezazelenih/len(sentence.words) > 0.8:
                bad_sentences.append(sentence.text)
                break
            if jePovedek:
                good_sentences.append(sentence.text)
            else:
                bad_sentences.append(sentence.text)
        print(f"{i}/{num_files}, {j}/{dolzina} | {st_povedi}")

    file_changed = full_name.replace(path, destPath)
    file_good = file_changed.replace(".txt", "_good.txt")
    #file_bad = file_changed.replace(".txt", "_bad.txt")
    with open(file_good, 'w', encoding="utf-8") as f:
        for line in good_sentences:
            f.write(f"{line}\n")
    #with open(file_bad, 'w', encoding="utf-8") as f:
    #    for line in bad_sentences:
    #        f.write(f"{line}\n")
 
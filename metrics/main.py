import pandas as pd
import evaluate
import classla
import os


def bleu_evaluate(bleu, predictions, references):
    results = bleu.compute(predictions=predictions, references=references)
    print(results)


def meteor_evaluate(meteor, predictions, references):
    results = meteor.compute(predictions=predictions, references=references)
    print(results)


def bert_score_evaluate(bertscore, predictions, references):
    results = bertscore.compute(predictions=predictions, references=references, lang='sl')
    precision = sum(results['precision']) / len(results['precision'])
    f1 = sum(results['f1']) / len(results['f1'])
    recall = sum(results['recall']) / len(results['recall'])
    print('Precision:', precision, 'F1:', f1, 'Recall:', recall)


def lematize(nlp, list_texts):
    result_list = [''] * len(list_texts)
    for i, text in enumerate(list_texts):
        doc = nlp(text)
        # nlp('France prešeren je rojen v vrbi. Danes je sončen dan.').sentences[0].words[0].lemma
        result = ''
        for sentence in doc.sentences:
            for word in sentence.words:
                result += word.lemma + ' '
        result_list[i] = result
    return result_list


# naredimo branje treh stolpcev - fraza, parafraza, nasa fraza
def get_phrases(path):
    filename_path = os.path.join(path, 'results.csv')
    file = pd.read_csv(filename_path)
    # _, test = train_test_split(file, test_size=0.20, shuffle=False, random_state=10)
    our_paraphrase = file.our_paraphrase.to_numpy(dtype=str)
    for i, para in enumerate(our_paraphrase):
        our_paraphrase[i] = str.replace(str.replace(para, '/s>', ''), '<pad>', '')

    return file.original, file.paraphrase, our_paraphrase


if __name__ == '__main__':
    print('heeeeyoo')
    # _, references, predictions = get_phrases('../small_data_results/second_results')
    _, references, predictions = get_phrases('../thesaurus/small_data_results')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    bertscore = evaluate.load('bertscore')
    classla.download('sl')
    nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma')

    bert_score_evaluate(bertscore, predictions, references)
    references = lematize(nlp, references)
    predictions = lematize(nlp, predictions)
    bleu_evaluate(bleu, predictions, references)
    meteor_evaluate(meteor, predictions, references)




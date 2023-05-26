# Natural language processing course 2022/23: `Project 3: Paraphrasing sentences`

Team members:

* `Gregor Zadnik`, `63170313`, `gz4131@student.uni-lj.si`
* `Lenart Rupnik`, `63220472`, `lr95263@student.uni.lj.si`
* `Franc Benjamin Demšar`, `63170075`, `fd8651@student.uni-lj.si`

Group public acronym/name: `Ross`

> This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

## Report

The report is available in the `submission 3` folder.

## Data preparation

The initial unfiltered dataset (ccGigafida) is available in the `data/ccGigafida` folder. To filter out the non-sentences, simply run the `data/filter_data.py` script.
This will produce the data present in the `data/gigafidaCleaned` folder.

To generate training data you need to download and extract the [following zip file](https://drive.google.com/file/d/1VoHoFJv52mxa9Ebr7-vjEXtufiRLoyBp/view?usp=sharing) into any folder. Afterwards, follow the `Deployment` instructions on [this Github repository](https://github.com/clarinsi/Slovene_NMT) inside the extracted folder. When the docker container is setup and has been running for about 5 minutes, the NMT model is ready to translate.

To do so, run the `data/to_english.py` script, which will begin translating the sentences into english. When it is done, or you cancel the script, run the `data/to_slovene.py` script. The training data should now be available in the `paraphrases` directory. The last step is to combine all the files into one big csv file by running the `data/join_data.py` script. The final file is now available as `bigger dataset/paraphrases_all.csv`

As this whole process takes a while, you might want to use the prepared `bigger dataset/paraphrases_all.csv` or `bigger dataset/paraphrases_30k_filtered.csv` instead.

## Transformer model usage

In order to use our model for pharaphrasing, you first need to create a new environment using `requirements_tf.yaml` file by running:
```
$ conda env create -n <ENV_NAME> -f requirements_tf.yaml
```

Since we used PyTorch with GPU enchanchments you also need an Nvidia compatable GPU with installed CUDA 11.4 framework to create this environment. Installation guide can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). If your CUDA version DOES NOT MATCH, the environment can not be recreated.

Next you need to download our model from [here](https://drive.google.com/file/d/14ArLqplnn1WAA34IkjTq3p1jod6fYX4j/view?usp=sharing). To simplify the usage, place the downloaded model in `models/` folder.

To run the model on our dataset you need to run `create_pharaphrases.py` from `transformers/` folder.

## Training new models

For training new models via a pre-trained `t5_small` from [cjvt](https://huggingface.co/cjvt/t5-sl-small), you first need to download pre-trained model using `transformers/download_t5.py` file. Next you can modify training parameters in `transformers/train_transformers.py` or simply run this file to train the model with our parameters. If you lack GPU RAM to train the model, you can reduce `BATCH_SIZE` or `MAX_LENGTH` parameters in `transformers/train_transformers.py`.

## Thesaurus - synonyms method usage

In order to use our synonym method - Thesaurus for paraphrasing, you first need to create a new environment using `requirements_mt.yaml` file by running:

```
$ conda create --name <environment_name> --file requirements_mt.yaml
```
Afterwards, run the `thesaurus/main.py` script. The results will be visible in file `thesaurus/manual_data_results/results.csv`

## Metrics usage

In order to use our metrics, you first need to create a new environment using `requirements_mt.yaml` file by running:

```
$ conda create --name <environment_name> --file requirements_mt.yaml
```

You can also use the environment from the thesaurus method.
Afterwards, run the `metrics/main.py` script. The results will be visible in file `metrics/metric_results/small_data_results.txt`
We implemented 3 different metrics: Bert-Score, BLEU and METEOR.

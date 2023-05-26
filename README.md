# Natural language processing course 2022/23: `Project 3: Paraphrasing sentences`

Team members:

* `Gregor Zadnik`, `63170313`, `gz4131@student.uni-lj.si`
* `Lenart Rupnik`, `63220472`, `lr95263@student.uni.lj.si`
* `Franc Benjamin Demšar`, `63170075`, `fd8651@student.uni-lj.si`

Group public acronym/name: `Ross`

> This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

## Data preparation

The initial unfiltered dataset (ccGigafida) is available in the `data/ccGigafida` folder. To filter out the non-sentences, simply run the `data/join_data.py` script.
This will produce the data present in the `data/gigafidaCleaned` folder.

To generate training data, you need to download and extract the [following zip file](https://drive.google.com/file/d/1VoHoFJv52mxa9Ebr7-vjEXtufiRLoyBp/view?usp=sharing) into any folder. Afterwards, follow the `Deployment` instructions on [this Github repository](https://github.com/clarinsi/Slovene_NMT) inside the extracted folder. When the docker container is setup and has been running for about 5 minutes, the NMT model is ready to translate.

To do so, run the `data/to_english.py` script, which will begin translating the sentences into english. When it is done, or you cancel the script, run the `data/to_slovene.py` script. The training data should now be available in the `paraphrases` directory. The last step is to combine all the files into one big csv file by running the `data/join_data.py` script. The final file is now available as `bigger dataset/paraphrases_all.csv`

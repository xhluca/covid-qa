# COVID-QA

*A collection of COVID-19 Q&A pairs and transformer baselines for evaluating question-answering models*

### Links

* 💾 [Official Kaggle Dataset](https://www.kaggle.com/xhlulu/covidqa)
* 💻 [Official Github Repository](https://github.com/xhlulu/covid-qa)
* :books: [Technical Report on Arxiv]() (To be added)

### Data summary

* 400+ pairs collected from 15 English news websites across 4 continents
* 600+ pairs queried from 26 Stackexchange communities, grouped in 3 distinct categories
* 800+ pairs retrieved from CDC and WHO's official FAQs, available in 8 languages
* All the pairs are cleaned with regex, labelled with metadata, converted to tables, and included heuristically-selected negative samples (wrong answers).

In addition, we included a clean, tabular version of 290k non-COVID Q&A pairs, queried from the same Stackexchange communities. You can [download it here](https://www.kaggle.com/xhlulu/stackexchange-qa-pairs).

### Model summary

* Electra-small and Electra-base, both trained on 500k subsample of [Healthtap Q&A pairs](https://github.com/durakkerem/Medical-Question-Answer-Datasets).
* Electra-small and Electra-base, both trained on 290k question and answer pairs from Stackexchange.
* 2 versions of Multilingual DistilBERT, trained on Healthtap and on Stackexchange, respectively.
* All the models were finetuned in `tf.keras` on Kaggle's TPUs. All the training scripts will be made available for reproducibility.

### Cite this work

TBD

## Model Usage


# COVID-QA

*A collection of COVID-19 Q&A pairs and transformer baselines for evaluating question-answering models*

### Links

💾 [Official Kaggle Dataset](https://www.kaggle.com/xhlulu/covidqa)

💻 [Official Github Repository](https://github.com/xhlulu/covid-qa)

:bookmark: [Alternate Download Link](https://github.com/xhlulu/covid-qa/releases)

### Data summary

* **400+ pairs** collected from 15 English news websites across 4 continents.
* **600+ pairs** queried from 26 Stackexchange communities, grouped in 3 distinct categories.
* **800+ pairs** retrieved from CDC and WHO's official FAQs, available in 8 languages.
* All pairs are cleaned with regex, labelled with metadata, converted to tables, and stored in CSV files.
* Each question also includes a heuristically-sampled negative (incorrect) answer. The selection process varies depending on the dataset, and is optional.

In addition, we included a clean, tabular version of **290k non-COVID Q&A pairs**, queried from the same Stackexchange communities. You can [download it here](https://www.kaggle.com/xhlulu/stackexchange-qa-pairs).

### Model summary

* Electra-small and Electra-base, both trained on 500k subsample of [Healthtap Q&A pairs](https://github.com/durakkerem/Medical-Question-Answer-Datasets).
* Electra-small and Electra-base, both trained on 290k question and answer pairs from Stackexchange.
* 2 versions of Multilingual DistilBERT, trained on Healthtap and on Stackexchange, respectively.
* All the models were finetuned in `tf.keras` on Kaggle's TPUs. 
* All the training scripts are available on Kaggle, and can be easily rerun. Check out [this section](#kaggle-notebooks) for more information.

### How do the baseline models work?

In order to make it accessible, we designed our baselines with the simplest Q&A mechanism available for transformer models: concatenate the question with the answer, and let the model learn to predict if it is a correct match (label of 1) or incorrect match (label of 0). Here's an example of how we process the data:

[![](images/HowDoesCOVIDQAWork-Page-1.svg)](#)


### Why do we need this type of Q&A Models?

The baseline do not auto-regressively generate an answer, so it is not a generative model. Instead, it can tell you if a pair of question and answer is reasonable or not. This is useful when you have a new question (e.g. asked by a user) and a small set of candidate answers (that we pre-filtered from a database of answers), and your goal is to either select the best answer, or rerank those candidates in order of relevance. The latter is used by [Neural Covidex](https://arxiv.org/abs/2004.05125), a search engine about COVID-19. Here's how you could visually think about it:

![](images/HowDoesCOVIDQAWork-Page-2.svg)

### Are you releasing a new model? Can we start using it for our projects?

The goal of COVID-QA is not to release new models, but to **provide a dataset for evaluating your own Q&A models**, along with strong **baselines that you can easily reproduce and improve**. Both the data and models are there to help you for your research projects or R&D prototypes. **If you are planning to build and deploy any model or system that uses COVID-QA in some way, please ensure that it is sufficiently tested and validated by medical and public health experts. The content of this collection has not been medically validated.**

### Cite this work

We don't currently have a paper about this work. Feel free to link to this repository, or to the Kaggle dataset. Please reach out if you are interested in citing a technical report.

## Data Usage

To load the data, simply download the data from Kaggle or from the alternative link. Then, use pandas to load it:

```python
import pandas as pd

community = pd.read_csv("path/to/dataset/community.csv")
community.head()
```

## Model Usage

### Preliminary
First, make sure to download the data from Kaggle or from the alternative link, and unzip the directory. Also, make sure to have the `utils` script in your current directory. For example:

```
wget https://github.com/xhlulu/covid-qa/releases/download/v1.0/electra-small-healthtap.zip
unzip electra-small-healthtap.zip
```

Then, make sure that transformers and tensorflow are correctly installed:
```
pip install transformers>=2.8.0
pip install tensorflow>=2.1.0 # or pip install tensorflow-gpu>=2.1.0
```

### Helper function
Then, define the following helper functions in your Python script:
```python
import os
import pickle

import tensorflow as tf
import tensorflow.keras.layers as L
import transformers as trfm

def build_model(transformer, max_len=None):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_ids = L.Input(shape=(max_len, ), dtype=tf.int32)
    
    x = transformer(input_ids)[0]
    x = x[:, 0, :]
    x = L.Dense(1, activation='sigmoid', name='sigmoid')(x)
    
    # BUILD AND COMPILE MODEL
    model = tf.keras.Model(inputs=input_ids, outputs=x)
    model.compile(
        loss='binary_crossentropy', 
        metrics=['accuracy'], 
        optimizer=Adam(lr=1e-5)
    )
    
    return model

def load_model(sigmoid_dir='transformer', transformer_dir='transformer', architecture="electra", max_len=None):
    """
    Special function to load a keras model that uses a transformer layer
    """
    sigmoid_path = os.path.join(sigmoid_dir,'sigmoid.pickle')
    
    if architecture == 'electra':
        transformer = trfm.TFElectraModel.from_pretrained(transformer_dir)
    else:
        transformer = trfm.TFAutoModel.from_pretrained(transformer_dir)
    model = build_model(transformer, max_len=max_len)
    
    sigmoid = pickle.load(open(sigmoid_path, 'rb'))
    model.get_layer('sigmoid').set_weights(sigmoid)
    
    return model
```

### Loading model
Then, you can load it as a `tf.keras` model:

```python
model = load_model(
  sigmoid_dir='/path/to/sigmoid/dir/', 
  transformer_dir='/path/to/transformer/dir/'
)
```

Sometimes the sigmoid file is not stored in the same directory as the transformer files, so make sure to double check it.

### Loading tokenizer

The tokenizer used is exactly the same as the original tokenizers that we loaded from huggingface model repository. E.g.:
```python
tokenizer = trfm.ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
```

You can also load the fast tokenizer from Huggingface's `tokenizers` library:
```python
from tokenizers import BertWordPieceTokenizer
fast_tokenizer = BertWordPieceTokenizer('/path/to/model/vocab.txt', lowercase=True, add_special_tokens=True)
```
Where `add_special_tokens` depends on whether you are using adding the tags manually or not.

Then, you can use the following function to encode the questions and answers:
```python
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512, enable_padding=False):
    """
    ---
    Inputs:
        tokenizer: the `fast_tokenizer` that we imported from the tokenizers library
    """
    tokenizer.enable_truncation(max_length=maxlen)
    if enable_padding:
        tokenizer.enable_padding(max_length=maxlen)
    
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
```

### Advanced model usage

For more advanced and complete examples of using the models, please check out those notebooks:
* [Community-QA](https://www.kaggle.com/xhlulu/evaluate-models-on-community-data)
* [News-QA](https://www.kaggle.com/xhlulu/evaluate-models-on-news-data)
* [Multilingual-QA](https://www.kaggle.com/xhlulu/evaluate-models-on-multilingual-data)

### Future works for ease of access

We are hoping to potentially host the base model on the Huggingface repository. Currently, we are faced with problems concerning the sigmoid layer, which can't be easily added to the model. We will evaluate the next step in order to make the model available.

We are also planning to make a `utils` file that you can download off this repo, so you won't need to copy paste those files.

## Kaggle Notebooks

TBA

## Results

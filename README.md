# COVID-QA

*A collection of COVID-19 Q&A pairs and transformer baselines for evaluating question-answering models*

### Links

💾 [Official Kaggle Dataset](https://www.kaggle.com/xhlulu/covidqa)

💻 [Official Github Repository](https://github.com/xhlulu/covid-qa)

:books: [Technical Report on Arxiv]() (To be added)

:bookmark: [Alternative Download Links](https://github.com/xhlulu/covid-qa/releases)

### Data summary

* **400+ pairs** collected from 15 English news websites across 4 continents.
* **600+ pairs** queried from 26 Stackexchange communities, grouped in 3 distinct categories.
* **800+ pairs** retrieved from CDC and WHO's official FAQs, available in 8 languages.
* All pairs are cleaned with regex, labelled with metadata, converted to tables, and stored in CSV files.
* Each question also includes a heuristically-sampled negative (wrong) answer. The selection process varies depending on the dataset.

In addition, we included a clean, tabular version of **290k non-COVID Q&A pairs**, queried from the same Stackexchange communities. You can [download it here](https://www.kaggle.com/xhlulu/stackexchange-qa-pairs).

### Model summary

* Electra-small and Electra-base, both trained on 500k subsample of [Healthtap Q&A pairs](https://github.com/durakkerem/Medical-Question-Answer-Datasets).
* Electra-small and Electra-base, both trained on 290k question and answer pairs from Stackexchange.
* 2 versions of Multilingual DistilBERT, trained on Healthtap and on Stackexchange, respectively.
* All the models were finetuned in `tf.keras` on Kaggle's TPUs. All the training scripts will be made available for reproducibility.

### Cite this work

TBD

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
wget 
```

Then, make sure that transformers and tensorflow are correctly installed:
```
pip install transformers==2.8.0
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
  sigmoid_dir='/path/to/the/model/sigmoid.pickle', 
  transformer_dir='/path/to/model/transformer/'
)
```

Sometimes the sigmoid file is not stored in the same directory as the transformer files, so make sure to load it correctly.

### Loading tokenizer

The tokenizer used is exactly the same as the original tokenizers that we loaded from huggingface model repository. E.g.:
```python
tokenizer = trfm.ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
```

You can also load the fast tokenizer from Huggingface's `tokenizers` library:
```python
fast_tokenizer = BertWordPieceTokenizer('/path/to/model/vocab.txt', lowercase=True, add_special_tokens=True)
```

Then, you can use the following function to encode the questions and answers:
```
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

We are planning to host the base model on the Huggingface repository. Currently, we are faced with problems concerning the sigmoid layer, which can't be easily added to the model. We will evaluate the next step in order to make the model available.

We are also planning to make a `utils` file that you can download off this repo, so you won't need to copy paste those files.

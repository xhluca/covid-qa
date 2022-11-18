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
* All the training scripts are available on Kaggle, and can be easily rerun. Check out the ["Source Code" section](#source-code-and-kaggle-notebooks) for more information.

### Are you releasing a new model for diagnosing COVID-19? Can we start using it for our projects?

The goal of COVID-QA is not to release novel models, but to **provide a dataset for evaluating your own Q&A models**, along with strong **baselines that you can easily reproduce and improve**. In fact, the datasets relate more closely to news, public health, and community discussions; it is **not intended to be used in a clinical setting, and should not be used to influence clinical outcomes**. Both the data and models are there to help you for your research projects or R&D prototypes. **If you are planning to build and deploy any model or system that uses COVID-QA in some way, please ensure that it is sufficiently tested and validated by medical and public health experts. The content of this collection has not been medically validated.**

### How do the baseline models work?

In order to make it accessible, we designed our baselines with the simplest Q&A mechanism available for transformer models: concatenate the question with the answer, and let the model learn to predict if it is a correct match (label of 1) or incorrect match (label of 0). Ideally, when trained correctly, we want our model to behave this way:

![](images/HowDoesCOVIDQAWork-Page-1.svg)


### Why do we need this type of Q&A Models?

The baselines do not auto-regressively generate an answer, so it is not a generative model. Instead, it can tell you if a pair of question and answer is reasonable or not. This is useful when you have a new question (e.g. asked by a user) and a small set of candidate answers (that was pre-filtered from a database of **reliable and verified answers**), and your goal is to either select the best answer, or rerank those candidates in order of relevance. The latter is used by [Neural Covidex](https://arxiv.org/abs/2004.05125), a search engine about COVID-19. Here's how you could visually think about it:

![](images/HowDoesCOVIDQAWork-Page-2.svg)

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

For more advanced and complete examples of using the models, please check out the [model evaluation section](#model-validation)


### Future works for ease of access

We are hoping to potentially host the base model on the Huggingface repository. Currently, we are faced with problems concerning the sigmoid layer, which can't be easily added to the model. We will evaluate the next step in order to make the model available.

We are also planning to make a `utils` file that you can download off this repo, so you won't need to copy paste those files.

## Source Code and Kaggle Notebooks

For this project, our workflow mostly consisted of pipelines of Kaggle notebooks that first preprocess the data, then train a model, and finally evaluate them on each of the tasks we are proposing. To reproduce our results, simply click "Copy and Edit" any of the notebooks below. If you are not familiar with Kaggle, [check out this video](https://www.youtube.com/watch?v=O1P4r0Iy55U).

For archival purposes, we also included all the notebooks inside this repository under `notebooks`.

### Preprocessing

The following notebooks show how to preprocess the relevant datasets for training:
* [Preprocess HealthTap Data](https://www.kaggle.com/xhlulu/healthtap-eda)
* [Preprocess StackExchange Data](https://www.kaggle.com/xhlulu/stackexchange-eda-and-preprocess)

Since the StackExchange dataset consumed a lot of memory, we decided to create and save the encoded input of the training data in a separate notebook:
* [Encode StackExchange for ELECTRA models](https://www.kaggle.com/xhlulu/stackexchange-encode-for-electra)
* [Encode StackExchange for Multilingual DistilBERT](https://www.kaggle.com/xhlulu/encode-stackexchange-for-mdistilbert)

### Model Training

Each of the 6 baselines were trained using a TPU notebook. You can find them here:
* [Finetune Electra Small on HealthTap](https://www.kaggle.com/xhlulu/healthtap-joint-electra-small)
* [Finetune Electra Base on HealthTap](https://www.kaggle.com/xhlulu/healthtap-joint-electra-base)
* [Finetune Electra Small on StackExchange](https://www.kaggle.com/xhlulu/stackexchange-finetune-electra-small)
* [Finetune Electra Base on StackExchange](https://www.kaggle.com/xhlulu/stackexchange-finetune-electra-base)
* [Finetune Multilingual DistilBERT on HealthTap](https://www.kaggle.com/xhlulu/finetune-mdistilbert-on-healthtap)
* [Finetune Multilingual DistilBERT on StackExchange](https://www.kaggle.com/xhlulu/finetune-mdistilbert-on-stackexchange)

### Model Validation

* [Evaluate on Community-QA](https://www.kaggle.com/xhlulu/evaluate-models-on-community-data)
* [Evaluate on News-QA](https://www.kaggle.com/xhlulu/evaluate-models-on-news-data)
* [Evaluate on Multilingual-QA](https://www.kaggle.com/xhlulu/evaluate-models-on-multilingual-data)

## Acknowledgements

Thank you to: @JunhaoWang and @Makeshn for helping build the dataset from scratch; Akshatha, Ashita, Louis-Philippe, Jeremy, Joao, Joumana, Mirko, Siva for the helpful and insightful discussions.

## Aggregated Results

Below are some aggregated results (Macro-averaged across all sources) from the output of our evaluation notebooks. Please check them out for more complete metrics! 

* `ht`: HealthTap
* `se`: StackExchange
* `ap`: Average Precision
* `roc_auc`: Area under ROC curve

### Community-QA

|          |   electra_ht_small |   electra_ht_base |   electra_se_small |   electra_se_base |
|:---------|-------------------:|------------------:|-------------------:|------------------:|
| ap       |             0.5609 |            0.6792 |             0.9429 |            0.9396 |
| roc_auc  |             0.5898 |            0.7097 |             0.9559 |            0.9586 |
| f1_score |             0.6744 |            0.6817 |             0.8946 |            0.915  |
| accuracy |             0.5218 |            0.5374 |             0.891  |            0.912  |

### Multilingual-QA

|          |   mdistilbert_ht |   mdistilbert_se |
|:---------|-----------------:|-----------------:|
| ap       |           0.7635 |           0.5611 |
| roc_auc  |           0.7709 |           0.5963 |
| f1_score |           0.7219 |           0.688  |
| accuracy |           0.6222 |           0.5501 |

### News-QA

|          |   electra_ht_small |   electra_ht_base |   electra_se_small |   electra_se_base |
|:---------|-------------------:|------------------:|-------------------:|------------------:|
| ap       |             0.9038 |            0.9273 |             0.6691 |            0.7553 |
| roc_auc  |             0.9186 |            0.9327 |             0.7164 |            0.8053 |
| f1_score |             0.8433 |            0.8527 |             0.7113 |            0.7762 |
| accuracy |             0.842  |            0.8524 |             0.659  |            0.7266 |

## AP score by source

Below are the average precisions (AP) for each source, for every task.

### Multilingual-QA

|            |   mdistilbert_ht |   mdistilbert_se |
|:-----------|-----------------:|-----------------:|
| chinese    |           0.8075 |           0.5281 |
| english    |           0.8191 |           0.6495 |
| korean     |           0.5926 |           0.5091 |
| spanish    |           0.7892 |           0.5546 |
| vietnamese |           0.6264 |           0.5994 |
| arabic     |           0.7339 |           0.5669 |
| french     |           0.8605 |           0.5876 |
| russian    |           0.7951 |           0.4844 |

### News-QA

|                |   electra_ht_small |   electra_ht_base |   electra_se_small |   electra_se_base |
|:---------------|-------------------:|------------------:|-------------------:|------------------:|
| ABC Australia  |             0.8968 |            0.886  |             0.6931 |            0.74   |
| ABC News       |             0.8825 |            0.9334 |             0.6492 |            0.6274 |
| BBC            |             0.8977 |            0.9259 |             0.7382 |            0.8679 |
| CNN            |             0.9525 |            0.9436 |             0.7052 |            0.8598 |
| CTV            |             0.8225 |            0.9339 |             0.7062 |            0.8579 |
| Forbes         |             0.7534 |            0.8302 |             0.7077 |            0.7361 |
| LA Times       |             0.875  |            0.95   |             0.7095 |            0.6458 |
| NDTV           |             0.8675 |            0.8915 |             0.679  |            0.7449 |
| NPR            |             0.972  |            0.9637 |             0.6752 |            0.8085 |
| NY Times       |             0.9604 |            0.9455 |             0.6489 |            0.8077 |
| SCMP           |             0.9415 |            0.9464 |             0.8155 |            0.8523 |
| The Australian |             0.8179 |            0.8    |             0.607  |            0.8556 |
| The Hill       |             0.9377 |            0.9734 |             0.6382 |            0.7539 |
| Times Of India |             0.9869 |            0.9824 |             0.7823 |            0.7366 |
| USA Today      |             0.8995 |            0.9391 |             0.7237 |            0.7689 |

### Community-QA

|             |   electra_ht_small |   electra_ht_base |   electra_se_small |   electra_se_base |
|:------------|-------------------:|------------------:|-------------------:|------------------:|
| biomedical  |             0.5851 |            0.6902 |             0.9508 |            0.947  |
| general     |             0.571  |            0.7097 |             0.9538 |            0.956  |
| expert      |             0.5265 |            0.6233 |             0.8994 |            0.8858 |


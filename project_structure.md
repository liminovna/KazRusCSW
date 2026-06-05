# Структура проекта
:earth_americas: For English version see [below](#project%20structure)

# Project structure
This repository includes notebooks for text processing, experiments and analysis, and notebooks for model training.

## Text processing

The following notebooks represent the four stages the data went through to form the KazRusCSWCorpus (not including the zero-th step which is data scraping): 
1) preprocessing, 
2) filtering out irrelevant texts, 
3) token annotation and elicitaion of code-mixed data and 
4) code-mixing metrics calculation

- `1__data_prep.ipynb` + `preprocessing.py` 

The notebook contains code for text preprocessing. It assumes a certain structure in the data so the code is not really reusable. Still, it provides an insight into how the raw texts are cleaned and sensitive data is masked . For preprocessing, the notebook uses the function from `preprocessing.py`.

Basically, during this stage, we first filter out the comments that contain a dividing line which usually indicates that the comment is just a message that is written first in one language and then translated into another. Then, we get rid of comments that do not contain any cyrillic letters, mask emoji, card and phone numbers, links, hashtags, emails and mentions. We replace newlines with `\\n` since we treat it as a meaningful token (like a full stop, for instance), and swap all the other whitespaces with a regular space. Then we filter out comments that do not contain at least three words written in cyrillic characters. Finally we drop duplicate texts and assign a uuid to each comment.

In our case, we started out with 1,590,529 texts and after preprocessing we were left with 1,122,792 texts.

- `2__filtering.ipynb`
In this notebook, we filter out texts that are not mainly in either Kazakh or Russian. We do that by running a [GlotLID model](https://github.com/cisnlp/GlotLID) on each document and if the most likely language is not Kazakh or Russian we get rid of it.

The decision to use this model specifically is bassed on the analysis that was carried out in the two notebooks [here](colab_notebooks/experiments)

After this step we were left with 1,009,159 texts.

- `3__token_annot.ipynb`
After filtering the 

- `4__metrics.ipynb`
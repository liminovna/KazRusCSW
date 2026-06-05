# Структура проекта
:earth_americas: For English version see [below](#project%20structure)

Репозиторий содержит блокноты с обработкой текстов, экспериментами и анализом, а также обучением моделей для потокенной разметки.

## Обработка текстов

Перечисленные ниже блокноты представляют собой 4 этапа, через которые прошли сырые тексты (не считая этапа сбора этих самых текстов), чтобы в конце концов образовался корпус KazRusCSWCorpus:

1) предобработка
2) фильтрация от "нерелевантных" текстов
3) автоматическая разметка на уровне токенов и выявление текстов с переключением кодов
4) подсчет метрик по корпусу

- `1__data_prep.ipynb` + `preprocessing.py` 

Блокнот содержит код для предобработка текстов. Он подразумевает определенную структуру хранения данных, поэтому переиспользовать его будет трудно. Однако по нему можно понять, как сырые тексты чистятся и маскируются. Основной алгоритм предобработки содержится в `preprocessing.py`.

Если кратко, то на этом этапе мы отфильтровываем все комментарии, содержащие разделительную линию, поскольку, как показал опыт, это означает, что текст представляет собой дублирование одного и того же предложения на двух языках. Затем мы избавляемся от комментариев, которые не содержат кириллических символов, маскируем эмодзи, номера банковских карт и телефонов, ссылки, хэштеги, эл. почты и упоминания. Также мы заменяем переносы строки на `\\n`, т.к. в нашем случае он приравнивается к символу окончания предложения, и заменяем все whitespace-символы на обычный пробел. Затем мы отсеиваем комментарии, которые не содержат по крайней мере трех слов, написанных на кириллице. В конце мы удаляем дубли и добавляем уникальный идентификатор к оставшимся комментариям.

В нашем случае мы изначально собрали 1,590,529 сырых текстов, и после предобработки and осталось 1,122,792 текстов.

- `2__filtering.ipynb`

В этом блокноте мы отфильтровываем все комментарии, которые по большей части не написаны на казахском или русском. Для этого мы прогоняем модель [GlotLID model](https://github.com/cisnlp/GlotLID) на комментариях и если наиболее вероятный язык не является казахским или русским, мы его откидываем.

Решение использовать именно эту модель основано на анализе, представленном в [этих блокнотах](colab_notebooks/experiments)

После этого этапа осталось 1,009,159 текстов.

- `3__token_annot.ipynb`

После этапа фильтрации мы потокенно размечаем все оставшиеся тексты с помощью дообученной модели на основе [mBERT](https://huggingface.co/liminovna/KazRusCSW-mbert) ([здесь](colab_notebooks/model_training) можно увидеть блокнот с обучением модели).

Тэгсет включает следующие модели:
- `kz` -- слово казахского языка
- `ru` -- слово русского языка
- `skz` -- слово казахского языка, транслитерированное на кириллическом алфавите
- `ambig` -- слово, существующее в обоих языках (не включает омофоны)
- `other` -- слово из другого языка
- `mixed_kz-ru` -- казахский корень/основа с русским суффиксом/окончанием
- `mixed_ru-kz` -- русский корень с казахским суффиксом/окончанием
- `univ` -- пунктуация и маски, которые мы добавили на этапе предобработки

На основе потокенной разметки мы отобрали тексты, которые содержат теги:

a) и `kz` и `ru`

b) и `skz` и `ru`

c) `mixed_kz-ru` или

d) `mixed_ru-kz`.

После этого этапа у нас осталось только 80 тыс. текстов, что составляет около 8% от данных, с которых мы начали этап.

:warning: Важно: модель плохо выявляет minority-теги, т.е. все, кроме `kz`, `ru` и `univ`, поскольку в обучающем датасете большинство токенов относилось именно к этим тегам.

- `4__metrics.ipynb`

В этом блокноте считаются метрики для корпуса:
- Average Code-Mixing Index (CMI Avg)
- Average switch-points (SP Avg)
- Multilingual Index (M-index)
- Probability of Switching (I-index)
- Burstiness
- Language Entropy (LE)
- Span Entropy (SE)

# Project structure
This repository includes notebooks for text processing, experiments and analysis, and notebooks for model training.

## Text processing

The following notebooks represent the four stages the data went through to form the KazRusCSWCorpus (not including the zero-th step which is data scraping): 
1) preprocessing, 
2) filtering out irrelevant texts, 
3) token annotation and elicitaion of code-mixed data and 
4) code-mixing metrics calculation

- `1__data_prep.ipynb` + `preprocessing.py` 

The notebook contains code for text preprocessing. It assumes a certain structure in the data so the code is not really reusable. Still, it provides an insight into how the raw texts are cleaned and sensitive data is masked. For preprocessing, the notebook uses the function from `preprocessing.py`.

Basically, during this stage, we first filter out the comments that contain a dividing line which usually indicates that the comment is just a message that is written first in one language and then translated into another. Then, we get rid of comments that do not contain any cyrillic letters, mask emoji, card and phone numbers, links, hashtags, emails and mentions. We replace newlines with `\\n` since we treat it as a meaningful token (like a full stop, for instance), and swap all the other whitespaces with a regular space. Then we filter out comments that do not contain at least three words written in cyrillic characters. Finally we drop duplicate texts and assign a uuid to each comment.

In our case, we started out with 1,590,529 texts and after preprocessing we were left with 1,122,792 texts.

- `2__filtering.ipynb`
In this notebook, we filter out texts that are not mainly in either Kazakh or Russian. We do that by running a [GlotLID model](https://github.com/cisnlp/GlotLID) on each document and if the most likely language is not Kazakh or Russian we get rid of it.

The decision to use this model specifically is based on the analysis that was carried out in the two notebooks [here](colab_notebooks/experiments)

After this step we were left with 1,009,159 texts.

- `3__token_annot.ipynb`
After filtering out irrelevant texts, we provide each document with token-level annotation using our [mBERT-based model](https://huggingface.co/liminovna/KazRusCSW-mbert) (also see [notebook](colab_notebooks/model_training) for the training details).

The tagset includes the following tags:
- `kz` -- Kazakh word
- `ru` -- Russian word
- `skz` -- Kazakh word transliterated into Cyrillic alphabeth
- `ambig` -- word that exists in both languages (context sensitive)
- `other` -- word from some other language
- `mixed_kz-ru` -- Kazakh root with Russian inflection
- `mixed_ru-kz` -- Russian root with Kazakh inflection
- `univ` -- punctuation and masks we added at the preprocessing stage

Based on the token-level annotation, we keep only those documents that include either

a) both `kz` and `ru`

b) both `skz` and `ru`

c) `mixed_kz-ru` or

d) `mixed_ru-kz`.

After this step we got to keep only 80k documents, which is about 8% of the data we started this stage with.

:warning: Note: the model does poorly at distinguishing the minority tags, which is all of them except `kz`, `ru` and `univ`, since the vast majority of the tokens in the trainig dataset had these tags. 

- `4__metrics.ipynb`

In this notebook, we calculate the following code-mixing metrics:
- Average Code-Mixing Index (CMI Avg)
- Average switch-points (SP Avg)
- Multilingual Index (M-index)
- Probability of Switching (I-index)
- Burstiness
- Language Entropy (LE)
- Span Entropy (SE)
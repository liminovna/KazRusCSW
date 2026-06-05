# KazRusCSW

:earth_americas: For English version see [below](#overview)

## Описание
Это репозиторий проекта KazRusCSW, выполненного в рамках магистерской диссертации по направлению "Компьютерная лингвистика" в НИУ ВШЭ. 
Конечная цель проекта -- собрать и автоматически разметить тексты с казахско-русским переключением кодов.

В процессе работы было решено большое количество задач:
- написан скрипт для сбора материалов с YouTube через YouTube Data API
- разработан пайплайн для предобработки текстов
- вручную размечены 10 тыс. текстов на уровне документа
- проанализированы методы фильтрации текстов на "нерелевантных" языках
- проанализированы методы выявления текстов с переключением кодов
- вручную размечены 3 тыс. текстов на уровне слов
- обучено 3 модели для потокенной классификации
- обработано более 1 млн. текстов, из которых 80 тыс. попали в корпус как потенциально написанные на двух языках -- казахском и русском
- проведен лингвистический анализ корпуса

В этом репозитории собраны основные блокноты и функции. Золотой стандарт с потокенной разметкой и модель для разметки выложена на HF: https://huggingface.co/collections/liminovna/kazruscsw.

❗ проект (и репозитории на github и huggingface) все еще в разработке, поэтому ссылка на текст работы и сам корпус будут опубликованы позже.

## Overview
This repository contains code for the KazRusCSW project, which is carried out as part of a master's thesis in the Computational Linguistics program at HSE University, Moscow. The aim of this project is to gather and provide token-level annotation for documents containing Kazakh-Russian code-switching.

To accomlish this, we have solved numerous tasks:
- wrote a script for data scraping from YouTube using YouTube Data API
- developed a pipeline for text preprocessing
- manually annotated 10k documents at the document level
- analyzed methods for filtering out texts in "irrelevant" languages
- analyzed methods for eliciting texts that contain code-switching
- manually annotated 3k documents at the token level
- trained (or rather finetuned?) 3 models for token-level annotation
- processed over 1 million documents, of which 80k were included into the corpus as they are assumed to contain Kazakh-Russian code-switching
- performed a linguistic analysis of the corpus data

The repository contains notebooks for processing and analysis and supplementary functions (in the .py files). The golden data with token-level annotation and a model for token-level annotation are available on HF: https://huggingface.co/collections/liminovna/kazruscsw.

❗ The project is still in progress, the paper and the corpus itself will be published later.

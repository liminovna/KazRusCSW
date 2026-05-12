from datetime import datetime
import pandas as pd
import os

import shortuuid
import emoji
import re


def save_data(df: pd.DataFrame, name: str=None, save_to_path: str='') -> None:
    """
    Сохраняет датасет в csv-формате с указанным наименованием {NAME}__cleaned.csv.

    Arguments:
        df: датафрейм, который нужно сохранить.
        name: название источника данных (в данном случае канала) или любая другая строка, которая поможет понять, какие данные лежат в файле.
        save_to_path: папка, в которую нужно сохранить файл
    """

    # если в датафрейме нет данных, то прерываем выполнение функции
    if len(df) == 0:
        print('DataFrame is empty!')
        return

    if save_to_path and not os.path.exists(save_to_path):
        os.mkdir(save_to_path)

    # текущее время
    ts = datetime.now()

    # название результирующего файла
    CSV_FILENAME = f'{name}__cleaned.csv'

    # сохраняем датасет без индекса
    df.to_csv(os.path.join(save_to_path, CSV_FILENAME), index=False)

    print(ts.strftime('%Y-%m-%d %H:%M:%S'), f'{CSV_FILENAME} cleaned data: {len(df)} rows', sep='\t')


cyr_alph = 'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя' # кириллический алфавит
# lat_alph = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz'
special_char = 'ӘәҒғҚқҢңӨөҰұҮүҺһІі' # специализированные символы казахской письменности
alph = ''.join(list(cyr_alph + special_char)) # расширенный кириллический алфавит

def clean_comment(text):
    """
    Почистить текст от эмодзи и лишних отступов, привести к нижнему регистру.
    Вернуть список токенов.
    """
    # возвращаем пустоту, если сообщение короче 5 знаков
    if len(text)<5:
        return ''

    # маскируем эмодзи
    text = emoji.replace_emoji(text, replace=' [EMOJI] ')

    # убираем или маскируем # Имя Фамилия (или Фамилия Имя)
    # text = re.sub(r'[ЦЗЕІЪӘЖЩУҢҚЧВЫРХЁҰДКЛЭНҒШЯҮМЙСЬТПАҺФИБГОӨЮ]\w+ [ЦЗЕІЪӘЖЩУҢҚЧВЫРХЁҰДКЛЭНҒШЯҮМЙСЬТПАҺФИБГОӨЮ]\w+', '[NAME]', text) 
    
    text = re.sub(r'\u200b', '', text) # пробел

    text = re.sub(r'(\+?\d[ (-]*\d{3,4}[ )-]*\d{2,3}[ -]*\d{2}[ -]*\d{2}[ -]*|(?:\d{4} ?){3,4})', '[NUMBER]', text) # заменяем номера телефонов, карт и т.д.

    text = re.sub(r'\b[a-zA-Z0-9_]+@[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+\b', ' [EMAIL] ', text) # электронные почты

    text = re.sub(r'(?:^|\s)(?:@|id)\S+\b', ' [MENTION] ', text) # упоминания

    text = re.sub(r'(?:^|\s)#\S+\b', ' [HASHTAG] ', text) # хэштеги

    text = re.sub(r':\S+:', '[EMOJI]', text) # эмодзи

    text = re.sub(r'( *\[EMOJI\] *)+', ' [EMOJI] ', text) # эмодзи

    text = re.sub(r'https?://\S+\b', ' [LINK] ', text) # ссылки

    text = re.sub(r'\s+', r' ', text) # лишние пробелы и отступы

    text = re.sub(r'(\n)+', r'\\n', text) # перенос

    return text.strip()


def tokenize_comment(text):
    """
    Извлечь все слова -- последовательности кириллических символов -- из текста.
    """
    if not text:
        return []

    return re.findall(f'[{alph}]+', text)

class cleanupDataSet:
    """
    При инициализации класса данные вычитываются из указанного файла, и запускается предобработка: маскируются эмодзи, номера телефонов и карт, ссылки, хэштеги, упоминания. В результате возвращается датафрейм, где к исходному набору данных добавляются столбцы: 
        - uuid -- уникальный идентификатор комментария
        - clean_comment_text -- предобработанный текст комментария
        - comment_tokens -- список слов (последовательности кириллических символов)
        - comment_words -- comment_tokens, соединенные в одну строку 
    """
    def __init__(self, filename: str, comment_col: str):

        """
        filename: название файла, который нужно прочитать.
        comment_col: название колонки, в которой содержится текст комментария.
        """

        # читаем все данные чата из датафрейма
        self.filename = filename
        self.comment_col = comment_col
        self.df = pd.read_csv(filename)

        assert self.comment_col in self.df.columns, f'Specified column {comment_col} not found!'

        self.ts = datetime.now() # текущее время

        if 'Unnamed: 0' in self.df.columns:
            self.df.drop(columns=['Unnamed: 0'], inplace=True)

        self.init_nrows = len(self.df) # количество строк до чистки
        print('Rows before the cleanup:', self.init_nrows)

        self.start_cleanup()

        self.df['ts'] = self.ts # на всякий случай записываем дату и время обработки данных в столбец

        self.final_nrows = len(self.df) # количество строк после чистки

    def start_cleanup(self):

        # удаляем строки, где нет текста сообщения
        self.df.drop(self.df[self.df[self.comment_col].isna()].index, inplace=True)
        print('Rows after deleting comments with no text:', len(self.df))

        # удаляем строки, где есть разделительная линия -- признак текста, продублированного на обоих языках
        self.df.drop(self.df[self.df[self.comment_col].str.contains('———')].index, inplace=True)
        print('Rows after deleting comments after deleting comments potentially translated into two languages:', len(self.df))

        # удаляем строки, где нет кириллицы
        self.df.drop(self.df[~self.df[self.comment_col].str.contains('[а-яА-ЯёЁ]')].index, inplace=True)
        print('Rows after deleting comments with no Cyrillic characters:', len(self.df))

        # чистим текст
        print('Applying masking, normalizing whitespaces')
        self.df['clean_comment_text'] = self.df[self.comment_col].str.strip().apply(clean_comment)

        # возвращаем список токенов
        self.df['comment_tokens'] = self.df['clean_comment_text'].apply(tokenize_comment)

        # удаляем строки, где меньше 3 токенов
        self.df.drop(self.df[(self.df['comment_tokens'].apply(len) < 3)].index, inplace=True)
        print('Rows after deleting comments with less than 3 words with Cyrillic characters:', len(self.df))

        # соединяем токены в одну строку и удаляем повторяющиеся комментарии
        self.df['comment_words'] = self.df['comment_tokens'].apply(' '.join)
        self.df.drop_duplicates('comment_words', inplace=True)
        print('Rows after deleting duplicate comments:', len(self.df))

        # генерируем уникальный uuid для каждого комментария
        self.df['uuid'] = [shortuuid.uuid() for _ in range(len(self.df))]

        print('Cleanup finished!', '\nDUPLICATE IDS FOUND!' if len(self.df[self.df['uuid'].duplicated()]) else '')

def print_rows(df: pd.DataFrame, n_rows: int=None, seed: int=42):
    """
    Показать n_rows текстов из сэмла.
    Печатаются очищенные тексты в том виде, в каком они будут обрабатываться дальше.

    Arguments:
        df: датафрейм, из которого требуется напечатать примеры.
        n_rows: количество примеров, которые нужно напечатать.
        seed: random seed, которое участвует в формировании выборки.
    """
    if not n_rows:
        n_rows = len(df)
    print(*df.sample(n_rows, random_state=seed)['clean_comment_text'].to_list(), sep='\n' + '='*100 + '\n')
import re
from string import punctuation
from shared import cyr_alph, special_char, lat_alph

alph = ''.join(set(list(cyr_alph + special_char + lat_alph)))

def tokenize(text):
    patt = f'\[[A-Z]+\]|\s|\\\\n|\d+|[{punctuation},]|\w+|.'

    tokens, spans = [], []

    for m in re.finditer(patt, text):
        token = m.group()
        if token != ' ':
            tokens.append(token)
            spans.append(m.span())
    return tokens, spans
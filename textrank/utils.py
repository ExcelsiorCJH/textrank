import platform
from collections import Counter

import numpy as np

# tokenizer import
from konlpy.tag import Okt, Komoran, Hannanum, Kkma

if platform.system() == "Windows":
    try:
        from eunjeon import Mecab
    except:
        print("please install eunjeon module")
else:  # Ubuntu일 경우
    from konlpy.tag import Mecab

from types_ import *


def get_tokenizer(tokenizer_name):
    if tokenizer_name == "komoran":
        tokenizer = Komoran()
    elif tokenizer_name == "okt":
        tokenizer = Okt()
    elif tokenizer_name == "mecab":
        tokenizer = Mecab()
    elif tokenizer_name == "hannanum":
        tokenizer = Hannanum()
    elif tokenizer_name == "kkma":
        tokenizer = Kkma()
    else:
        tokenizer = Mecab()
    return tokenizer


def get_tokens(sents: List[List[str]], noun=False, tokenizer="mecab") -> List[List[str]]:

    tokenizer = get_tokenizer(tokenizer)

    if noun:
        return [tokenizer.nouns(sent) for sent in sents]

    return [[f"{word}/{pos}" for word, pos in tokenizer.pos(sent)] for sent in sents]


def get_vocab(corpus: List[List[str]], min_count=2, min_len=2) -> List[str] and Dict:

    counter = Counter(token for tokens in corpus for token in tokens)
    counter = {
        w: c for w, c in counter.items() if c >= min_count and len(w.split("/")[0]) >= min_len
    }

    idx_vocab = [w for w, _ in sorted(counter.items(), key=lambda x: -x[1])]
    vocab_idx = {vocab: idx for idx, vocab in enumerate(idx_vocab)}
    return idx_vocab, vocab_idx

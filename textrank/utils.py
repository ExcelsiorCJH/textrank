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

from utils.types_ import *


def get_tokenizer(tokenizer_name: str = "mecab") -> "tokenizer":
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
        return None

    return tokenizer


def get_tokens(sent: List[str], noun: bool = False, tokenizer: str = "mecab") -> List[str]:
    tokenizer = get_tokenizer(tokenizer)

    if tokenizer:
        if noun:
            nouns = tokenizer.nouns(sent)
            nouns = [word for word in nouns if len(word) > 1]
            return nouns
        return tokenizer.morphs(sent)
    else:
        return sent.split()

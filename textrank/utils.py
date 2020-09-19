import platform
from collections import Counter

import numpy as np
import spacy

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


class LemmaTokenizer(object):
    def __init__(self, use_pos=False):
        self.spacynlp = spacy.load("en_core_web_sm")
        self.pos_filter = False
        if use_pos:
            self.pos_filter = ["NOUN", "ADJ"]

    def __call__(self, doc):
        nlpdoc = self.spacynlp(doc)
        if self.pos_filter:
            nlpdoc = [
                token.lemma_
                for token in nlpdoc
                if ((len(token.lemma_) > 1) or (token.lemma_.isalnum()))
                and (token.pos_ in self.pos_filter)
            ]
        else:
            nlpdoc = [
                token.lemma_
                for token in nlpdoc
                if (len(token.lemma_) > 1) or (token.lemma_.isalnum())
            ]
        return nlpdoc


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

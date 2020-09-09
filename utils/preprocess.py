import os
import re
import numpy as np
import pandas as pd
import spacy

from glob import glob
from pdfminer.high_level import extract_text
from .types_ import *

# ignore warning
import warnings

warnings.filterwarnings(action="ignore")

NLP = spacy.load("en_core_web_sm")


def pdf_to_text(file_path: str) -> List[str]:
    """
    PDF to Text method
    ===================
    Arguments
    ---------
    file_path : str
        path of pdf file
    
    Returns
    -------
    sentences : list of str
        preprocessed sentences
    """
    # 1. pdf -> text
    text = extract_text(file_path)
    # 2. sentence split
    doc = NLP(text)
    sentences = [sent.text for sent in doc.sents]
    # 3. text cleaning
    sentences = [clean_text(sent) for sent in sentences]
    avg_sent_len = np.mean([len(sent) for sent in sentences])

    # 3. extract sentences
    sentences = [sent for sent in sentences if len(sent) > avg_sent_len]
    return sentences


def clean_text(text):
    """기사 내용 전처리 함수
    Args:
        - text: str 형태의 텍스트
    Return:
        - text: 전처리된 텍스트"""
    # Common
    # 개행문자 제거
    text = re.sub("\n", " ", text)
    text = re.sub("\v", " ", text)
    text = re.sub("\f", " ", text)
    # E-mail 제거#
    text = re.sub("([\w\d.]+@[\w\d.]+)", "", text)
    text = re.sub("([\w\d.]+@)", "", text)
    # 괄호 안 제거#
    text = re.sub("<[\w\s\d‘’=/·~:&,`]+>", "", text)
    text = re.sub("\([\w\s\d‘’=/·~:&,`]+\)", "", text)
    text = re.sub("\[[\w\s\d‘’=/·~:&,`]+\]", "", text)
    text = re.sub("【[\w\s\d‘’=/·~:&,`]+】", "", text)
    # 전화번호 제거#
    text = re.sub("(\d{2,3})-(\d{3,4}-\d{4})", "", text)  # 전화번호
    text = re.sub("(\d{3,4}-\d{4})", "", text)  # 전화번호
    # 홈페이지 주소 제거#
    text = re.sub("(www.\w.+)", "", text)
    text = re.sub("(.\w+.com)", "", text)
    text = re.sub("(.\w+.co.kr)", "", text)
    text = re.sub("(.\w+.go.kr)", "", text)
    # 기자 이름 제거#
    text = re.sub("/\w+[=·\w@]+\w+\s[=·\w@]+", "", text)
    text = re.sub("\w{2,4}\s기자", "", text)
    # 한자 제거#
    text = re.sub("[\u2E80-\u2EFF\u3400-\u4DBF\u4E00-\u9FBF\uF900]+", "", text)
    # 특수기호 제거#
    text = re.sub("[◇#/▶▲◆■●△①②③★○◎▽=▷☞◀ⓒ□?㈜♠☎]", "", text)
    # 따옴표 제거#
    text = re.sub("[\"'”“‘’]", "", text)
    # 2안_숫자제거#
    # text = regex.sub('[0-9]+',"",text)
    text = " ".join(text.split())
    return text

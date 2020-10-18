import numpy as np

import gensim
from .pagerank import pagerank
from .graph import sent_graph, word_graph
from utils.types_ import *


class TextRank:
    def __init__(
        self,
        min_count: int = 2,
        min_sim: float = 0.3,
        language: str = "ko",
        tokenizer: str = "mecab",
        model: str = "gensim",
        noun: bool = False,
        vectorizer: str = "tfidf",
        similarity: str = "cosine",
        df: float = 0.85,
        max_iter: int = 50,
        method: str = "iterative",
        stopwords: List[str] = ["뉴스", "기자"],
    ):
        """
        TextRank Class
        ==============

        Arguments
        ---------
        min_count : int
            Minumum frequency of words will be used to construct sentence graph
        min_sim : float
            Minimum similarity of sents or words will be used to construct sentence graph
        language: str
            available language = ['ko', 'en']
        tokenizer : str
            Tokenizer for korean, default is mecab
        model : str
            if language is 'en' then model options are ['gensim', 'textrank']
        noun : bool
            option for using just nouns, default is False but True when use keyword extraction
        vectorizer: str
            available vectorizer = ['tfidf', 'count']
        similarity : str
            available similarity = ['cosine', 'textrank']
        df : float
            PageRank damping factor, default is 0.85
        max_iter : int
            Number of PageRank iterations
        method : str
            available method = ['iterative', 'algebraic']
        stopwords: list of str
            Stopwords for Korean
        """

        self.language = language
        self.tokenizer = tokenizer
        self.model = model
        self.min_count = min_count
        self.min_sim = min_sim
        self.noun = noun
        self.vectorizer = vectorizer
        self.similarity = similarity
        self.df = df
        self.max_iter = max_iter
        self.method = method
        self.stopwords = stopwords

    def _sent_textrank(self, sents: List[str]) -> None:
        G = sent_graph(
            sents=sents,
            language=self.language,
            min_count=self.min_count,
            min_sim=self.min_sim,
            tokenizer=self.tokenizer,
            noun=self.noun,
            similarity=self.similarity,
            vectorizer=self.vectorizer,
            stopwords=self.stopwords,
        )
        self.R = pagerank(G, self.df, self.max_iter, self.method)
        return None

    def _word_textrank(self, sents: List[str]) -> None:
        G, _, self.idx_vocab = word_graph(
            sents=sents,
            language=self.language,
            min_count=self.min_count,
            min_sim=self.min_sim,
            tokenizer=self.tokenizer,
            noun=True,
            vectorizer=self.vectorizer,
            stopwords=self.stopwords,
        )
        self.wr = pagerank(G, self.df, self.max_iter, self.method)
        return None

    def summarize(self, sents: List[str], topk: int = 3) -> List[str]:
        if self.language == "en" and self.model == "gensim":
            keysents = gensim.summarization.summarize(" ".join(sents), split=True)
            keysents = keysents[:topk]
        else:
            self._sent_textrank(sents)
            idxs = self.R.argsort()[-topk:]
            # keysents = [(idx, sents[idx]) for idx in sorted(idxs)]
            keysents = [sents[idx] for idx in sorted(idxs)]
        return keysents

    def keywords(self, sents: List[str], topk: int = 10) -> List[str]:
        if self.language == "en" and self.model == "gensim":
            keywords = gensim.summarization.keywords(" ".join(sents), split=True, lemmatize=True)
            keywords = keywords[:topk]
        else:
            self._word_textrank(sents)
            idxs = self.wr.argsort()[-topk:]
            keywords = [self.idx_vocab[idx] for idx in reversed(idxs)]
        return keywords

import numpy as np

from functools import partial
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer

from .utils import get_tokens
from .types_ import *


def vectorize_sents(
    sents: List[str],
    stopwords: List[str] = None,
    min_count: int = 2,
    tokenizer: str = "mecab",
    noun: bool = False,
) -> Union[np.ndarray, Dict, Dict]:

    vectorizer = CountVectorizer(
        stop_words=stopwords,
        tokenizer=partial(get_tokens, noun=noun, tokenizer=tokenizer),
        min_df=min_count,
    )

    vec = vectorizer.fit_transform(sents)
    vocab_idx = vectorizer.vocabulary_
    idx_vocab = {idx: vocab for vocab, idx in vocab_idx.items()}
    return vec, vocab_idx, idx_vocab


def similarity_matrix(x: np.ndarray, min_sim: float = 0.3, min_length: int = 1) -> np.ndarray:
    """
    $$
    sim(s_1, s_2) = 
    \frac{\vert \{ w_k \vert w_k \in S_1 \& w_k \in S_2 \} \vert}
    {log \vert S_1 \vert + log \vert S_2 \vert}
    $$
    """

    # binary csr_matrix
    numerators = (x > 0) * 1

    # denominator
    min_length = 1
    denominators = np.asarray(x.sum(axis=1))
    denominators[np.where(denominators <= min_length)] = 10000
    denominators = np.log(denominators)
    denom_log1 = np.matmul(denominators, np.ones(denominators.shape).T)
    denom_log2 = np.matmul(np.ones(denominators.shape), denominators.T)

    sim_mat = np.dot(numerators, numerators.T)
    sim_mat = sim_mat / (denom_log1 + denom_log2)
    sim_mat[np.where(sim_mat <= min_sim)] = 0

    return sim_mat


def cosine_similarity_matrix(x: np.ndarray, min_sim: float = 0.3) -> np.ndarray:
    sim_mat = 1 - pairwise_distances(x, metric="cosine")
    sim_mat[np.where(sim_mat <= min_sim)] = 0

    return sim_mat


def word_similarity_matrix(x, min_sim=0.3) -> np.ndarray:
    sim_mat = 1 - pairwise_distances(x.T, metric="cosine")
    sim_mat[np.where(sim_mat <= min_sim)] = 0

    return sim_mat


def sent_graph(
    sents: List[str],
    min_count: int = 2,
    min_sim: float = 0.3,
    tokenizer: str = "mecab",
    noun: bool = False,
    similarity: bool = None,
    stopwords: List[str] = ["뉴스", "그리고"],
) -> np.ndarray:

    mat, vocab_idx, idx_vocab = vectorize_sents(
        sents, stopwords, min_count=min_count, tokenizer=tokenizer, noun=noun
    )

    if similarity == "cosine":
        mat = cosine_similarity_matrix(mat, min_sim=min_sim)
    else:
        mat = similarity_matrix(mat, min_sim=min_sim)

    return mat


def word_graph(
    sents: List[str],
    min_count: int = 2,
    min_sim: float = 0.3,
    tokenizer: str = "mecab",
    noun: bool = True,
    stopwords: List[str] = ["뉴스", "그리고", "기자"],
):

    mat, vocab_idx, idx_vocab = vectorize_sents(
        sents, stopwords, min_count=min_count, tokenizer=tokenizer, noun=noun
    )

    mat = word_similarity_matrix(mat, min_sim=min_sim)

    return mat, vocab_idx, idx_vocab

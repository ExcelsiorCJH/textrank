import argparse

from textrank import TextRank
from utils import get_data
from utils.types_ import *

from nltk.corpus import stopwords  # for english stopwords

######### Install nltk stopwords #########
# $ pip install nltk
# $ python
# >>> import nltk
# >>> nltk.download('stopwords')
##########################################

# Parser
parser = argparse.ArgumentParser(description="Extractive Summarization using TextRank")

# mode ['sentence', 'keyword']
parser.add_argument("--mode", type=str, default="sentences", help="Select the mode to use")
# model
parser.add_argument(
    "--min_count",
    type=int,
    default=2,
    help="Minumum frequency of words will be used to construct sentence graph",
)
parser.add_argument(
    "--min_sim",
    type=float,
    default=0.3,
    help="Minimum similarity of sents or words will be used to construct sentence graph",
)
parser.add_argument(
    "--tokenizer", type=str, default="mecab", help="Tokenizer for korean, default is mecab"
)
parser.add_argument("--noun", type=bool, default=False, help="option for using just nouns")
parser.add_argument(
    "--similarity",
    type=str,
    default="cosine",
    help="similarity type to use choose cosine or textrank",
)
parser.add_argument("--df", type=float, default=0.85, help="PageRank damping factor")
parser.add_argument("--max_iter", type=int, default=50, help="Number of PageRank iterations")
parser.add_argument("--method", type=str, default="iterative", help="Number of PageRank iterations")
parser.add_argument("--topk", type=int, default=3, help="Number of sentences/words to summarize")
# data
parser.add_argument("--data_type", type=str, default="cnndm", help="Data type to load")

args = parser.parse_args()


if __name__ == "__main__":
    if args.data_type == "cnndm":
        sents = get_data("data/train.json", "cnndm")
        args.tokenizer = None
        # stopwords of english
        stopwords = stopwords.words("english")
        stopwords += [",", "-", ":", ";", "!", "?", "'", '"']
    else:
        sents = get_data("data/sents.txt", "news")
        # stopwords of korean
        stopwords = ["뉴스", "기자", "그리고", "연합뉴스"]

    # initialize Textrank
    textrank = TextRank(
        min_count=args.min_count,
        min_sim=args.min_sim,
        tokenizer=args.tokenizer,
        noun=args.noun,
        similarity=args.similarity,
        df=args.df,
        max_iter=args.max_iter,
        method=args.method,
        stopwords=stopwords,
    )

    # extraction setences or keywords
    if args.mode == "sentences":
        results = textrank.summarize(sents, topk=args.topk)
        results = [sent for _, sent in results]
        results = "\n".join(results)
    else:
        args.mode = "words"
        results = textrank.keywords(sents, topk=args.topk)

    print(f"{args.mode}")
    print("=" * 20)
    print(f"{results}")

# TextRank


Code for [TextRank: Brining Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

Some codes and spliting files are inspired by [lovit/textrank](https://github.com/lovit/textrank)


## Usage

```python
from textrank import TextRank

sents = ['list of str form 1', 'list of str form 2', ...]

stopwords = ["뉴스", "기자", ...]
textrank = TextRank(tokenizer="mecab", stopwords=stopwords, method="iterative")

# sentences extraction
keysents = textrank.summarize(sents)

# keywords extraction
keywords = textrank.keywords(sents)
```

## References

- Github: https://github.com/lovit/textrank
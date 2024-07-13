
# 1. From Text to tokens


Text -(tokenize)-> N token idx -(project)-> R^{N*E}

1. Character Tokenization
   1. Ralated paper: [Finding Functoin in Form:Compositional Open Vocabulary Word Representation]()
   2. Example: 
```python
import torch 
from torch.nn import functional as F

str_ = "Tokenizing text is a core task in NLP"
character_tokenized_text = list(str_)
token2idx_map = {ele: idx for idx, ele in enumerate(sorted(set(character_tokenized_text)))}
one_hot_encoding = F.one_hot(
    torch.tensor([token2idx_map[i] for i in character_tokenized_text]), 
    num_classes=len(token2idx_map)
)
```
2. Word Tokenization
   1. Related papers: 
      1. [2013: Efficient Estimation of Word Representations in Vector Space](https://www.semanticscholar.org/reader/f6b51c8753a871dc94ff32152c00c01e94f90f09) (skip-grams CBOW)
      2. [2014: GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162.pdf)
         1. [github: stanfordnlp/GloVe glove.c](https://github.com/stanfordnlp/GloVe/blob/master/src/glove.c)
         2. It combines `local contextual information` and `global corpus statistical information` to learn the vector representation of words
   2. Inferiority
      1. large vocabulary size -> large spare onehot matrix (N) -> large Embedding weight Matrix(N * emb_dim)
      2. reduce vocabulary -> out of vocabulary problem -> use `UNK`
   3. Example: 
```python
import torch 
from torch import nn
from torch.nn import functional as F

str_ = "Tokenizing text is a core task in NLP"
character_tokenized_text = str_.split(' ')
token2idx_map = {ele: idx for idx, ele in enumerate(sorted(set(character_tokenized_text)))}
idx_tensor = torch.tensor([token2idx_map[i] for i in character_tokenized_text]).long()
one_hot_encoding = F.one_hot(
    idx_tensor, 
    num_classes=len(token2idx_map)
)
# onehot -> embedding weight -> embedding result
emb_ = nn.Embedding(num_embeddings=len(token2idx_map), embedding_dim=4)
emb_res = emb_(idx_tensor)
```
3. Subword Tokenization
    1. Related papers: 
       1. [2016: Enriching Word Vectors with Subword Information](https://www.semanticscholar.org/reader/e2dba792360873aef125572812f3673b1a85d850)
       2. [2016: Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://www.semanticscholar.org/reader/c6850869aa5e78a107c378d2e8bfa39633158c0c)
          1. `WordPiece Model`  [bert tokenization.py](https://github.com/google-research/bert/blob/master/tokenization.py)
       3. [2019: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.semanticscholar.org/reader/df2b0e26d0599ce3e70df8a9da02e51594e0e992)
       4. [2021: Fast WordPiece Tokenization](https://arxiv.org/pdf/2012.15524)
    2. Byte Pair Encoding(BPE) - GPT2
    3. Example: 
```python
from modelscope import AutoTokenizer

str_ = "Tokenizing text is a core task in NLP"

# WordPiece Model
model_ckpt = "AI-ModelScope/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
idx_ = tokenizer(str_, return_tensors="pt")['input_ids']
token = tokenizer.convert_ids_to_tokens(idx_[0])
# ['[CLS]', 'token', '##izing', 'text', 'is', 'a', 'core', 'task', 'in', 'nl', '##p', '[SEP]']

# Byte Pair Encoding(BPE)
model_ckpt = 'iiBcai/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
idx_ = tokenizer(str_, return_tensors="pt")['input_ids']
token = tokenizer.convert_ids_to_tokens(idx_[0])
# ['Token', 'izing', 'Ġtext', 'Ġis', 'Ġa', 'Ġcore', 'Ġtask', 'Ġin', 'ĠN', 'LP']
```

# Reference

- Natural language Processing with Transformers(Building Language Applications with Hugging Face)
- Build a Large Language Model (From Scratch)
  - [Github: LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
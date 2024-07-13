# python3
# Create Date: 20240704
# Author: Scc_hy
# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# Func: Byte Pair Encoding(BPE)
# ===============================================================================
import json 
import os 
import regex as re
import requests
from tqdm.auto import tqdm 
from functools import lru_cache
from typing import Dict, List, Tuple


@lru_cache
def bytes_to_unicode():
    """
    return list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in you vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent converage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on. 
    """
    # ASCII 0~31(control char) | 33 ~ 126(print char) | 127 delete
    # 128: 欧元符号（€）- 尽管欧元符号是较新引入的，但许多现代字体和编码系统已经支持它。
    # 129 到 159: 这些通常是保留给特定语言或符号的扩展字符，不同语言和编码页可能有不同的映射。
    # 160: 非断行空格（Non-breaking space）- 这个空格不会在文本换行时被分割。
    # 173: 连续的连字符
    # use 161~172 + 174~255
    bs = list(range(ord("!"), ord("~")+1)) + \
            list(range(ord("¡"), ord("¬")+1)) + \
            list(range(ord("®"), ord("ÿ")+1))
    
    cs = bs[:]
    n = 0 
    # 将特殊的ASCII位，用256后的char
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pair(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length string)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder: Dict, bpe_merges: List[Tuple], errors: str='replace'):
        self.encoder = encoder 
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def bpe(self, token):
        # 已经Byte Pair Encoding的token
        if token in self.cache:
            return self.cache[token]        
        
        word = tuple(token)
        pairs = get_pair(word)
        # only one char
        if not pairs:
            return token

        while True:
            # word pair中排序最小的并进行组合, 执行直到没有再需要组合单词为止
            bigram = min(
                pairs, key=lambda p: self.bpe_ranks.get(p, float('inf'))
            )
            if bigram not in self.bpe_ranks:
                break 
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j 
                except ValueError:
                    new_word.extend(word[i:])
                    break 
                
                # 合并
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pair(word)
        
        word = ' '.join(word)
        self.cache[token] = word 
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text): # 拆分
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # bpe组合(chair-pair查bpe_merges) -> encoder Map中查表 -> int
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text 


def get_encoder(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding='utf-8') as f:
        bpe_data = f.read()
    
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    subdir = 'gpt2_model'
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    subdir = subdir.replace('\\', '/')  # needed for Windows
    for filename in ['encoder.json', 'vocab.bpe']:
        r = requests.get(
            "https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, 
            stream=True
        )
        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers['content-length'])
            chunk_size = 1000
            with tqdm(ncols=100, desc='Fetching' + filename, total=file_size, unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


if __name__ == '__main__':
    str_ = 'preference'
    word = tuple(str_)
    pairs = get_pair(word)
    print(pairs)
    print(get_pair('a'))
    encoder = get_encoder(model_name="gpt2_model", models_dir=".")
    bigram = min(
        pairs, key=lambda p: encoder.bpe_ranks.get(p, float('inf'))
    )
    first, second = bigram
    i = 1
    # the index of first-char which start from i 
    # Return first index of value.
    j = word.index(first, i)
    print(f"{word[i:j]=}")
    
    i = 0
    new_word = []
    while i < len(word):
        try:
            j = word.index(first, i)
            new_word.extend(word[i:j])
            i = j 
        except ValueError:
            new_word.extend(word[i:])
            break 

        if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    new_word = tuple(new_word)
    print([encoder.bpe_ranks.get(p, float('inf')) for p in pairs])
    print(f'{bigram=} {i=} {j=} {new_word=}')

    res_ = encoder.encode(str_)
    print(encoder.bpe(str_))
    print(re.findall(encoder.pat, str_))
    print([encoder.encoder[bpe_token] for bpe_token in encoder.bpe(str_).split(' ')])


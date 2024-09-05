# python3
# Create Date: 2024-09-03
# Author: Scc_hy
# Func:  tokenizer 
# ====================================================================================================

from transformers import AutoTokenizer
from datasets import load_dataset, DownloadConfig
import os 
import psutil
from tqdm.auto import tqdm
import keyword
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
try:
    CUR_DIR = os.path.dirname(__file__)
except:
    CUR_DIR = os.getcwd()

def tok_list(tokenizer, str_):
    ipt_ids = tokenizer(str_, add_special_tokens=False)['input_ids']
    return [tokenizer.decode(tok) for tok in ipt_ids]


def safe_mkdir(path):
    if os.path.exists(path):
        return None 
    os.makedirs(path)


def model_download():
    TOKEN = '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'
    data_name = 'google-t5/t5-base'
    final_out_path = '/home/scc/sccWork/devData/myData/hf_models/t5-base'
    safe_mkdir(final_out_path)
    os.system(f"""
    export HF_ENDPOINT=https://hf-mirror.com && \
    huggingface-cli download --resume-download {data_name} --local-dir-use-symlinks False \
    --repo-type model \
    --local-dir {final_out_path} \
    --cache-dir {final_out_path}/cache
    """)
    os.system(f'rm -rf {final_out_path}/cache')
    
    data_name = 'openai-community/gpt2'
    final_out_path = '/home/scc/sccWork/devData/myData/hf_models/gpt2'
    safe_mkdir(final_out_path)
    os.system(f"""
    export HF_ENDPOINT=https://hf-mirror.com && \
    huggingface-cli download --resume-download {data_name} --local-dir-use-symlinks False \
    --repo-type model \
    --local-dir {final_out_path} \
    --cache-dir {final_out_path}/cache   \
    """)
    os.system(f'rm -rf {final_out_path}/cache')


def tok_test():    
    python_code = r"""def say_hello():
    print("Hello, World!")
    # Print it
    say_hello()
    """
    gpt2_p = '/home/scc/sccWork/devData/myData/hf_models/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(gpt2_p)
    t5_p = '/home/scc/sccWork/devData/myData/hf_models/t5-base'
    tokenizer_T5 = AutoTokenizer.from_pretrained(t5_p)
    
    print(tokenizer(python_code).tokens())
    print(tokenizer_T5(python_code).tokens())

    print(f'T5 tokens for "sex": {tok_list(tokenizer_T5, "sex")}')
    print(f'gpt2 tokens for "sex": {tok_list(tokenizer, "sex")}')
    
    print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(python_code))
    
    # 
    tokens = sorted(tokenizer.vocab.items(), key=lambda x: len(x[0]), reverse=True)
    a = [t for t, _ in tokens[:8]]
    print('last 8 tokenizer.vocab.items()', f'{tokenizer.convert_tokens_to_string(a)}')
    
    print('\n', '--'*25)
    tokens = sorted(tokenizer.vocab.items(), key=lambda x: x[1], reverse=True)
    a = [t for t, _ in tokens[:12]]
    print(f'{tokenizer.convert_tokens_to_string(a)}')


# *********************************************************************************************************
# Training a Tokenizer
def batch_iterator(iter_d, length, batch_size=10):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_d)['content'] for _ in range(batch_size)]


def train_vocab_check_keyword(new_tokenizer):
    print(f'There are in total {len(keyword.kwlist)} Python keywords.')
    for keyw in keyword.kwlist:
        if keyw not in new_tokenizer.vocab:
            print(f'No, keyword `{keyw}` is not in the vocabulary')


def train_new_tokenizer(length=200000, vocab_size=32768):
    """
    
    """
    python_code = r"""def say_hello():
    print("Hello, World!")
    # Print it
    say_hello()
    """
    # prepaer base vocab
    byte_to_unicode_map = bytes_to_unicode()
    unicode_to_byte_map = dict((v, k) for k, v in byte_to_unicode_map.items())
    base_vocab = list(unicode_to_byte_map.keys())

    # load tokeinzer 
    gpt2_p = '/home/scc/sccWork/devData/myData/hf_models/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(gpt2_p)
    
    # load data
    
    dataset = load_dataset(
        path='/home/scc/sccWork/devData/myData/train_data/codeparrot', 
        split="train",
        streaming=True
    )
    iter_dataset = iter(dataset)
    
    # training
    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(iter_dataset, length),
        vocab_size=vocab_size,
        initial_alphabet=base_vocab
    )
    
    # observing 
    tokens = sorted(new_tokenizer.vocab.items(), key=lambda x: x[1], reverse=False)
    a = [t for t, _ in tokens[257:280]]
    print( tokenizer.convert_tokens_to_string(a) )
    a = [t for t, _ in tokens[-12:]]
    print( tokenizer.convert_tokens_to_string(a) )
    print(new_tokenizer(python_code).tokens())
    
    # check 
    train_vocab_check_keyword(new_tokenizer)    
    # CUR_DIR = '/home/scc/sccWork/myGitHub/simpleLLM/src/code_auto_complete'
    save_directory = f"{CUR_DIR}/cac_tokenizer"
    new_tokenizer.save_pretrained(save_directory)
    return new_tokenizer


if __name__ == '__main__':
    # model_download()
    # tok_test()
    train_new_tokenizer(length=200000, vocab_size=32768)

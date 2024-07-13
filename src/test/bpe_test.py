
import tiktoken
from os.path import dirname
from modelscope import AutoTokenizer
import sys 
sys.path.append(dirname(dirname(__file__)))
from tokenizer.bpe import get_encoder, download_vocab
  

def tiktoken_bpe_test():
    print("tiktoken_bpe_test()")
    tik_tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, world. Is this-- a test?"
    integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print("encoder = \n", integers)
    strings = tik_tokenizer.decode(integers)
    print("\ndecoder = \n", strings)


def hf_bpe_test():
    print("hf_bpe_test()")
    model_ckpt = 'iiBcai/gpt2'
    text = "Hello, world. Is this-- a test?"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    idx_ = tokenizer(text, return_tensors="pt")['input_ids']
    print("encoder = \n", idx_[0])
    token = tokenizer.convert_ids_to_tokens(idx_[0])
    print("\ndecoder = \n", token)


def self_bpe_test():
    print("self_bpe_test()")
    text = "Hello, world. Is this-- a test?"
    # download_vocab()
    orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")
    # bpe组合(chair-pair查bpe_merges) -> encoder Map中查表 -> int
    integers = orig_tokenizer.encode(text)
    print("encoder = \n", integers)
    strings = orig_tokenizer.decode(integers)
    print("\ndecoder = \n", strings)
 

if __name__ == '__main__':
    tiktoken_bpe_test()
    # hf_bpe_test()
    # self_bpe_test()


# python3
# Create Date: 2024-09-05
# Author: Scc_hy
# Func:  dataLoader
# ====================================================================================================

import torch 
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


chars_per_token_doc = """ 
examples, total_characters, total_tokens = 500, 0, 0
for _, example in tqdm(zip(range(examples), iter(dataset)), total=examples):
    total_characters += len(example['content'])
    total_tokens += len(tokenizer(example['content']).tokens())

characters_per_token = total_characters / total_tokens
print(characters_per_token)

- OpenAI Official 
    - A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly ¾ of a word (so 100 tokens ~= 75 words)

- summary:
    - 英文上的效果基本差不太多。一个 token 大概占 0.75～0.8 个单词
    - 国内的模型在中文语料上特训之后，中文编码的效率显著高于英文的 ChatGPT 和 Llama。一个 token 大概占1.5 个汉字。
"""

class cacDataset(IterableDataset):
    def __init__(self, tokenizer, dataset, seq_length=1024, num_of_sequences=1024, chars_per_token=3.6, print_flag=False):
        """_summary_

        Args:
            tokenizer (AutoTokenizer): _description_
            dataset (_type_): _description_
            seq_length (int, optional): the number of tokens per sequence returned by the tokenizer. Defaults to 1024.
            num_of_sequences (int, optional): the number of (truncated) sequences we would like from our tokenizer. Defaults to 1024.
            chars_per_token (float, optional): tokens per characters. Defaults to 3.6.

            example:
                [xxxxx][xxxxx][xxxxx]
                seq_length&*chars_per_token = 5 
                num_of_sequences = 3
                input_characters = 3*5 = 15
        """
        self.print_flag = print_flag
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id 
        self.dataset = dataset
        self.seq_length = seq_length
        # The number of characters in the string input to our tokenizer
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        
    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True 
        while more_examples:
            buffer_, buffer_len = [], 0
            # concat sample
            while True:
                if buffer_len >= self.input_characters:
                    if self.print_flag:
                        print(
                            f'Buffer full: {buffer_len} >= {self.input_characters:.0f}'
                        )
                    break
                try:
                    if self.print_flag:
                        print(
                            f"Fill buffer: {buffer_len} < {self.input_characters:.0f}"
                        )
                    buffer_.append(next(iterator)['content'])
                    buffer_len += len(buffer_[-1])
                except StopIteration:
                    iterator = iter(self.dataset)
            
            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer_, truncation=False)
            for ipts in tokenized_inputs['input_ids']:
                all_token_ids.extend(ipts + [self.eos])
            
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i:i+self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)



if __name__ == '__main__':
    p_ = '/home/scc/sccWork/myGitHub/simpleLLM/src/code_auto_complete/cac_tokenizer'
    new_tokenizer = AutoTokenizer.from_pretrained(p_)
    dataset = load_dataset(
            path='/home/scc/sccWork/devData/myData/train_data/codeparrot', 
            split="train",
            streaming=True
        )
    shuffled_data = dataset.shuffle(buffer_size=100)
    cac_dataset = cacDataset(new_tokenizer, shuffled_data, num_of_sequences=10, print_flag=True)
    data_iter = iter(cac_dataset)
    len_ = [len(b) for _, b in zip(range(5), data_iter)]
    print(f'Lengths of the sequences: {len_}')


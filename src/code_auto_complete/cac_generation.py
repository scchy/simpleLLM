# python3
# Create Date: 2024-09-10
# Author: Scc_hy
# Func:  code auto complete test 
# ====================================================================================================
import os 
import re
from transformers import pipeline, set_seed, AutoTokenizer, GPT2LMHeadModel
try:
    CUR_DIR = os.path.dirname(__file__)
except Exception as e:
    CUR_DIR = "/home/scc/sccWork/myGitHub/simpleLLM/src/code_auto_complete"

p_ = f"{CUR_DIR}/cac_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(p_)
model_p = f"{CUR_DIR}/results/checkpoint-30000"
model = GPT2LMHeadModel.from_pretrained(model_p)
generation = pipeline("text-generation", model=model, tokenizer=tokenizer)


def first_block(str_):
    return re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif", str_)[0].rstrip()


def complete_code(pipe, prompt, max_length=64, num_completions=4, seed=1):
    set_seed(seed)
    gen_kwargs = {
        "temperature": 0.55, # 0.4,
        "top_p": 0.95,
        "top_k": 0,
        "num_beams": 1,
        "do_sample": True
    }
    code_gens = pipe(prompt, num_return_sequences=num_completions, max_length=max_length, **gen_kwargs)
    code_strings = []
    for c in code_gens:
        # print("=="*25)
        # print(c['generated_text'])
        g_c = first_block(c['generated_text'][len(prompt):]) 
        code_strings.append(g_c)
    
    print(('\n'+'='*80+'\n').join(code_strings))


prompt = '''
def area_of_rectangle(a: float, b: float):
    """Return the area of the rectangle"""
'''
complete_code(generation, prompt, max_length=64)
prompt = '''
import numpy as np
def mean(a):
    """Computes the mean of a matrix with a given matrix."""
'''
complete_code(generation, prompt, max_length=64)

# prompt = '''
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# from sklearn.datasets import load_digits
# from sklearn.svm import SVC

# digits = load_digits()
# X, y = digits.data, digits.target

# model = SVC(kernel='linear')

# train_sizes, train_scores, test_scores = learning_curve(
#     model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))
# '''
# complete_code(generation, prompt, max_length=64)


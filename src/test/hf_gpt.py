# python3
# Create Date: 2024-07-30
# Func: compare FFN params 
# ================================================================================
import os 
import torch 
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config, pipeline 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''

gpt = pipeline('text-generation', model='openai-gpt')
g_gpt2 = pipeline('text-generation', model='gpt2')
gpt2 = GPT2LMHeadModel(GPT2Config()) # wpe wte

# FFN
ffn_ = gpt2.transformer.h[0].mlp
ffn_MB = sum([t.numel() * 4 for t in ffn_.parameters()])/1024**2

att_ = gpt2.transformer.h[0].attn
att_MB = sum([t.numel() * 4 for t in att_.parameters()])/1024**2
ffn_MB / (att_MB + ffn_MB) # 66.6%

ffn_.c_fc.weight.shape      # 768 -> 3072  key mem     word partern
ffn_.c_proj.weight.shape    # 3072 -> 768  value mem   content partern
 
att_.c_attn.weight.shape    # 768 -> 2304    q k v 768  qk -> a && av -> o
att_.c_proj.weight.shape    # 768 -> 768     w^T o -> o



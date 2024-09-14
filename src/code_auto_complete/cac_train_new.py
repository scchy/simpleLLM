# python3
# Create Date: 2024-09-08
# Author: Scc_hy
# Func:  training 
# Detial:   4070Ti-12Gb
#           Duration 27h 2m 41s 
#           效果不佳
# ====================================================================================================
import os 
import json
from torch.utils.data import IterableDataset
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, GPT2Config
from transformers import get_scheduler, set_seed, TrainingArguments, Trainer
from torch.optim import AdamW
from datasets import load_dataset
import datasets, transformers
from torch.utils.data import DataLoader
from argparse import Namespace
from datetime import datetime
from tqdm.auto import tqdm
import torch 
import logging 
import wandb 
from pynvml import (
    nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetMemoryInfo, 
    nvmlDeviceGetName,  nvmlShutdown, nvmlDeviceGetCount
)
from cac_dataloader import cacDataset
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
CUR_DIR = os.path.dirname(__file__)

def cuda_mem():
    # 21385MiB / 81920MiB
    fill = 0
    n = datetime.now()
    nvmlInit()
    # 创建句柄
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        # 获取信息
        info = nvmlDeviceGetMemoryInfo(handle)
        # 获取gpu名称
        gpu_name = nvmlDeviceGetName(handle)
        # 查看型号、显存、温度、电源
        print("[ {} ]-[ GPU{}: {}".format(n, i, gpu_name), end="    ")
        print("总共显存: {:.3}G".format((info.total // 1048576) / 1024), end="    ")
        print("空余显存: {:.3}G".format((info.free // 1048576) / 1024), end="    ")
        model_use = (info.used  // 1048576) - fill
        print("模型使用显存: {:.3}G({}MiB)".format( model_use / 1024, model_use))
    nvmlShutdown()


def safe_mkdir(path):
    if os.path.exists(path):
        return None 
    os.makedirs(path)


def model_download():
    data_name = 'openai-community/gpt2-xl'
    final_out_path = '/home/scc/sccWork/devData/myData/hf_models/gpt2-xl'
    safe_mkdir(final_out_path)
    os.system(f"""
    export HF_ENDPOINT=https://hf-mirror.com && \
    huggingface-cli download --resume-download {data_name} --local-dir-use-symlinks False \
    --repo-type model \
    --local-dir {final_out_path} \
    --cache-dir {final_out_path}/cache
    """)
    os.system(f'rm -rf {final_out_path}/cache')


def model_size(model):
    return sum([t.numel() * 4 for t in model.parameters()])/1024**2


def test_model():
    tokenizer = AutoTokenizer.from_pretrained(f"{CUR_DIR}/cac_tokenizer")
    gpt2_xl_dir = '/home/scc/sccWork/devData/myData/hf_models/gpt2-xl'
    config = AutoConfig.from_pretrained(gpt2_xl_dir, vocab_size=len(tokenizer))
    model = AutoModelForCausalLM.from_config(config)
    print(f'GPT-2 (xl) size: {model_size(model):.1f}M parameters')


def simple_data_collator(features, return_tensors="pt"):
    return {"input_ids": torch.stack(features), "labels": torch.stack(features), }


def create_dataloaders(dataset_path, tokenizer, args):
    ds_kwargs = {
        'streaming': True,
        'chunksize': 40<<10
    }
    train_data = load_dataset(dataset_path, split="train", **ds_kwargs)
    valid_data = load_dataset(dataset_path +'-val', split="validation", **ds_kwargs)
    train_dataset = cacDataset(tokenizer, train_data, seq_length=args.seq_length, num_of_sequences=args.num_of_sequences)
    valid_dataset = cacDataset(tokenizer, valid_data, seq_length=args.seq_length, num_of_sequences=args.num_of_sequences)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader, train_dataset, valid_dataset



proj_name = 'cac_taining'
config = {
    "seq_length": 1024,
    "num_of_sequences": 128, #512,
    "train_batch_size": 2,
    "valid_batch_size": 2,
    "weight_decay": 0.1,
    "shuffle_buffer": 1000,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 750,
    "gradient_accumulation_steps": 16,
    "max_train_steps": 1000, # 50000,
    "max_eval_steps": -1,
    "seed": 1,
    "save_checkpoint_steps": 1000, # 50000
}
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    optim='adamw_torch',
    warmup_steps=750,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    logging_dir='./logs',
    logging_steps=10,
    save_steps=2000,
    # Accelerator
    fp16=True, # 启用混合精度
    fp16_opt_level='O1', # O2 这是更激进的混合精度训练，提供额外的优化
    dispatch_batches=True, 
    # fsdp=True,
    # fsdp_config=dict(activation_checkpointing=True),
    report_to=['wandb'],
    max_steps=30000, # IterableDataset 50000
    # do_eval=True,
    # eval_steps=0.2
)
# print(f">>>>>>>>>>>>>>>>>>>> training_args={training_args}")
args = Namespace(**config)
set_seed(args.seed)
wandb.init(project=proj_name, config=training_args.to_dict())

# tokenizer & model 
p_ = f"{CUR_DIR}/cac_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(p_)
vocab_size = len(tokenizer.vocab.items())
with open(f"{CUR_DIR}/cac_config.json", 'r') as f:
    cfg_js = json.load(f)
    
print(f">>>> tokenizer: {tokenizer.eos_token_id=} config-json: {cfg_js['eos_token_id']=}")
print(f">>>> tokenizer: {vocab_size=}             config-json: {cfg_js['vocab_size']=}")
# gpt2_xl_dir = '/home/scc/sccWork/devData/myData/hf_models/gpt2-xl'
# model = GPT2LMHeadModel.from_pretrained(gpt2_xl_dir, ignore_mismatched_sizes=True, config=GPT2Config(**cfg_js)) #, gradient_checkpointing=True)
model = GPT2LMHeadModel(config=GPT2Config(**cfg_js))
print(f'>>>>>>>>> GPT2LMHeadModel size: {model_size(model):.1f}M parameters')
cuda_mem()

# dataloader
data_p = '/home/scc/sccWork/devData/myData/train_data/codeparrot'
tr_dataloader, val_dataloader, train_dataset, valid_dataset = create_dataloaders(data_p, tokenizer, args)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=simple_data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # tokenizer=tokenizer,
)


trainer.train()
# results = trainer.evaluate()
# print(results)



# if __name__ == '__main__':
#     # model_download()
#     test_model()


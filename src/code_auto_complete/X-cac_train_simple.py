# python3
# Create Date: 2024-09-04
# Author: Scc_hy
# Func:  training 
# Block  QKV attention 占用过多的显存
# ====================================================================================================
import os 
import json
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, GPT2Config
from transformers import get_scheduler, set_seed
from torch.optim import AdamW
from datasets import load_dataset
from accelerate import Accelerator
import datasets, transformers
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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


def setup_logging(project_name, args):
    logger = logging.getLogger(__name__)
    if not os.path.exists(f'{CUR_DIR}/log'):
        os.makedirs(f'{CUR_DIR}/log')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d_%H:%M:%S", 
        level=logging.INFO, 
        handlers=[
        logging.FileHandler(f"{CUR_DIR}/log/debug_0.log"),
        logging.StreamHandler()]
    )
    # wandb.init(project=project_name, config=args)
    # run_name = wandb.run.name
    run_name = None
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_info()
    transformers.utils.logging.set_verbosity_info()
    return logger, run_name


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
    return train_dataloader, eval_dataloader


def get_groupped_params(model, args, no_decay=['bias', 'LayerNorm.weight']):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay): 
            params_without_wd.append(p)
            continue 
        params_with_wd.append(p)
    return [
        {'params': params_with_wd, 'weight_decay': args.weight_decay},
        {'params': params_without_wd, 'weight_decay': 0.0}
    ]


def log_metrics(logger, step, metrics):
    logger.info(f"Step {step}: {metrics}")
    # wandb.log(metrics)


@torch.no_grad
def evaluate(model, eval_dataloader, args):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        out = model(batch, labels=batch)
        loss = out.loss.repeat(args.valid_batch_size)
        losses.append(loss)
        if args.max_eval_steps > 0 and step >= args.max_eval_steps: 
            break
    loss = torch.mean(torch.cat(losses))
    try: 
        perplexity = torch.exp(loss)
    except OverflowError: 
        perplexity = float("inf")
    return loss.item(), perplexity.item()



proj_name = 'cac_taining'
config = {
    "seq_length": 1024,
    "num_of_sequences": 512,
    "train_batch_size": 2,
    "valid_batch_size": 2,
    "weight_decay": 0.1,
    "shuffle_buffer": 1000,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 750,
    "gradient_accumulation_steps": 16,
    "max_train_steps": 100, # 50000,
    "max_eval_steps": -1,
    "seed": 1,
    "save_checkpoint_steps": 100, # 50000
}
args = Namespace(**config)
set_seed(args.seed)
logger, run_name = setup_logging(proj_name, args)

# tokenizer & model 
p_ = f"{CUR_DIR}/cac_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(p_)
vocab_size = len(tokenizer.vocab.items())


gpt2_xl_dir = '/home/scc/sccWork/devData/myData/hf_models/gpt2-xl'
with open(f"{CUR_DIR}/cac_config.json", 'r') as f:
    cfg_js = json.load(f)
    
print(f">>>> tokenizer: {tokenizer.eos_token_id=} config-json: {cfg_js['eos_token_id']=}")
print(f">>>> tokenizer: {vocab_size=}             config-json: {cfg_js['vocab_size']=}")
# model = GPT2LMHeadModel.from_pretrained(gpt2_xl_dir, ignore_mismatched_sizes=True, config=GPT2Config(**cfg_js)) #, gradient_checkpointing=True)
model = GPT2LMHeadModel(config=GPT2Config(**cfg_js))
print("model=", model)
model.to('cuda')
print(f'>>>>>>>>> GPT2LMHeadModel (xl) size: {model_size(model):.1f}M parameters')
cuda_mem()

# dataloader
data_p = '/home/scc/sccWork/devData/myData/train_data/codeparrot'
tr_dataloader, val_dataloader = create_dataloaders(data_p, tokenizer, args)

# optimzer
optimizer = AdamW(get_groupped_params(model, args), lr=args.learning_rate)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps
)
def get_lr():
    return optimizer.param_groups[0]['lr']


# Train model
model.train()
completed_steps = 0
for step, batch in tqdm(enumerate(tr_dataloader, start=1)):
    optimizer.zero_grad()
    batch = batch.to('cuda')
    cuda_mem()
    print(f'{batch.shape=}')
    loss = model(batch, labels=batch, use_cache=True).loss
    print(f'{loss=}')
    log_metrics(
        logger, 
        step, 
        {
            'lr': get_lr(), 
            'samples': step*samples_per_step,
            'steps': completed_steps, 
            'loss/train': loss.item()
        }
    )
    loss.backward()
    torch.clip_grad_norm_ (model.parameters(), 1.0)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    completed_steps += 1

    if step % args.save_checkpoint_steps == 0:
        logger.info('Evaluating and saving model checkpoint')
        eval_loss, perplexity = evaluate(model, val_dataloader, args)
        log_metrics(
            logger,
            step, 
            {'loss/eval': eval_loss, 'perplexity': perplexity}
        )
        model.save_pretrained(f"{CUR_DIR}/")
        model.train()
    if completed_steps >= args.max_train_steps:
        break

# Evaluate and save the last checkpoint
logger.info('Evaluating and saving model after training')
eval_loss, perplexity = evaluate(args)
log_metrics(
    logger,
    step, 
    {'loss/eval': eval_loss, 'perplexity': perplexity}
)
model.save_pretrained(f"{CUR_DIR}/")

# if __name__ == '__main__':
#     # model_download()
#     test_model()


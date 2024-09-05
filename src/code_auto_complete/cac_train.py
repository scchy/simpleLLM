# python3
# Create Date: 2024-09-04
# Author: Scc_hy
# Func:  training 
# ====================================================================================================
import os 
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import AdamW, get_scheduler, set_seed
from datasets import load_dataset
from accelerate import Accelerator
import datasets, transformers
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace
import torch 
import logging 
import wandb 
from cac_dataloader import cacDataset
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
CUR_DIR = os.path.dirname(__file__)

# Accelerator
accelerator = Accelerator(dispatch_batches=True)
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}


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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d_%H:%M:%S", 
        level=logging.INFO, 
        handlers=[
        logging.FileHandler(f"log/debug_{accelerator.process_index}.log"),
        logging.StreamHandler()]
    )
    if accelerator.is_main_process: # we only want to setup looging once
        wandb.init(project=project_name, config=args)
        run_name = wandb.run.name
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        run_name = ''
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, run_name


def create_dataloaders(dataset_path, tokenizer, args):
    ds_kwargs = {
        'streaming': True,
        'chunksize': 40<<20,
        'error_bad_chunck': False
    }
    train_data = load_dataset(dataset_path, split="train", **ds_kwargs)
    train_dataset = cacDataset(tokenizer, train_data, seq_length=args.seq_length)
    # valid_dataset = cacDataset(tokenizer, valid_data, seq_length=args.seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    # eval_dataloader=DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, None


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
    if accelerator.is_main_process:
        wandb.log(metrics)


@torch.no_grad
def evaluate(model, eval_dataloader, args):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        out = model(batch, labels=batch)
        loss = out.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
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
    "train_batch_size": 2,
    "valid_batch_size": 2,
    "weight_decay": 0.1,
    "shuffle_buffer": 1_000,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 750,
    "gradient_accumulation_steps": 16,
    "max_train_steps": 50000,
    "max_eval_steps": -1,
    "seq_length": 1024,
    "seed": 1,
    "save_checkpoint_steps": 50_000
}
args = Namespace(**config, **acc_state)
samples_per_step = accelerator.state.num_processes * args.train_batch_size
set_seed(args.seed)
logger, run_name = setup_logging(proj_name)
logger.info(accelerator.state)

# tokenizer & model 
gpt2_xl_dir = '/home/scc/sccWork/devData/myData/hf_models/gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(gpt2_xl_dir, gradient_checkpointing=True)
p_ = f"{CUR_DIR}/cac_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(p_)
data_p = '/home/scc/sccWork/devData/myData/train_data/codeparrot'

# dataloader
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

model, optimizer, tr_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, tr_dataloader, val_dataloader 
)

# Train model
model.train()
completed_steps = 0
for step, batch in enumerate(tr_dataloader, start=1):
    loss = model(batch, labels=batch, use_cache=False).loss
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
    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % args.gradient_accumulation_steps == 0:
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
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
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(f"{CUR_DIR}/")
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
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if accelerator.is_main_process:
    unwrapped_model.save_pretrained(f"{CUR_DIR}/")


# if __name__ == '__main__':
#     # model_download()
#     test_model()


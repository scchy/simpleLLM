# python3
# Create Date: 2024-08-20
# Author: Scc_hy
# Func: data download & loading check
# desc: public reposiitories from the snapshot on Google BigQuery
# ====================================================================================================

# data download
from datasets import load_dataset, DownloadConfig
import os 
import psutil
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def download():
    TOKEN = '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'

    data_name = 'transformersbook/codeparrot'
    final_out_path = '/home/scc/sccWork/devData/myData/train_data/codeparrot'
    os.system(f"""
    export HF_ENDPOINT=https://hf-mirror.com && \
    huggingface-cli download --resume-download {data_name} --local-dir-use-symlinks False \
    --repo-type dataset \
    --local-dir {final_out_path} \
    --cache-dir {final_out_path}/cache 
    """)
    # --token {TOKEN}
    os.system(f'rm -rf {final_out_path}/cache')



def loading_check():
    # load in desk
    dataset = load_dataset(
        path='/home/scc/sccWork/devData/myData/train_data/codeparrot', 
        split="train",
        streaming=True
    )
    iterator = iter(dataset)
    sample_ = next(iterator)
    print(f'{sample_.keys()=}\n\n------------------------------\n', sample_['content'])


if __name__ == '__main__':
    # download()
    loading_check()


####
# Download datasets from ChatRAG-Bench (test splits only)
# https://huggingface.co/datasets/nvidia/ChatRAG-Bench
####
import os
from datasets import load_dataset

data_dir = 'data/nvidia_raw'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dataset_names = [
    'coqa', 'inscit', 'topiocqa', 'hybridial', 'doc2dial', 'quac', 'qrecc', 'doqa_cooking',
    'doqa_movies', 'doqa_travel', 'sqa',
]
for dataset_name in dataset_names:
    print(f'Downloading `{dataset_name}`')
    dataset = load_dataset('nvidia/ChatRAG-Bench', dataset_name)
    if 'test' in dataset:
        dataset['test'].to_parquet(os.path.join(data_dir, f'{dataset_name}_test.parquet'))
        print(f'{dataset_name}_test.parquet saved')
    else:
        print(f'skipping `{dataset_name}` - no test split')

import os
os.environ.pop('HF_ENDPOINT')
print(os.environ.get('HF_ENDPOINT'))

from datasets import load_dataset

dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
dataset = load_dataset(dataset_id, split='train[:5%]')

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']
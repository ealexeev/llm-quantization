from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from typing import Callable, Dict, Optional


# TokenizeDataset returns a dataset with a single column containing the tokenized input using process_func to select from a shuffled (based on seed) and sliced (based on count).
def _TokenizeDataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase, 
    split=str,
    trust_remote_code=False,
    process_fn: Optional[Callable[[Dict], Dict[str, str]]] = None,
    config_name: Optional[str] = None,
    seed: Optional[int] = None, 
    count: Optional[int] = None,
):

    ds = load_dataset(dataset_name, name=config_name, split=split, streaming=True)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=1000)
    column_names = ds.column_names

    def proces_fn(example: Dict[str, str]):
        text = tokenizer.apply_chat_template(example[column], tokenize=False)
        return tokenizer(text, truncation=True, max_length=2048)

    ds = ds.map(process_fn)
    ds = ds.remove_columns(column_names)
    if count > 0:
        ds = ds.take(count)
    return ds
    

def TokenizeUltraChat(tokenizer: PreTrainedTokenizerBase, seed: int, count: int):
    def process_fn(row):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return tokenizer(text, truncation=True, max_length=2048)

    return _TokenizeDataset(dataset_name="HuggingFaceH4/ultrachat_200k",
                            tokenizer=tokenizer,
                            split="train_sft",
                            process_fn=process_fn,
                            seed=seed,
                            count=count)

def TokenizeFictionBooks(tokenizer: PreTrainedTokenizerBase, seed: int, count: int):
    def process_fn(row):
        text = tokenizer.apply_chat_template([{"role": "user", "content": row["Input"]}], tokenize=False)
        return tokenizer(text, truncation=True, max_length=2048)

    return _TokenizeDataset(dataset_name="mrcedric98/fiction_books_v8",
                            tokenizer=tokenizer,
                            split="train",
                            process_fn=process_fn,
                            seed=seed,
                            count=count)
                                             

def TokenizeC4(tokenizer: PreTrainedTokenizerBase, seed: int, count: int):
    def process_fn(row):
        text = tokenizer.apply_chat_template([{"role": "user", "content": row["text"]}], tokenize=False)
        return tokenizer(text, truncation=True, max_length=2048)

    return _TokenizeDataset(dataset_name="allenai/c4",
                            config_name="en",
                            tokenizer=tokenizer,
                            split="train",
                            process_fn=process_fn,
                            seed=seed,
                            count=count)

# This does not work because PG19 uses some legacy script for loading.  I'll figure it out later.
def TokenizePG19(tokenizer: PreTrainedTokenizerBase, seed: int, count: int):
    def process_fn(row):
        data = row["text"][len(row["text"])//2:50]
        print(data)
        text = tokenizer.apply_chat_template([{"role": "user", "content": data}], tokenize=False)
        return tokenizer(text, truncation=True, max_length=2048)

    return _TokenizeDataset(dataset_name="deepmind/pg19",
                            tokenizer=tokenizer,
                            split="train",
                            trust_remote_code=True,
                            process_fn=process_fn,
                            seed=seed,
                            count=count)
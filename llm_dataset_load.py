from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from typing import Callable, Dict, Optional


def _LoadDataset(
    dataset_name: str,
    split: str,
    config_name: Optional[str] = None,
    seed: Optional[int] = None,
    count: Optional[int] = None,
    trust_remote_code=False,
):
    ds = load_dataset(dataset_name, name=config_name, split=split, streaming=True, trust_remote_code=trust_remote_code)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=1000)
    if count is not None and count > 0:
        ds = ds.take(count)
    return ds


def _TokenizeDataset(
    dataset_name: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    process_fn: Callable[[Dict], Dict[str, str]],
    config_name: Optional[str] = None,
    seed: Optional[int] = None,
    count: Optional[int] = None,
    trust_remote_code=False,
):
    ds = _LoadDataset(dataset_name, split, config_name, seed, count, trust_remote_code)
    column_names = ds.column_names
    ds = ds.map(process_fn)
    ds = ds.remove_columns(column_names)
    return ds


def GetUltraChat(seed: int, count: int):
    return _LoadDataset(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        seed=seed,
        count=count
    )


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

def GetFictionBooks(seed: int, count: int):
    return _LoadDataset(
        dataset_name="mrcedric98/fiction_books_v8",
        split="train",
        seed=seed,
        count=count
    )


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
                                             

def GetC4(seed: int, count: int):
    return _LoadDataset(
        dataset_name="allenai/c4",
        config_name="en",
        split="train",
        seed=seed,
        count=count
    )


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
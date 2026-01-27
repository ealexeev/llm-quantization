import argparse
import json
from collections import Counter
from datasets import concatenate_datasets

import llm_dataset_load


def _calculate_scalar(size: int, ratio: Counter) -> int:
    scalar = size // ratio.total() or 1
    if ratio.total() * scalar < size:
        scalar += 1
    return scalar


def _prepare_dataset(seed: int, size: int, ratio: Counter):
    scalar = _calculate_scalar(size, ratio)
    ratio = ratio.copy()
    for k, v in ratio.items():
        ratio[k] = v * scalar

    datasets_list = []

    # UltraChat
    if ratio["ultra_chat"] > 0:
        print(f"Loading UltraChat (ratio: {ratio['ultra_chat']})...")
        ds = llm_dataset_load.GetUltraChat(seed, ratio["ultra_chat"])
        columns_to_remove = [c for c in ds.column_names if c != "text"]
        def process_ultrachat(example):
            return {"text": json.dumps(example["messages"])}
        
        ds = ds.map(process_ultrachat)
        ds = ds.remove_columns(columns_to_remove)
        datasets_list.append(ds)

    # Fiction Books
    if ratio["fiction_v8"] > 0:
        print(f"Loading Fiction Books (ratio: {ratio['fiction_v8']})...")
        ds = llm_dataset_load.GetFictionBooks(seed, ratio["fiction_v8"])
        columns_to_remove = [c for c in ds.column_names if c != "text"]
        ds = ds.map(lambda x: {"text": x["Input"]})
        ds = ds.remove_columns(columns_to_remove)
        datasets_list.append(ds)

    # C4
    if ratio["c4_en"] > 0:
        print(f"Loading C4 (ratio: {ratio['c4_en']})...")
        ds = llm_dataset_load.GetC4(seed, ratio["c4_en"])
        columns_to_remove = [c for c in ds.column_names if c != "text"]
        ds = ds.remove_columns(columns_to_remove)
        datasets_list.append(ds)

    if not datasets_list:
        raise ValueError("No datasets selected via ratios.")

    iterable_ds = concatenate_datasets(datasets_list)
    return iterable_ds


def parse_sizes(value):
    return [int(x) for x in value.split(',')]


def main():
    parser = argparse.ArgumentParser(
        prog='export_calibration_dataset',
        description='Export calibration dataset to JSONL.'
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output filename base (without extension).")
    parser.add_argument("-s", "--size", type=parse_sizes, required=True, help="Size of dataset. Can be a single integer or a comma-separated list.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed used to shuffle datasets.")
    parser.add_argument("--ultra_chat", type=int, required=True, help="Ratio of dataset to build from ultrachat_200k")
    parser.add_argument("--c4_en", type=int, required=True, help="Ratio of dataset to build from C4")
    parser.add_argument("--fiction_v8", type=int, required=True, help="Ratio of dataset to build fiction books v8")

    args = parser.parse_args()

    ratio = Counter({
        "ultra_chat": args.ultra_chat,
        "c4_en": args.c4_en,
        "fiction_v8": args.fiction_v8
    })

    if ratio.total() == 0:
        print("Error: Total ratio is zero. Please specify at least one dataset ratio > 0.")
        return

    for size in args.size:
        output_filename = f"{args.output}_{size}S.jsonl"
        print(f"Preparing dataset for size {size}...")
        
        sample_count = _calculate_scalar(size, ratio) * ratio.total()
        
        dataset = _prepare_dataset(args.seed, size, ratio)
        
        print(f"Writing to {output_filename}...")
        count = 0
        with open(output_filename, 'w') as f:
            for example in dataset:
                json.dump(example, f)
                f.write('\n')
                count += 1
                
        print(f"Wrote {count} lines.")

if __name__ == "__main__":
    main()

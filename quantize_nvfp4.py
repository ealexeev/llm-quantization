from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset, concatenate_datasets, Dataset

import argparse
from collections import Counter

import llm_dataset_load


def _calculate_scalar(size: int, ratio: Counter) -> int:
    scalar = size // ratio.total() or 1
    if ratio.total() * scalar < size:
        scalar += 1
    return scalar

def _prepare_dataset(tokenizer: PreTrainedTokenizerBase, seed: int, size: int, ratio: Counter):
    scalar = _calculate_scalar(size, ratio)
    for k, v in ratio.items():
        ratio[k]=v*scalar

    ultra_chat = llm_dataset_load.TokenizeUltraChat(tokenizer, seed, ratio["ultra_chat"])
    fiction_v8 = llm_dataset_load.TokenizeFictionBooks(tokenizer, seed, ratio["fiction_v8"])
    c4_en = llm_dataset_load.TokenizeC4(tokenizer, seed, ratio["c4_en"])    
    iterable_ds = concatenate_datasets([ultra_chat, fiction_v8, c4_en])
    return Dataset.from_generator(lambda: (yield from iterable_ds), features=iterable_ds.features)



def quantize(model: str, output: str, size:int, seed: int, basic: bool):

    tokenizer = AutoTokenizer.from_pretrained(model, fix_mistral_regex=True) # Mistral only, pre 2503 is ok.
    sample_count = _calculate_scalar(size, ratio)*ratio.total()
    dataset = _prepare_dataset(tokenizer, seed, size, ratio)
    
    print(f"Loading {model}...")
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="cuda:0",
        torch_dtype="auto",
    )

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=["lm_head"]
    )

    print(f"Starting calibration over {sample_count} samples...")
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=sample_count,
        tokenizer=tokenizer,
        pipeline="basic" if basic else "sequential",
    )

    print(f"Saving to {output}...")
    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='quantize_nvfp4',
        description='Quantize an LLM model to NVFP4 using')
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")
    parser.add_argument("-s", "--size", type=int, required=True, help="Size of calibration dataset.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed used to shuffle datasets.")
    parser.add_argument("--ultra_chat", type=int, required=True, help="Ratio of dataset to build from ultrachat_200k")
    parser.add_argument("--c4_en", type=int, required=True, help="Ratio of dataset to build from C4")
    parser.add_argument("--fiction_v8", type=int, required=True, help="Ratio of dataset to build fiction books v8")
    parser.add_argument("--pipeline_basic", action="store_true", help="Run llmcompressor BasicPipeline for a full GPU VRAM offload, SequentialPipeline when not set.")
    args = parser.parse_args()
    ratio = Counter({
        "ultra_chat": args.ultra_chat,
        "c4_en": args.c4_en,
        "fiction_v8": args.fiction_v8})

    quantize(model=args.model, output=args.output, size=args.size, seed=args.seed, basic=args.pipeline_basic)


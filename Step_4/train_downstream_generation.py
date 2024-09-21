import datasets
import transformers
import sys
import logging
import torch
import ast
import data_utils
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from pynvml import *
from functools import partial
import numpy as np

def compute_metrics(pred, tokenizer):
    # Decode the predictions and references
    predictions = pred.predictions.argmax(-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Decode the labels
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id  # Replace -100 with pad_token_id
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute Next Token Accuracy
    accuracy = (predictions == label_ids).astype(np.float32).mean()

    result = {"accuracy": accuracy * 100}

    # If evaluation loss is provided, compute perplexity
    if hasattr(pred, 'eval_loss'):
        perplexity = np.exp(pred.eval_loss)
        result["perplexity"] = perplexity
    else:
        # If eval_loss is not available, log a warning or handle accordingly
        result["perplexity"] = None

    return result

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name: str = field(default="NousResearch/Llama-2-13b-hf", metadata={
        "help": "Model name in HuggingFace, e.g. 'Llama-2-7b'"
    })
    dataset_name: str = field(default="knkarthick/dialogsum", metadata={
        "help": "Dataset name in HuggingFace, e.g. 'dialsum'"
    })
    sequence_len: int = field(default=512, metadata={
        "help": "Maximum sequence length"
    })

@dataclass
class LoraArguments:
    enable_lora: bool = field(default=True, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=16, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.1, metadata={
        "help": "LoRA dropout"
    })

    target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'v_proj'],  # Example target modules for LoRA
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q_proj', 'v_proj']"
        },
    )

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        params["target_modules"] = ast.literal_eval(params["target_modules"][0])
        return LoraConfig(**params)

@dataclass
class Arguments:
    train: transformers.TrainingArguments
    model: ModelArguments
    lora: LoraArguments

def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.train.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.train.local_rank}, device: {args.train.device}, n_gpu: {args.train.n_gpu}, "
        f"distributed training: {bool(args.train.local_rank != -1)}, 16-bits training: {args.train.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args.train}")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name,use_auth_token=True, cache_dir = '/root/autodl-tmp/DSI-QG/models')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    dataset_train = datasets.load_dataset("json", data_files="/root/autodl-tmp/DSI-QG/data/dialsum/dialsum_private.jsonl", split="train")
    dataset_valid = datasets.load_dataset(args.model.dataset_name)['validation']
    dataset_valid = dataset_valid.remove_columns(["id","topic"])
    dataset = datasets.DatasetDict({
    'train': dataset_train,
    'validation': dataset_valid})

    def preprocess_function(examples):
        inputs = examples['dialogue']
        modified_inputs = [f"An AI tool that predicts the next token in a sequence.\n### Input: {dialogue}\n### Output:" for dialogue in inputs]
        model_inputs = tokenizer(modified_inputs, max_length=args.model.sequence_len, padding="max_length", truncation=True)
        labels = tokenizer(examples['summary'], max_length=args.model.sequence_len, padding="max_length", truncation=True)
        
        # # Make sure labels have the correct format and padding
        labels["input_ids"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_list]
            for labels_list in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with args.train.main_process_first(desc="tokenizing dataset"):
        tokenized_datasets = dataset.map(
            preprocess_function, batched=True, num_proc=8, desc="tokenizing dataset", 
            remove_columns=dataset['train'].column_names
        )

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name,cache_dir = '/root/autodl-tmp/DSI-QG/models',use_auth_token='xxx')
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.train.gradient_checkpointing)

    if args.lora.enable_lora:
        logger.info("Using LoRA")
        model = get_peft_model(model=model, peft_config=args.lora.as_peft_config())
    else:
        logger.info("Not using LoRA")
    if args.train.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    trainer = transformers.Trainer(
    args=args.train,
    model=model,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, tokenizer=tokenizer)
)

    result = trainer.train()

    def print_summary(result):
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        print_gpu_utilization()

    print_summary(result)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((transformers.TrainingArguments, ModelArguments, LoraArguments))
    train_args, model_args, lora_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, model=model_args, lora=lora_args))

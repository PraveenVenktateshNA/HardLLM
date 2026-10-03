from data import IndexingTrainDataset, IndexingCollator
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from trainer import DSITrainer, DocTqueryTrainer
import numpy as np
import torch
try:
    import wandb
except ImportError:  # logging is optional; run with --report_to none
    wandb = None
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
from tqdm import tqdm
set_seed(313)
import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Current directory:", current_directory)

@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    # 256, not the upstream 32. The SST-2 default truncates this pool badly:
    # median row is 46 words and only 29% of medical rows fit in 32 tokens, so
    # the model would see one sentence while predicting a text_id that was
    # clustered from the whole passage.
    max_length: Optional[int] = field(default=256)
    id_max_length: Optional[int] = field(default=20)
    remove_prompt: Optional[bool] = field(default=False)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    task: str = field(default=None,  metadata={"help": "DSI, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)
    # How many training rows to reuse as the eval set. Evaluation runs 20-beam
    # constrained generation, so this is the dominant eval cost -- keep it small
    # for a smoke test, and note that DSI eval is deliberately measured ON seen
    # data (indexing is memorisation, not generalisation).
    eval_subset: Optional[int] = field(default=1000)


def make_compute_metrics(tokenizer, valid_ids):

    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams,
                                               skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)

            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
        return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}
    return compute_metrics


def main():
    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()
    # We use wandb logger: https://wandb.ai/site.
    # local_rank is -1 for single-process runs and 0 for the main process of a
    # distributed run; both are "the main process". Only touch wandb when the
    # caller actually asked for it, otherwise wandb.login() blocks on a prompt.
    if (training_args.local_rank in (-1, 0)
            and wandb is not None
            and "wandb" in (training_args.report_to or [])):
        # Initialize wandb run
        wandb.login()
        wandb.init(project="DSI", name=training_args.run_name)

    if 'mt5' in run_args.model_name:
        # transformers 5.x removed MT5Tokenizer / MT5TokenizerFast. AutoTokenizer
        # resolves the right class from the checkpoint config. google/mt5-* ships
        # only spiece.model (no tokenizer.json), so the fast variant is converted
        # at load time -- that conversion needs sentencepiece and protobuf.
        tokenizer = AutoTokenizer.from_pretrained(run_args.model_name, cache_dir='./cache', use_fast=False)
        fast_tokenizer = AutoTokenizer.from_pretrained(run_args.model_name, cache_dir='./cache', use_fast=True)
        if run_args.model_path:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='./cache')
        else:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='./cache')
    else:
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name, cache_dir='./cache')
        fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='./cache')
        if run_args.model_path:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='./cache')
        else:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='./cache')



    train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='./cache',
                                             tokenizer=tokenizer)

    n_eval = min(run_args.eval_subset, len(train_dataset))
    valid_dataset = torch.utils.data.Subset(train_dataset, list(range(n_eval)))
        ################################################################
        # docid generation constrain, we only generate integer docids.
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
        ################################################################
    # # Freeze all layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Unfreeze the last layer
    # for param in model.lm_head.parameters():
    #     param.requires_grad = True
    trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length
        )
    # Pass the flag through so --resume_from_checkpoint works. Trainer.train()
    # does not read it off args by itself, and without this a resize, crash or
    # preemption means restarting from step 0.
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

if __name__ == "__main__":
    main()


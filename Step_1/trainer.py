"""Trainers for Step 1: Training Retrieval Model via Document Index.

This module was missing from the distributed source tree. It is reconstructed
from the contract imposed by its call sites, principally:

  * ``train_retrival_model.py`` -- instantiates ``DSITrainer`` with the extra
    keyword arguments ``restrict_decode_vocab`` and ``id_max_length``.
  * ``make_compute_metrics`` in the same file -- consumes
    ``eval_preds.predictions`` as a *ranked beam list per example* and
    ``eval_preds.label_ids`` as one doc-id sequence per example, then reports
    Hits@1 / Hits@10.

Those two facts fully determine the shapes this trainer must emit:

    predictions : (num_examples, num_return_sequences, id_max_length)
    label_ids   : (num_examples, id_max_length)

The doc-id vocabulary is constrained at generation time via
``prefix_allowed_tokens_fn`` so the model can only emit integer doc-ids, which
is what ``restrict_decode_vocab`` provides.
"""

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput

# DSI evaluation ranks the top-20 beams and scores Hits@1 / Hits@10 from them.
DEFAULT_NUM_BEAMS = 20


def _normalise_trainer_kwargs(kwds: dict) -> dict:
    """Accept ``tokenizer=`` on every supported transformers version.

    ``Trainer.__init__`` renamed ``tokenizer`` to ``processing_class`` in
    transformers 4.46 and dropped the old name in 5.x. The call site in
    ``train_retrival_model.py`` passes ``tokenizer=``, so translate it when the
    installed Trainer no longer accepts that name. This keeps the original
    call site untouched.
    """
    if "tokenizer" not in kwds:
        return kwds
    params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in params:
        return kwds
    if "processing_class" in params:
        kwds = dict(kwds)
        kwds["processing_class"] = kwds.pop("tokenizer")
    return kwds


class DSITrainer(Trainer):
    """Trainer for the ``sentence -> document index`` seq2seq objective.

    Training is ordinary seq2seq cross-entropy. Evaluation is what differs:
    instead of scoring logits, we run constrained beam search and hand the
    decoded beams to ``compute_metrics`` so it can compute retrieval hit rates.
    """

    def __init__(
        self,
        restrict_decode_vocab,
        id_max_length: int,
        num_return_sequences: int = DEFAULT_NUM_BEAMS,
        **kwds,
    ):
        super().__init__(**_normalise_trainer_kwargs(kwds))
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length
        self.num_return_sequences = num_return_sequences

    @property
    def _tok(self):
        """The tokenizer, under whichever attribute this version exposes."""
        return getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)

    def _pad_token_id(self) -> int:
        tok = self._tok
        if tok is not None and getattr(tok, "pad_token_id", None) is not None:
            return tok.pad_token_id
        if tok is not None and getattr(tok, "eos_token_id", None) is not None:
            return tok.eos_token_id
        pad = getattr(self.model.config, "pad_token_id", None)
        if pad is None:
            raise ValueError(
                "pad_token_id must be set on the tokenizer or the model config "
                "so that generated doc-ids can be padded to a uniform length."
            )
        return pad

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # **kwargs absorbs `num_items_in_batch`, which transformers >=4.46
        # passes through for gradient-accumulation loss scaling.
        loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        ).loss
        if return_outputs:
            return loss, [None, None]
        return loss

    def _pad_to_id_max_length(self, tensor: torch.Tensor) -> torch.Tensor:
        """Right-pad the last axis to ``id_max_length``.

        Every eval batch must agree on this axis, otherwise the Trainer cannot
        concatenate batches into a single predictions array.
        """
        if tensor.shape[-1] >= self.id_max_length:
            return tensor[..., : self.id_max_length]
        pad_token_id = self._pad_token_id()
        padded = tensor.new_full(
            (*tensor.shape[:-1], self.id_max_length), pad_token_id
        )
        padded[..., : tensor.shape[-1]] = tensor
        return padded

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if prediction_loss_only:
            return (None, None, None)

        model.eval()
        inputs = self._prepare_inputs(inputs)
        batch_size = inputs["input_ids"].shape[0]

        with torch.no_grad():
            beams = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=self.id_max_length,
                num_beams=self.num_return_sequences,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=self.num_return_sequences,
                early_stopping=True,
            )

        # generate() returns (batch * num_return_sequences, gen_len), ordered
        # best-first within each example. Reshape so compute_metrics receives
        # one ranked beam list per example.
        beams = beams.reshape(batch_size, self.num_return_sequences, -1)
        beams = self._pad_to_id_max_length(beams)

        labels = inputs["labels"]
        # IndexingCollator masks label padding with -100 for the loss. Decoding
        # a negative id raises, so restore the real pad id before handing the
        # labels to compute_metrics.
        labels = labels.masked_fill(labels == -100, self._pad_token_id())
        labels = self._pad_to_id_max_length(labels)

        return (None, beams, labels)


class DocTqueryTrainer(Trainer):
    """Trainer for the docTquery arm of DSI-QG (``document -> query``).

    ``train_retrival_model.py`` imports this name but never instantiates it;
    the HardLLM pipeline uses the clustered-index arm above. It is implemented
    here so the import resolves and so the docTquery variant remains available.
    """

    def __init__(self, do_generation: bool = True, **kwds):
        super().__init__(**_normalise_trainer_kwargs(kwds))
        self.do_generation = do_generation

    @property
    def _tok(self):
        return getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        ).loss
        if return_outputs:
            return loss, [None, None]
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.do_generation:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

        model.eval()
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            generated = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=self.max_length,
                do_sample=True,
                top_k=self.top_k,
                num_return_sequences=self.num_return_sequences,
            )
        labels = inputs.get("labels")
        return (None, generated, labels)

    def predict(
        self,
        test_dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        max_length: int = 32,
        num_return_sequences: int = 10,
        top_k: int = 10,
    ) -> PredictionOutput:
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.top_k = top_k
        return super().predict(
            test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

#!/usr/bin/env python

# Usage: ./train.py --output_dir=out --num_train_epochs=1 --gradient_checkpointing=True --per_device_train_batch_size=1

import torch
from transformers import HfArgumentParser, TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_from_disk
from peft import LoraConfig, TaskType
from dataclasses import dataclass, field, fields
from typing import Optional, Union


def main():
    parser = HfArgumentParser((TrainerArguments, TrainingArguments, LoraArguments))
    trainer_args, training_args, lora_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    assert (unknown_args == []), f"Unknown: {unknown_args}"
    trainer = Trainer(**to_dict(trainer_args))
    trainer.args = training_args
    lora_config = LoraConfig(**to_dict(lora_args))
    
    if trainer.args.gradient_checkpointing:
        # To prevent "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn" when gradient_checkpointing=True
        # https://github.com/huggingface/peft/issues/137
        trainer.model.enable_input_require_grads()
        # To prevent "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        trainer.model.config.use_cache = False
    if trainer.tokenizer.pad_token_id is None:
        # To prevent ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
        trainer.tokenizer.pad_token = trainer.tokenizer.eos_token 

    trainer.model.add_adapter(lora_config, adapter_name='adapter1')
    trainer.model.add_adapter(lora_config, adapter_name='adapter2')

    print("Training adapter1")
    trainer.model.set_adapter('adapter1')
    trainer.train()
    trainer.train()
    print("Training adapter2")
    trainer.model.set_adapter('adapter2')
    trainer.train()
    trainer.train()

    
@dataclass
class TrainerArguments:
    model: str = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "The model to train, evaluate or use for predictions."})
    #args: TrainingArguments provided separately
    data_collator: str = field(default="DataCollatorForSeq2Seq", metadata={"help": "The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`."})
    train_dataset: str = field(default="dataset1", metadata={"help": "Path to a preprocessed dataset to use for training."})
    eval_dataset: str = field(default="dataset1", metadata={"help": "Path to a preprocessed dataset to use for evaluation."})
    tokenizer: str = field(default=None, metadata={"help": "The tokenizer used to preprocess the data."})
    #model_init: We only support pretrained models
    compute_metrics: str = field(default=None, metadata={"help": "The function that will be used to compute metrics at evaluation."})
    callbacks: str = field(default=None, metadata={"help": "A comma separated list of callbacks to customize the training loop."})
    #optimizers: We configure this using TrainingArguments
    preprocess_logits_for_metrics: str = field(default=None, metadata={"help": "A function that preprocess the logits right before caching them at each evaluation step."})
    def __post_init__(self):
        model_name = self.model
        tokenizer_name = self.tokenizer if self.tokenizer else model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        print(f"{model_name} {self.model.dtype} {self.model.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.train_dataset = load_from_disk(self.train_dataset) if self.train_dataset else None
        self.eval_dataset = load_from_disk(self.eval_dataset) if self.eval_dataset else None
        self.data_collator = globals()[self.data_collator](tokenizer=self.tokenizer)
        self.compute_metrics = globals()[self.compute_metrics] if self.compute_metrics else None
        self.callbacks = [ globals()[x] for x in self.callbacks.split(',') ] if self.callbacks else None
        self.preprocess_logits_for_metrics = globals()[self.preprocess_logits_for_metrics] if self.preprocess_logits_for_metrics else None
        

@dataclass
class LoraArguments:
    # https://huggingface.co/docs/peft/quicktour does not set the first 3 args
    # This one is set by get_peft_model
    ## base_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    # Not sure whether this is used or set anywhere in peft
    ## revision: str = field(default=None, metadata={"help": "The specific model version to use."})
    # This is set by LoraConfig init
    ## peft_type: Union[str, PeftType] = field(default=PeftType.LORA, metadata={"help": "Peft type"})
    task_type: Union[str, TaskType] = field(default=TaskType.CAUSAL_LM, metadata={"help": "Task type"})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[str] = field(default="up_proj,down_proj,gate_proj", metadata={"help": "Comma separated list of module names (e.g. up_proj,down_proj,gate_proj) or (if no comma) regex expression of the module names to replace with Lora. For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(default=False, metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[str] = field(default=None, metadata={"help": "Comma separated list of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. For example, in Sequence Classification or Token Classification tasks, the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."})
    init_lora_weights: bool = field(default=True, metadata={"help": "Whether to initialize the weights of the Lora layers with their default initialization. Don't change this setting, except if you know exactly what you're doing."})
    layers_to_transform: Optional[str] = field(default=None, metadata={"help": "Comma separated list of layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."})
    layers_pattern: Optional[str] = field(default=None, metadata={ "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."})
    def __post_init__(self):
        if self.target_modules and ',' in self.target_modules:
            self.target_modules = self.target_modules.split(',')
        if self.modules_to_save:
            self.modules_to_save = self.modules_to_save.split(',')
        if self.layers_to_transform:
            self.layers_to_transform = [ int(x) for x in self.layers_to_transform.split(',') ]


def to_dict(obj): # can't use asdict, it is recursive; for shallow: https://docs.python.org/3/library/dataclasses.html
    return dict((field.name, getattr(obj, field.name)) for field in fields(obj))


if __name__ == "__main__":
    main()

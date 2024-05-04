import yaml
import argparse
import torch 
import datasets
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM
from transformers import DefaultDataCollator
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from transformers import Trainer, TrainingArguments
from preprocess_fns import sql_map_fn
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

import wandb
import os
wandb.login()

os.environ["WANDB_PROJECT"]="llama2-chat"
os.environ["WANDB_LOG_MODEL"] = "false" 
os.environ["WANDB_WATCH"]="all" 


def setup_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False)
    print(tokenizer.pad_token_id)
    tokenizer.pad_token_id=tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.add_bos_token = False
    return tokenizer


def setup_dataset(dataset_path,tokenizer,tokenizer_config):
    ds=datasets.load_from_disk(dataset_path) 
    sqldataset=ds.map(sql_map_fn, fn_kwargs={"tokenizer":tokenizer,"with_assistant_response":True, "tokenizer_config":tokenizer_config},remove_columns=['instruction', 'input', 'response'])
    return sqldataset


def setup_model(model_id, quantization_config, peft_config):
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
    return model


def main(config):
    tokenizer=setup_tokenizer(config["model"]["model_id"])

    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=config["model"]["quantization_config"]["llm_int8_threshold"],
        llm_int8_has_fp16_weight=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        llm_int8_enable_fp32_cpu_offload=True,
    )

    peft_config=LoraConfig(
        r=config["model"]["peft_config"]["r"],
        lora_alpha=config["model"]["peft_config"]["lora_alpha"],
        lora_dropout=config["model"]["peft_config"]["lora_dropout"],
        bias='none',
        task_type='CAUSAL_LM'
    )

    model=setup_model(config["model"]["model_id"], quantization_config=quantization_config, peft_config=peft_config)
    sql_dataset=setup_dataset(config["dataset"]["dataset_path"], tokenizer, config["tokenizer"]["tokenizer_config"])
    #collection_fns=DataCollatorForCompletionOnlyLM(response_template="[/INST]",instruction_template="[INST]" ,tokenizer=tokenizer)
    
    train_arg = TrainingArguments(**config["train_config"])
    trainer = Trainer(model=model, 
        args=train_arg, 
        train_dataset=sql_dataset["train"], 
        eval_dataset=sql_dataset["valid"], 
        )

    trainer.train()
    model.save_pretrained(config["model_output"])


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='Train Experiments')
    parser.add_argument('--configuration-file', default="config_exp_num_params.yml")
    args=parser.parse_args()
    with open(args.configuration_file, 'r') as file:
        config = yaml.safe_load(file)
    main(config)


import argparse
import yaml 

import torch 
import datasets
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig

from preprocess_fns import inf_map_fn 
import numpy as np


def setup_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

def setup_dataset(dataset_path,tokenizer):
    ds=datasets.load_from_disk(dataset_path) 
    sqldataset=ds.map(inf_map_fn, fn_kwargs={"tokenizer":tokenizer},remove_columns=['instruction','input'])
    return sqldataset

def setup_model(model_id,peft_model_id, quantization_config):
    model=AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    model.load_adapter(peft_model_id)
    return model

def evaluation(dataset, tokenizer, model, generation_config):
    exact_match=evaluate.load("exact_match")
    bleu_score=evaluate.load("bleu")
    rouge_score=evaluate.load("rouge")

    preds=list()
    labels=list()
    for item in dataset["test"]:
        length=len(item["input_formated"])-1
        input_ids=tokenizer(item["input_formated"],return_tensors="pt").to("cuda")    
        outputs = model.generate(input_ids["input_ids"], eos_token_id=tokenizer.eos_token_id, max_new_tokens=generation_config["max_new_tokens"], do_sample=True, top_p=generation_config["top_p"], temperature=generation_config["top_p"])
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        preds.append(response[0][length:])
        labels.append(item["response"])
        break

    match_scores=exact_match.compute(predictions=preds, references=labels,ignore_punctuation=True)
    print("exact_match:", round(match_scores["exact_match"],2)) 
    bleu_scores=bleu_score.compute(predictions=preds, references=labels)
    print(bleu_scores)
    rouge_scores=rouge_score.compute(predictions=preds, references=labels)
    print(rouge_scores)
    return 


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

    model=setup_model(config["model"]["model_id"], config["eval_config"]["peft_model_id"], quantization_config)
    sql_dataset=setup_dataset(config["dataset"]["dataset_path"],tokenizer)
    evaluation(sql_dataset, tokenizer, model, config["eval_config"]["generation_config"])


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='Model Evaluation')
    parser.add_argument('--configuration-file', default="config_exp_num_params.yml")
    args=parser.parse_args()
    with open(args.configuration_file, 'r') as file:
        config = yaml.safe_load(file)
    main(config)
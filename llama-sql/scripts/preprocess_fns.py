import torch 
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig

from transformers import Trainer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM


def sql_map_fn(element, tokenizer,with_assistant_response,tokenizer_config):
    system_prompt = (
        "You are a helpful programmer assistant that excels at SQL. "
        "When prompted with a task and a definition of an SQL table, you "
        "respond with a SQL query to retrieve information from the table. "
        "Don't explain your reasoning, only provide the SQL query."
    )

    user_prompt = "Task: {instruction}\nSQL table: {input}\nSQL query: "

    output = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format_map(element)},
    ]

    output.append({"role": "assistant", "content": element["response"]})
    
    formatted=tokenizer.apply_chat_template(output, tokenize=False)
    output_ids=tokenizer(formatted, max_length=tokenizer_config["max_length"], padding=tokenizer_config["padding"], truncation=tokenizer_config["truncation"])
    return {"input_ids": output_ids["input_ids"], "attention_mask": output_ids["attention_mask"], "label_ids": output_ids["input_ids"]}


def inf_map_fn(element, tokenizer):
    system_prompt = (
        "You are a helpful programmer assistant that excels at SQL. "
        "When prompted with a task and a definition of an SQL table, you "
        "respond with a SQL query to retrieve information from the table. "
        "Don't explain your reasoning, only provide the SQL query."
    )

    user_prompt = "Task: {instruction}\nSQL table: {input}\nSQL query: "

    output = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format_map(element)},
    ]

    input_formated=tokenizer.apply_chat_template(output, tokenize=False)
    return {"input_formated": input_formated}
import torch 
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig 
from transformers import AutoModelForCausalLM


quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    llm_int8_enable_fp32_cpu_offload=True,
)

device="cuda"
model_id= "../llama-sql-v2/meta-llama/Llama-2-7b-chat-hf"
peft_model_id= "../llama-sql-v2/output/exp_2.0/checkpoint-1/"
model = AutoModelForCausalLM.from_pretrained(model_id,  quantization_config=quantization_config, device_map="auto")
model.load_adapter(peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)

ds=datasets.load_from_disk("/root/llama-sql-v1/dataset/text-to-sql-v1-easy")
item=ds["valid"][0]

system_prompt = (
    "You are a helpful programmer assistant that excels at SQL. "
    "When prompted with a task and a definition of an SQL table, you "
    "respond with a SQL query to retrieve information from the table. "
    "Don't explain your reasoning, only provide the SQL query."
)
user_prompt = "Task: {instruction} \nSQL table: {input} \nSQL query: "

output = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt.format_map(item)},
]

input_formated=tokenizer.apply_chat_template(output, tokenize=False)
input_ids=tokenizer.apply_chat_template(output, return_tensors="pt").to(device)

outputs = model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_new_tokens=1000, do_sample=True, top_p=0.9,temperature=0.9)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)


print(f"\n\nCorrect response:\n{item['response']}")
print(f"\n\nLLM response:\n{response[0][len(input_formated)-1:]}")
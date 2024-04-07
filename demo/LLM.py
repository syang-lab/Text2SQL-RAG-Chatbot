from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

class SQL_LLM(LLM):
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        super().__init__()

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

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map="auto")
        self.model = self.model.eval()

    def _call(self, prompt: dict, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        
        system_prompt="[INST] You are a helpful programmer assistant that excels at SQL. When prompted with a task and a definition of an SQL table, you respond with a SQL query to retrieve information from the table. Don't explain your reasoning, only provide the SQL query."
        prompt=system_prompt+prompt+"[/INT]"
        print(prompt)
        
        input_ids=self.tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(input_ids.input_ids, eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=1000, do_sample=True, top_p=0.9,temperature=0.9)
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response[0][len(prompt):]


    @property
    def _llm_type(self) -> str:
        return "SQL_LLM"


# if __name__ == "__main__":
#     llm = SQL_LLM(model_path = "/root/data/model/meta-llama/Llama-2-7b-chat-hf")
#     prompts="Name the home team for carlton away team. CREATE TABLE table_name_77 ( home_team VARCHAR, away_team VARCHAR )"
#     result=llm.invoke(prompts)
#     print(result)
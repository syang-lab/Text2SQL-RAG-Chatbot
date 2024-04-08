### Project Overview
This project contains two parts: 
1. Instruction fine-tuning Llama model on Text-to-SQL dataset.
2. Constructed a vector database to enhance Llama’s performance utilizing the Retrieval-Augmented Generation (RAG) technique and the Langchain framework. And developed a Gradio chat web demo.
![download model](model_overview.png)

#### Instruction fine-tuning Llama model on Text-to-SQL dataset.
1. Data Preprocessing
<br>For instruction fine-tuning, an additional step required is to make the dataset following the Llama template. The template can be changed in general, but used the original template that Llama has been trained on will help obtain better performance.

    ```
    <s>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST] Answer </s><s>[INST] Prompt [/INST] Answer </s> 
    <s>[INST] Prompt [/INST] ....
    ```
    
The code corresponds to applying the template:

    ```
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
    ```
    
3. Model Memory Reduction and Speedup
<br>Training Llama 7B model will require (AdamW  8 bytes per parameter * 7 billion parameters) 56 GB of GPU memory with full precision. To reduce the memory consumption and speedup the training speed, exploying Quantation and LoRa method concurrently. The resulting number of trainable parameters is about 2.8 billion.
<br> a.Quantation method: representing weights and activations with lower-precision data types like 8-bit integers (int8)
<br> b.LoRa: inserting a smaller number of new weights into the model and only these are trained.

The following codes take the configuration of quantization and LoRa. 

```
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

model = get_peft_model(model, peft_config)

```

The configurations of quantization and LoRa are shown below: 

```
  peft_config:
    lora_alpha: 16
    lora_dropout: 0.1
    r: 64
    task_type: CAUSAL_LM
  quantization_config:
    bnb_4bit_compute_dtype: float16
    bnb_4bit_quant_type: nf4
    bnb_4bit_use_double_quant: true
    llm_int8_enable_fp32_cpu_offload: true
    llm_int8_has_fp16_weight: true
    llm_int8_threshold: 6.0
    load_in_4bit: true
    load_in_8bit: false
```

5. Training Speedup and Experiments
The deepspeed stage 2 is used for speedup training and model offload from GPU to CPU. Deepspeed is setup through the configuration file as shown below:




7. Evaluation Matrix
8. Challenged: challenging parts: a.tokenizer problem b.weight and bias problem.

#### Langchain RAG and Gradio Deployment
a.build vector database: the chunk size should be approperate.
b.LLmodel
c.gradio deployment

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
<br> The deepspeed stage 2 is used for speedup training and model offload from GPU to CPU. Deepspeed is setup through the configuration file as shown below:
```
  deepspeed:
    communication_data_type: fp16
    gradient_accumulation_steps: auto
    gradient_clipping: auto
    train_batch_size: auto
    train_micro_batch_size_per_gpu: auto
    zero_optimization:
      allgather_bucket_size: 500000000.0
      allgather_partitions: true
      contiguous_gradients: true
      offload_optimizer:
        device: cpu
        pin_memory: true
      overlap_comm: true
      reduce_bucket_size: 500000000.0
      reduce_scatter: true
      round_robin_gradients: true
      stage: 2
```

![training_loss](training_loss.png)

6. Evaluation Matrix
<br> In general pretrained large language model are evaluation through widly used benchmark datasets including Alpaca etc.. Here, the test poration of the original dataset is used for evaluation the performnace. In addition, the evaluation metrix including exact match, blue score and rouge score. Meanwhile, temperature, and topk and topp can be tuned to to boost the performance.

9. Challenging Debugging Parts
<br> Weight&bias and deepspeed will give the following bug ```AttributeError: 'Accelerator' object has no attribute 'deepspeed_config```, if ```os.environ["WANDB_LOG_MODEL"] =  "checkpoint" ```, for which the information is very confusing. It takes me long to understand why this is the case, because before start tracking with weight&bias, I did not see the problem. Though check the model sections by sections, I find the problem is either from deepspeed or from weight&bias. And eventually find the solution: set ```os.environ["WANDB_LOG_MODEL"] = False```

#### Langchain RAG and Gradio Deployment
1.Build vector database
<br> Build vector database use Langchain framework is very straight forward. Read the file, and split into chunks. Then load an embedding model to 

```
    from langchain.vectorstores import Chroma
    embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

    persist_directory = 'data_base/vector_db/chroma'

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
```

2. Create LLM Model 
<br> The model should inherent the LLM model in langchain

```
    from langchain.llms.base import LLM
    class SQL_LLM(LLM):
        tokenizer : AutoTokenizer = None
        model: AutoModelForCausalLM = None
        ...
```

3. Add Information to Prompt
Then pass the model and vector database to the RetrievalQA:

```
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm,
            retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}),
            return_source_documents=True,
            chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
```

3.Gradio Deployment
![gradio_deployment](gradio_deployment.png)







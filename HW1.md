## HW1
### LLM Application Working Flow:

![LLM Application Working Flow](LLM-Application-Working-Flow.png)

### LLM Model
#### Pretrained LLM Model
1. Model Selection 
2. Pre-train data preprocessing 
  2.1 datadistributation 
  2.2 data processing: (Format data)->(Clean data)->(Dedup data)->(Safe data).
3. Pre-train tokenization 

#### Training LLM Model
1. distributed training
2. experiment tracking 

#### Fine-tunning LLM Model
1. SFT(supervised fine-tunning): fine-tunning for downstream application
   
    a. define high level objective

   b. select model and tokenizer
     1. in general left padding is recommended, however LLaMA2 used right padding.

   c. select dataset, and preprocess data to follow the default template.
   
   1. convert dataset formate to xtuner formate, for example:
    
   2. add system prompt, and convert to template format
    
   d. evaluation matrix 

   e. select traing speedup
      1. Adapter 
      2. LoRA or QLoRA
      5. Framework: directly optimize memory and optimization--deepspeed: model scale, speed, scalibility

     
   f. inference speed up 
      1. Quantation to reduce the number of bits: GPU-AWQ/GPTQ, CPU-GGUF/GGML
      2. Pageattention
      3. Flashattention
      4. Framework:deepspeed(only zero-3)
    
    g. deployment 

3. RLHF(reinforce learning with human feedback)

#### Evaluation 
1. Evaluation Benchmark Dataset 
    1. Performance on Downstream Tasks:(1) 全面测试(),(2) 语言和知识,(3) 推理和数学,(4) 多种编程语言编程,(5) 长文本建模,(6) 工具利用。
    2. Performance on Alignment: 评估模型的对齐能力对于判断LLMs是否真正满足人类需求至关重要。
    
2. Evalution Matrix

#### Ablation Study

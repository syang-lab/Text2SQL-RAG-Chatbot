### HW 1
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

   c. select dataset, and preprocess data to follow the default template.
      For instance LLaMA chat template: 

   d. evaluation matrix 

   e. select traing speedup
      1. adapter 
      2. LoRA or QLoRA 
    
    f. inference speed up 
      1. Quantation to reduce the number of bits: GPU-AWQ/GPTQ, CPU-GGUF/GGML
      2. pageattention
      3. flashattention
    
    g. deployment 

3. RLHF(reinforce learning with human feedback)

#### Evaluation 
1. Evaluation Benchmark Dataset 
    1. Performance on Downstream Tasks:(1) 全面测试(),(2) 语言和知识,(3) 推理和数学,(4) 多种编程语言编程,(5) 长文本建模,(6) 工具利用。
    2. Performance on Alignment: 评估模型的对齐能力对于判断LLMs是否真正满足人类需求至关重要。
2. Evalution Matrix 
#### Ablation Study




### HW 1
#### Overview LLM working flow:

#### Pretrained LLM Model
1. Model Selection 
2. Pre-train data preprocessing 
  2.1 数据源分布 
  2.2 数据处理流程: 格式化数据(Format data)->清洗数据(Clean data)->去重数据(Dedup data)->安 全数据(Safe data).
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

   e. select traing speedup: 
      1. adapter 
      2. LoRA or QLoRA 
      3. flash attension or page attension 
      4. auantation? Deepzero
    
    f. inference speed up 
      1. Quantation 
    
    g. deployment 

3. RLHF(reinforce learning with human feedback) 

#### Evaluation 
1. Performance on Downstream Tasks:(1) 全面测试,(2) 语言和知识,(3) 推理和数学,(4) 多种编程语言编程,(5) 长文本建模,(6) 工具利用。
2. Performance on Alignment: 评估模型的对齐能力对于判断LLMs是否真正满足人类需求至关重要。

#### Ablation Study




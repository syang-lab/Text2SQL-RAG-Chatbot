### HW1
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
   
     For instance LLaMA chat template (prompt_template = PROMPT_TEMPLATE.llama2_chat):
    ```
    <bos_token>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST] Answer <eos_token><bos_token>[INST] Prompt [/INST] Answer <eos_token> \
    <bos_token>[INST] Prompt [/INST] ....
    ```
   xtuner setup the dataset configuration:
    ```
    prompt_template = PROMPT_TEMPLATE.llama2_chat
    dataset_map_fn=alpaca_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template)
    ```
   contains two steps:


    1. convert dataset formate to xtuner formate, for example:  
    ```
    def oasst1_map_fn(example):
        r"""Example before preprocessing:
            example['text'] = '### Human: Can you explain xxx'
                              '### Assistant: Sure! xxx'
                              '### Human: I didn't understand how xxx'
                              '### Assistant: It has to do with a process xxx.'
    
        Example after preprocessing:
            example['conversation'] = [
                {
                    'input': 'Can you explain xxx',
                    'output': 'Sure! xxx'
                },
                {
                    'input': 'I didn't understand how xxx',
                    'output': 'It has to do with a process xxx.'
                }
            ]
        """
        data = []
        for sentence in example['text'].strip().split('###'):
            sentence = sentence.strip()
            if sentence[:6] == 'Human:':
                data.append(sentence[6:].strip())
            elif sentence[:10] == 'Assistant:':
                data.append(sentence[10:].strip())
        if len(data) % 2:
            # The last round of conversation solely consists of input
            # without any output.
            # Discard the input part of the last round, as this part is ignored in
            # the loss calculation.
            data.pop()
        conversation = []
        for i in range(0, len(data), 2):
            single_turn_conversation = {'input': data[i], 'output': data[i + 1]}
            conversation.append(single_turn_conversation)
        return {'conversation': conversation}
    
    ```
   2. add system prompt, and convert to template format
    
    ```
    def template_map_fn(example, template):
        conversation = example.get('conversation', [])
        for i, single_turn_conversation in enumerate(conversation):
            input_text = ''
            input = single_turn_conversation.get('input', '')
            if input != '' and input is not None:
                input = template.INSTRUCTION.format(input=input, round=i + 1)
                input_text += input
                instruction_postfix = ''
            else:
                instruction_postfix = template.INSTRUCTION.split('{input}')[-1]
            system = single_turn_conversation.get('system', '')
            if system != '' and system is not None:
                system = template.SYSTEM.format(system=system)
                input_text = system + input_text
            single_turn_conversation['input'] = input_text + instruction_postfix
        return {'conversation': conversation}
    
    
    def template_map_fn_factory(template):
        return partial(template_map_fn, template=template)
    ```
   d. evaluation matrix 

   e. select traing speedup
      1. adapter 
      2. LoRA or QLoRA
      3. framework:directly optimize memory and optimization--deepspeed: model scale, speed, scalibility
    
   f. inference speed up 
      1. Quantation to reduce the number of bits: GPU-AWQ/GPTQ, CPU-GGUF/GGML
      2. pageattention
      3. flashattention
      4. framework:deepspeed(only zero-3)
    
    g. deployment 

3. RLHF(reinforce learning with human feedback)

#### Evaluation 
1. Evaluation Benchmark Dataset 
    1. Performance on Downstream Tasks:(1) 全面测试(),(2) 语言和知识,(3) 推理和数学,(4) 多种编程语言编程,(5) 长文本建模,(6) 工具利用。
    2. Performance on Alignment: 评估模型的对齐能力对于判断LLMs是否真正满足人类需求至关重要。
2. Evalution Matrix

#### Ablation Study

- [ ] 1.download hugging face pretrained model
- [ ] 2.download dataset
- [ ] 3.run the training
- [ ] 5.how does LoRA and deepspeed work? how does inference speedup work.
- [ ] 4.RAG or agent: how does RAG or agent work?
- [ ] 5.Modify to flexiable framework for training and deployment.
- [ ] 3.add evaluation matrix, why no evaluation matrix calculation in the fine-tunning?




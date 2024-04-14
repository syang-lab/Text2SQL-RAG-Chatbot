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
      1. Adapter 
      2. LoRA or QLoRA
      training LoRA

      ```
      from peft import LoraConfig, TaskType
      lora_config = LoraConfig(
          r=16,
          target_modules=["q_proj", "v_proj"],
          task_type=TaskType.CAUSAL_LM,
          lora_alpha=32,
          lora_dropout=0.05
      )
      model.add_adapter(peft_config)
      ```

      3. Quantation: bitsandbytes enables accessible large language models via k-bit quantization for PyTorch.
       to load and quantize a model to 4-bits and use the bfloat16 data type for compute:

       ```
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model_4bit = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloom-1b7",
            device_map=device_map,
            quantization_config=quantization_config,
        )
       ```

        quantation of optimizer
   
      ```
      from bitsandbytes.optim import PagedAdamW32bit
      ```
      <br> 量化的最主要目的是降低显存占用，主要包括两方面的显存：模型参数和中间过程计算结果。前者对应《3.2 W4A16 量化》，后者对应《3.1 KV Cache 量化》。
      <br> 量化在降低显存的同时，一般还能带来性能的提升，因为更小精度的浮点数要比高精度的浮点数计算效率高，而整型要比浮点数高很多。
      5. Framework: directly optimize memory and optimization--deepspeed: model scale, speed, scalibility

     ```
      from transformers.integrations import HfDeepSpeedConfig
      from transformers import AutoModel
      import deepspeed
      
      ds_config = {...}  # deepspeed config object or path to the file
      # must run before instantiating the model to detect zero 3
      dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
      model = AutoModel.from_pretrained("openai-community/gpt2")
      engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
      deepspeed --num_gpus=2 your_program.py <normal cl args> --do_eval --deepspeed ds_config.json

     ```
      
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

<br>斯坦福大学提出了较为系统的评测框架HELM，从准确性，安全性，鲁棒性和公平性等维度开展模型评测。
<br>纽约大学联合谷歌和Meta提出了SuperGLUE评测集，从推理能力，常识理解，问答能力等方面入手，构建了包括8个子任务的大语言模型评测数据集。
<br>加州大学伯克利分校提出了MMLU测试集，构建了涵盖高中和大学的多项考试，来评估模型的知识能力和推理能力。
<br>谷歌也提出了包含数理科学，编程代码，阅读理解，逻辑推理等子任务的评测集Big-Bench，涵盖200多个子任务，对模型能力进行系统化的评估。

2. Evalution Matrix

#### Ablation Stud
#### tmux 
*apt update -y
#### tmux 
*apt install tmux -y
#### create session
tmux new -s exp1
#### return to original enviroment 
control+b d
#### tmux attach -t exp1
#### tmux list-sessions
* 1.change the map, apply tokenizer default template, do not apply tokenizer. * 2. put tokenizer to trainer 






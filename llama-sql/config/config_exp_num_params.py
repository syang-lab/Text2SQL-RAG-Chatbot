import yaml
import torch 

config={
    "model":{
            "model_id": "meta-llama/Llama-2-7b-chat-hf",
            "quantization_config":{
                "load_in_4bit":True,
                "load_in_8bit":False,
                "llm_int8_threshold":6.0,
                "llm_int8_has_fp16_weight":True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant":True,
                "bnb_4bit_quant_type":'nf4',
                "llm_int8_enable_fp32_cpu_offload":True,
            },

            "peft_config":{
                "r":64,
                "lora_alpha":16,
                "lora_dropout":0.1,
                "task_type":'CAUSAL_LM'
            },

    },

    "tokenizer":{"tokenizer_config":{
        "max_length":2048,
        "padding":True,
        "truncation":True,
        }
    },
    
    "dataset":{
        "dataset_path":"/root/llama-sql-v1/dataset/text-to-sql-v1-easy"
    },

    "train_config":{
        "output_dir": "./output/exp_1.0",
        "dataloader_drop_last": True,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "max_steps":1,
        "bf16": True,
        "max_grad_norm":1,
        "gradient_accumulation_steps":16,
        "evaluation_strategy":"steps",
        "eval_steps":1,
        "logging_strategy": "no",
        "logging_steps": 1,
        "save_strategy": "no",
        "save_steps": 10,
        "load_best_model_at_end":True,
        "metric_for_best_model":"eval_loss",
        "greater_is_better":False, 
        "deepspeed":{
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
                "round_robin_gradients": True
            },

            "train_micro_batch_size_per_gpu": "auto",
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "communication_data_type": "fp16"
        },

        "learning_rate":2e-4,
        "weight_decay":1e-2,
        "adam_beta1":0.9,
        "adam_beta2":0.999,
        "adam_epsilon":1e-08,
        "report_to":"wandb",
        "run_name":"exp_1.0",
    },
    
    "model_output":{ "./output/exp_1.0_model"}
}

with open('config_exp_num_params.yml', 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
import argparse
import wandb

from .data_loader import get_data
from .reward import combined_reward

def main(args):
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = get_data(args.dataset1_path, args.dataset2_path, type="grpo")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        log_on_each_node=args.log_on_each_node,
        use_vllm=False,
        vllm_gpu_memory_utilization=0.6,
        report_to="wandb",
        deepspeed=args.deepspeed_path,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            lora_dropout=args.lora_dropout,
        )

    if args.use_lora:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[combined_reward],
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
        )
    else:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[combined_reward],
            args=training_args,
            train_dataset=train_dataset,
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.tokenizer.save_pretrained(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with GRPO")
    parser.add_argument("--dataset1_path", type=str, required=True, help="Path to the first dataset")
    parser.add_argument("--dataset2_path", type=str, default=None, help="Path to the second dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the model")
    parser.add_argument("--deepspeed_path", type=str, default="./deepspeed_config.json", help="Path to the deepspeed config file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Pretrained model name")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for Adam optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps frequency")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=128, help="Maximum completion length")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=5000, help="Steps interval to save the model")
    parser.add_argument("--log_on_each_node", action='store_true', help="Log on each node in distributed training")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--use_lora", type=int, default=0, help="Whether to use LoRA for training")
    parser.add_argument("--wandb_project", type=str, default="amr-training", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default="grpo-run", help="WandB run name")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

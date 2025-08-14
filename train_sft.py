import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import os

from .data_loader import get_data
from .reward import compute_smatch_f1


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_dataset = get_data(args.dataset1_path, args.dataset2_path, type="sft")
    # split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # train_dataset = split_dataset["train"]
    # eval_dataset = split_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    ).to(device)

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Compute SMATCH F1
        f1 = compute_smatch_f1(decoded_labels, decoded_preds)
        return {"smatch_f1": f1}

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=2,
        report_to="none",
        completion_only_loss=False,
        deepspeed=args.deepspeed_path,
        # eval_strategy="steps",
        # eval_steps=args.eval_steps
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        args=training_args,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset1_path", type=str, required=True)
    parser.add_argument("--dataset2_path", type=str, default=None, help="Optional second dataset path for concatenation")
    parser.add_argument("--output_dir", type=str, default="./sft_lora_output")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    # Training parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--deepspeed_path", type=str, default=None)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=1024)
    parser.add_argument("--eval_steps", type=int, default=500)

    # LoRA parameters
    parser.add_argument("--use_lora", type=int, default=0, help="Use LoRA for training")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

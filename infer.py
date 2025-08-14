from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import penman
from penman.models.noop import NoOpModel
import time
import re
import pandas as pd
import ast
from typing import List, Tuple

from .data_processing import *
from .postprocessing import *
from .prompt import SYSTEM_PROMPT

class QwenReasoner:
    def __init__(self, model_name="ViQwen2-1.5B-rerank-GRPO", device="cuda:0"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def inference(self, prompt: str, max_new_tokens: int = 2048, is_extract_amr: bool = False) -> str:
        user_prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Chuyển câu sau thành biểu diễn AMR dạng chuỗi PENMAN một dòng theo đúng quy tắc trên."
                    f"Câu: {prompt}\n"
                )
        messages = [
            {"role": "system", "content": "Bạn là trợ lý chuyên gia AMR, nhiệm vụ của bạn là chuyển đổi câu tiếng Việt thành biểu diễn AMR dạng chuỗi PENMAN theo định dạng chuẩn."},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        if is_extract_amr:
            return self.extract_answer(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    @staticmethod
    def extract_answer(text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
if __name__ == "__main__":
    reasoner = QwenReasoner(model_name="/home/fit02/dien-workspace/viamr/outputs/Qwen-1.7B-SFT-1")
    prompt = vi_tokenize('không phải phép thần kỳ biến quê hương từ chỗ cay cực nhất vinh dự nhận danh hiệu tập thể " anh hùng lao động thời kỳ đổi mới " sau 15 năm đổi mới .')
    start_time = time.time()
    while True:
        answer = reasoner.inference(prompt, is_extract_amr=True)
        try:
            graph = penman.decode(answer)
            amr_str = penman.encode(graph)
            break
        except Exception as e:
            print(f"[Error] Cannot decode AMR (try {retry_count+1})")
            amr_str = "fail"
            retry_count += 1
    try:
        amr_str = dedup_and_tidy(amr_str, None)
        amr_str = fix_amr_vars(amr_str)
        amr_str = balance_parens(amr_str)
        amr_str = process_amr_general(amr_str)
        print(f"[Success] Processed AMR")
        graph = penman.decode(amr_str)
        amr_str = penman.encode(graph)
    except Exception as e:
        print(f"[Error] Failed to process AMR after retries: {e}")
        amr_str = "fail"
    end_time = time.time()
    print(f"Answer: {amr_str}")
    print(f"Inference Time: {end_time - start_time:.2f} seconds")

import re
import smatch

from .postprocessing import *

def get_amr_match(amr1, amr2):
    vals = smatch.get_amr_match(amr1, amr2)
    smatch.match_triple_dict.clear()
    return vals

def compute_smatch_f1(gold_str, pred_str):
    try:
        M, T, G = get_amr_match(gold_str, pred_str)
        precision = M / T if T else 0
        recall = M / G if G else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    except Exception as e:
        print(e)
        f1, precision, recall = 0.0, 0.0, 0.0
    return f1, precision, recall

def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", text.strip(), flags=re.DOTALL)
    return match.group(1).strip() if match else None

def check_valid_format(text: str) -> bool:
    return bool(text and re.search(r"<answer>.*?</answer>", text.strip(), flags=re.DOTALL))

def check_balanced_parens(s: str) -> bool:
    """Kiểm tra đóng mở dấu ngoặc tròn."""
    stack = 0
    for ch in s:
        if ch == '(':
            stack += 1
        elif ch == ')':
            stack -= 1
            if stack < 0:
                return False
    return stack == 0

def check_unique_vars(amr_str: str) -> bool:
    """Kiểm tra biến không trùng (x1, x2, ...)."""
    vars_found = re.findall(r"\((\w+)\s*/", amr_str)
    return len(vars_found) == len(set(vars_found))

def check_var_word_conflict(amr_str: str) -> bool:
    """
    Trả về True nếu TẤT CẢ biến và từ đều có ký tự đầu giống nhau (ví dụ: b1 / bi_kịch).
    Nếu có ít nhất 1 cặp không trùng -> trả về False.
    """
    matches = re.findall(r"\((\w+)\s*/\s*([^\s)]+)", amr_str)
    if not matches:
        return False 
    
    for var, word in matches:
        if not word or var[0].lower() != word[0].lower():
            return False
    return True

def combined_reward(prompts, completions, answers, **kwargs) -> list[float]:
    scores = []
    for completion, gold in zip(completions, answers):
        response_text = completion[0]['content'].strip()

        # Extract & postprocess prediction
        try:
            pred_answer = extract_answer(response_text)
            pred_answer = dedup_and_tidy(pred_answer, None)
            pred_answer = fix_amr_vars(pred_answer)
            pred_answer = balance_parens(pred_answer)
            pred_answer = process_amr_general(pred_answer)
        except Exception:
            pred_answer = ''

        gold_answer = extract_answer(gold) or gold.strip()

        format_score = 0.1 if check_valid_format(response_text) else 0.0
        paren_score = 0.1 if pred_answer and check_balanced_parens(pred_answer) else 0.0
        unique_var_score = 0.1 if pred_answer and check_unique_vars(pred_answer) else 0.0
        var_word_conflict_score = 0.1 if pred_answer and check_var_word_conflict(pred_answer) else 0.0

        smatch_f1 = 0
        if pred_answer:
            smatch_f1, _, _ = compute_smatch_f1(gold_answer, pred_answer)
        smatch_score = 0.6 * smatch_f1

        total_score = format_score + paren_score + unique_var_score + var_word_conflict_score + smatch_score
        total_score = min(total_score, 1.0) 

        scores.append(total_score)

        print(f"Format: {format_score:.2f}, Parens: {paren_score:.2f}, Unique vars: {unique_var_score:.2f}, "
              f"No var-word conflict: {var_word_conflict_score:.2f}, Smatch: {smatch_f1:.4f} ({smatch_score:.2f}), Total: {total_score:.4f}")

    return scores
if __name__ == "__main__":
    gold_str = "<answer>(x1 / bi_kịch :domain (x2 / chỗ :mod (x3 / đó)))</answer>"
    pred_str = "<answer>(x1 / bi_kịch :domain (x2 / chỗ :mod (x3 / đó)))</answer>"

    print("Extracted gold:", extract_answer(gold_str))
    print("Extracted pred:", extract_answer(pred_str))

    score_list = combined_reward(
        prompts=None,
        completions=[[{"content": pred_str}]],
        answers=[gold_str]
    )
    print(f"Final combined reward: {score_list[0]:.4f}")

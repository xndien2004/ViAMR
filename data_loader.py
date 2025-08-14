import pandas as pd
from datasets import Dataset
import json

from .prompt import *
from .data_processing import *


# def get_data(train_path1, dep_path, train_path2=None, type="grpo"):
#     df = read_amr(train_path1)
#     if train_path2:
#         df2 = read_amr(train_path2)
#         df = pd.concat([df, df2], ignore_index=True)

#     with open(dep_path, "r", encoding="utf-8") as f_json:
#         deps_data = json.load(f_json)  

#     if len(df) != len(deps_data):
#         print("Warning: Dataframe and dependencies list have different lengths.")
#     def process_df(df, deps_list):
#         processed = []
#         for idx, row in df.iterrows():
#             dep_entry = deps_list[idx] 
#             sentence = dep_entry["sentence"]
#             dep_info = dep_entry["dependency"]
#             dep_str = str(dep_info)

#             if type == "grpo":  
#                 user_prompt = (
#                     f"Chuyển câu sau thành biểu diễn AMR dạng PENMAN.\n"
#                     f"Câu: {sentence}\n"
#                     f"Dependency: {dep_str}"
#                 )
#                 prompt = [
#                     {"role": "system", "content": SYSTEM_PROMPT},
#                     {"role": "user", "content": user_prompt}
#                 ]
#                 processed.append({
#                     "prompt": prompt,
#                     "answers": row['actions']
#                 })
#             else:
#                 user_prompt = (
#                     f"{SYSTEM_PROMPT}\n\n"
#                     f"Chuyển câu sau thành biểu diễn AMR dạng PENMAN.\n"
#                     f"Câu: {sentence}\n"
#                     f"Dependency: {dep_str}"
#                 )
#                 processed.append({
#                     "prompt": [{"role": "user", "content": user_prompt}],
#                     "completion": [
#                         {"role": "assistant", "content": f"<answer>{row['actions']}</answer>"}
#                     ]
#                 })
#         return Dataset.from_list(processed)

#     train_dataset = process_df(df, deps_data)
#     return train_dataset

def get_data(train_path1, train_path2=None, type="grpo"):
    df = read_amr_direct(train_path1)
    if train_path2:
        df2 = read_amr_direct(train_path2)
        df = pd.concat([df, df2], ignore_index=True)

    def process_df(df):
        processed = []
        max_length_input = 0
        max_length_output = 0
        for idx, row in df.iterrows():
            sentence = row["query"]
            user_prompt = (
                f"Chuyển câu sau thành biểu diễn AMR dạng chuỗi PENMAN một dòng theo đúng quy tắc trên."
                f"Câu: {sentence}\n"
            )
            max_length_input = max(max_length_input, len(user_prompt.split(" ")))
            max_length_output = max(max_length_output, len(row['amr'].split(" ")))
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            if type == "grpo":  
                
                processed.append({
                    "prompt": prompt,
                    "answers": row['amr']
                })
            else:
                # user_prompt = (
                #     f"{SYSTEM_PROMPT}\n\n"
                #     f"Chuyển câu sau thành biểu diễn AMR dạng chuỗi PENMAN một dòng theo đúng quy tắc trên."
                #     f"Câu: {sentence}\n"
                # )
                # max_length_input = max(max_length_input, len(user_prompt.split(" ")))
                # max_length_output = max(max_length_output, len(row['amr'].split(" ")))
                processed.append({
                    "prompt": prompt,
                    "completion": [
                        {"role": "assistant", "content": f"<answer>{row['amr']}</answer>"}
                    ]
                })

        print(f"Max input length: {max_length_input}, Max output length: {max_length_output}")
        return Dataset.from_list(processed)

    train_dataset = process_df(df)
    return train_dataset

import penman
from penman.models.noop import NoOpModel
import os
from .infer import QwenReasoner
from .postprocessing import *
from .data_processing import *

def main(args):
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    else:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if args.my_test:
        df = read_amr_direct(args.input_file)
        lines = df["query"].tolist()
    else:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

    model = QwenReasoner(model_name=args.model_name)

    with open(args.output_file, "a", encoding="utf-8") as out_f:
        for idx, line in enumerate(lines):
            amr_str = "fail"
            retry_count = 0
            max_retries = 100
            while retry_count < max_retries:
                predict = model.inference(line.lower(), is_extract_amr=True)
                try:
                    graph = penman.decode(predict)
                    amr_str = penman.encode(graph)
                    break
                except Exception as e:
                    print(f"[Error] Cannot decode AMR (try {retry_count+1})")
                    amr_str = "fail"
                    retry_count += 1
            try:
                # amr_str = dedup_and_tidy(amr_str, None)
                # amr_str = fix_amr_vars(amr_str)
                # amr_str = balance_parens(amr_str)
                amr_str = process_amr_general(amr_str)
                amr_str = remove_single_prop_nodes(amr_str)
                print(f"[Success] Processed AMR")
                graph = penman.decode(amr_str)
                amr_str = penman.encode(graph)
            except Exception as e:
                print(f"[Error] Failed to process AMR after retries: {e}")

            if has_duplicate_nodes(amr_str):
                print(f"[Warning] AMR has duplicate nodes: {amr_str}")
            out_f.write(f"#::snt {idx} {line}\n")
            out_f.write(f"{amr_str}\n\n")
            out_f.flush()

            print(f"Processed {idx}: {line} (Retries: {retry_count})")

    print(f"Save completed. Results saved to {args.output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process AMR data.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen-7B")
    parser.add_argument("--my_test", type=int, default=0, help="Use for my test data")

    args = parser.parse_args()
    main(args)

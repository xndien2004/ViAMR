import argparse

from .data_processing import read_amr
from .reward import compute_smatch_f1
def main(args):
    predict_df = read_amr(args.predict_file)
    gold_df = read_amr(args.gold_file)
    predicts = predict_df["target"].tolist()
    golds = gold_df["target"].tolist()
    print(f"Number of predictions: {len(predicts)}, Number of golds: {len(golds)}")

    f1_scores = []
    precision_scores = []
    recall_scores = []
    for i in range(len(predict_df)):
        predict_amr = predicts[i]
        gold_amr = golds[i]
        f1, p, r = compute_smatch_f1(predict_amr, gold_amr)
        f1_scores.append(f1)
        precision_scores.append(p)
        recall_scores.append(r) 
    
    f1_avg = sum(f1_scores) / len(f1_scores)
    precision_avg = sum(precision_scores) / len(precision_scores)
    recall_avg = sum(recall_scores) / len(recall_scores)
    print(f"F1 Score: {f1_avg:.4f}, Precision: {precision_avg:.4f}, Recall: {recall_avg:.4f}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Compute AMR scores.")
    arg_parser.add_argument("--predict_file", type=str, required=True, help="Path to the predicted AMR file.")
    arg_parser.add_argument("--gold_file", type=str, required=True, help="Path to the gold AMR file.")
    args = arg_parser.parse_args()
    main(args)

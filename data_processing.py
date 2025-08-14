import pandas as pd
import re
import io
import contextlib
import penman
from penman.models.noop import NoOpModel


def penman_to_one_line(penman_str):
    lines = penman_str.strip().split('\n')
    one_line = ' '.join(line.strip() for line in lines)
    one_line = re.sub(r'\s+', ' ', one_line)
    return one_line


def fix_missing_closing_brackets(graph_str):
    open_count = graph_str.count('(')
    close_count = graph_str.count(')')
    missing = open_count - close_count
    if missing > 0:
        graph_str += ')' * missing
    return graph_str


def fix_multiword_nodes(graph_str):
    def repl(match):
        phrase = match.group(1)
        phrase_fixed = phrase.replace(' ', '_')
        return '/ ' + phrase_fixed
    pattern = r'/ ([^\(\):]+)'
    fixed_str = re.sub(pattern, repl, graph_str)
    return fixed_str


def decode_with_warnings(graph_str, sent):
    f = io.StringIO()
    with contextlib.redirect_stderr(f):
        try:
            graph = penman.decode(graph_str, model=NoOpModel())
            warnings = f.getvalue()
            if warnings.strip():
                print(f"Warning(s) during decoding sentence: {sent}")
                print(warnings)
            return graph, None
        except Exception as e:
            return None, e


def read_amr_direct(filename, one_line=True):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    queries = []
    amr_list = []
    current_sent = None
    current_graph_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("#::snt"):
            # save previous block
            if current_sent is not None and current_graph_lines:
                graph_str = "\n".join(current_graph_lines).strip()
                graph_str = fix_missing_closing_brackets(graph_str)
                graph_str = fix_multiword_nodes(graph_str)
                graph, error = decode_with_warnings(graph_str, current_sent)
                if not error:
                    amr_str = penman.encode(graph, model=NoOpModel())
                    if one_line:
                        amr_str = penman_to_one_line(amr_str)
                    queries.append(current_sent)
                    amr_list.append(amr_str)
                current_graph_lines = []
            current_sent = line[len("#::snt"):].strip()
        elif line == "":
            continue
        else:
            current_graph_lines.append(line)

    # process last block
    if current_sent is not None and current_graph_lines:
        graph_str = "\n".join(current_graph_lines).strip()
        graph_str = fix_missing_closing_brackets(graph_str)
        graph_str = fix_multiword_nodes(graph_str)
        graph, error = decode_with_warnings(graph_str, current_sent)
        if not error:
            amr_str = penman.encode(graph, model=NoOpModel())
            if one_line:
                amr_str = penman_to_one_line(amr_str)
            queries.append(current_sent)
            amr_list.append(amr_str)

    df = pd.DataFrame({"query": queries, "amr": amr_list})
    return df


if __name__ == "__main__":
    df = read_amr_direct("/home/fit02/dien-workspace/viamr/src/data/train_amr_1.txt")
    for index, row in df.head(10).iterrows():
        print(f"Query: {row['query']}")
        print(f"AMR: {row['amr']}")
        print("-" * 40)

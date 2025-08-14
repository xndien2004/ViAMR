import random

def read_amr_blocks(file_path):
    """Read AMR file into a list of blocks"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    blocks = content.split("\n\n")
    return [b.strip() for b in blocks if b.strip()]

def split_blocks(blocks, test_ratio=0.2):
    """Split into train/test sets"""
    random.shuffle(blocks)
    test_size = int(len(blocks) * test_ratio)
    test_blocks = blocks[:test_size]
    train_blocks = blocks[test_size:]
    return train_blocks, test_blocks

def write_blocks(file_path, blocks):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(blocks) + "\n")

if __name__ == "__main__":
    file1 = "/home/fit02/dien-workspace/viamr/src/data/train_amr_1.txt"
    file2 = "/home/fit02/dien-workspace/viamr/src/data/train_amr_2.txt"
    test_ratio = 0.15

    train1, test1 = split_blocks(read_amr_blocks(file1), test_ratio)
    train2, test2 = split_blocks(read_amr_blocks(file2), test_ratio)

    train_all = train1 + train2
    test_all = test1 + test2

    write_blocks("/home/fit02/dien-workspace/viamr/src/data/train.txt", train_all)
    write_blocks("/home/fit02/dien-workspace/viamr/src/data/test.txt", test_all)

    print(f"Train size: {len(train_all)}, Test size: {len(test_all)}")

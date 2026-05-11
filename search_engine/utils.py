def load_vocabulary(filename="golden_vocabulary.txt"):
    print(f"Loading vocabulary from {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        vocab = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(vocab)} words.")
    return vocab

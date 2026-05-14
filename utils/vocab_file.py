from config import VOCAB_FILE


def load_vocab_from_file():
    print(f"Loading vocabulary from {VOCAB_FILE}...")
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(vocab)} words.")
    return vocab


def save_vocab_to_file(vocab: set):
    print(f"Saving vocabulary to {VOCAB_FILE}...")
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        for word in sorted(vocab):
            f.write(f"{word}\n")
    print(f"Saved {len(vocab)} words.")

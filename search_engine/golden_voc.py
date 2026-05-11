from db_models import db, Chunk


def analyze_and_filter_vocabulary():
    hash_counts = {}
    total_chunks = 0
    query = Chunk.select(Chunk.compressed_json).iterator()

    for chunk in query:
        total_chunks += 1
        data = chunk.get_data()
        for word in data.keys():
            h = hash(word)
            if h in hash_counts:
                hash_counts[h] += 1
            else:
                hash_counts[h] = 1

        if total_chunks % 50000 == 0:
            print(f"Purging hash counts with frequency <=  at {total_chunks} chunks...")
            keys_to_delete = [h for h, count in hash_counts.items() if count <= 1]
            for h in keys_to_delete:
                del hash_counts[h]
            print(f"Hash counts purged. Remaining unique hashes: {len(hash_counts)}")

        if total_chunks % 10000 == 0:
            print(f"Processed {total_chunks} chunks...")
            print(f"Unique word hashes so far: {len(hash_counts)}")

    MIN_DF = 5
    MAX_DF = int(total_chunks * 0.50)
    MAX_FEATURES = 350000

    valid_items = [
        (h, count) for h, count in hash_counts.items() if MIN_DF <= count <= MAX_DF
    ]
    valid_items.sort(key=lambda x: x[1], reverse=True)
    print(f"Total valid hashes after applying DF thresholds: {len(valid_items)}")
    valid_hashes = {h for h, _ in valid_items[:MAX_FEATURES]}
    print(f"Top {MAX_FEATURES} valid hashes selected based on frequency.")

    hash_counts.clear()
    del hash_counts

    filtered_words = set()
    query = Chunk.select(Chunk.compressed_json).iterator()
    print("Filtering words based on valid hashes...")
    iter = 0
    for chunk in query:
        iter += 1
        if not valid_hashes:
            break
        data = chunk.get_data()
        for word in data.keys():
            h = hash(word)
            if h in valid_hashes:
                filtered_words.add(word)
                valid_hashes.remove(h)
                if not valid_hashes:
                    break
        if iter % 10000 == 0:
            print(f"Processed {iter} chunks for filtering...")
    print(f"Filtering complete. Final vocabulary size: {len(filtered_words)}")

    return filtered_words


if __name__ == "__main__":
    db.connect()

    golden_vocabulary = analyze_and_filter_vocabulary()

    filename = "golden_vocabulary.txt"
    print(f"Saving golden vocabulary to {filename}...")

    with open(filename, "w", encoding="utf-8") as f:
        for word in sorted(golden_vocabulary):
            f.write(word + "\n")

    print(
        f"Golden vocabulary saved to {filename}. Total words: {len(golden_vocabulary)}"
    )

from db.models import db, Chunk
from utils.vocab_file import save_vocab_to_file
from config import MIN_DF, MAX_DF_RATIO, MAX_FEATURES
from config import FLUSH_INTERVAL, PRINT_INTERVAL, MIN_DF_FLUSH_THRESHOLD


def analyze_and_filter_vocabulary():
    """
    Function creates a golden vocabulary.
    It iterates through all chunks in the database.
    Later using chunk bag of words (words are normalized and preprocessed using Porter Stemmer 2),
    it calculates frequency of each word across all chunks.
    After that it applies DF thresholds to filter out very common and very rare words.
    Finally it returns a set of filtered words as the golden vocabulary.
    """
    hash_counts = {}
    total_chunks = 0
    query = Chunk.select(Chunk.compressed_json).iterator()

    for chunk in query:
        total_chunks += 1
        data = chunk.get_json()
        for word in data.keys():
            h = hash(word)
            if h in hash_counts:
                hash_counts[h] += 1
            else:
                hash_counts[h] = 1

        if total_chunks % FLUSH_INTERVAL == 0:
            print(
                f"Purging hash counts with frequency <= {MIN_DF_FLUSH_THRESHOLD}  at {total_chunks} chunks..."
            )
            keys_to_delete = [
                h for h, count in hash_counts.items() if count <= MIN_DF_FLUSH_THRESHOLD
            ]
            for h in keys_to_delete:
                del hash_counts[h]
            print(f"Hash counts purged. Remaining unique hashes: {len(hash_counts)}")

        if total_chunks % PRINT_INTERVAL == 0:
            print(f"Processed {total_chunks} chunks...")
            print(f"Unique word hashes so far: {len(hash_counts)}")

    MAX_DF = MAX_DF_RATIO(total_chunks)

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
        data = chunk.get_json()
        for word in data.keys():
            h = hash(word)
            if h in valid_hashes:
                filtered_words.add(word)
                valid_hashes.remove(h)
                if not valid_hashes:
                    break
        if iter % PRINT_INTERVAL == 0:
            print(f"Processed {iter} chunks for filtering...")
    print(f"Filtering complete. Final vocabulary size: {len(filtered_words)}")

    return filtered_words


if __name__ == "__main__":
    db.connect()
    golden_vocabulary = analyze_and_filter_vocabulary()
    save_vocab_to_file(golden_vocabulary)
    db.close()

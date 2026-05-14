from search_engine.search_model import SearchModel
from search_engine.tfidf_model_svd import TfidfSvdSearchModel
from search_engine.rag_model import RAGSearchModel
from config import BM25_DIR, TFIDF_DIR  # Default BM25-SVD


class TerminalUI:
    def __init__(self):
        self.search_model: SearchModel = RAGSearchModel()

    def set_search_model(self, model: SearchModel):
        self.search_model = model

    def run(self):
        print("Welcome to the Search Engine Terminal UI!")
        print("Type 'exit' to quit.")
        while True:
            query = input("\nEnter your search query: ")
            if query.lower() == "exit":
                print("Goodbye!")
                break
            results = self.search_model.search(query, top_n=10)
            if not results:
                print("No results found.")
            else:
                self.display_results(results)

    def display_results(self, results):
        for dict_result in results:
            print(f"\nBook Title: {dict_result['book_title']}")
            print(f"Author: {dict_result['book_author']}")
            print(f"URL: {dict_result['book_url']}")
            print(f"Image URL: {dict_result['book_image_url']}")
            print(f"Chunk ID: {dict_result['chunk_id']}")
            print(f"Chunk Text: {dict_result['chunk_text']}")
            print(f"Chunk URL: {dict_result['chunk_url']}")
            print(f"Similarity Score: {dict_result['score']:.4f}")


if __name__ == "__main__":
    ui = TerminalUI()
    ui.run()

# Gutenberg Project Search Engine
Project of search engine for MOWNIT classes.

## Scrapper
For scrapper i used framework scrapy, all configuration needed to run is in the `config.py` file. To run scrapper run this command
in the working dircetory:
```sh
scrapy crawl gutenberg_org
```

During the scraping process, all documents are stored in an SQLite database. Before chunking, the text is normalized by removing punctuation and stop words. The documents are then split into chunks of 100 words with an overlap of 20 words. Each chunk is further normalized using the Porter Stemmer algorithm. For each chunk, a bag-of-words representation is created, and both the raw text and the bag-of-words are stored as blobs in the database.

## Data preprocessing
First step of data preprocessing is creation of the golden vocabulary set.
You do it via command:
```sh
python -m search_engine.golden_voc
```

After that you should build tf-idf and BM25 matrices for all documents. You can do it via command:
```sh
python -m search_engine.svd_tf_idf
python -m search_engine.svd_bm25
```

At last step you can build Vector database for all documents.
```sh
python -m search_engine.build_rag
```

**NOTE**: You need ollama installed and running to build vector database. You also need to have model "nomic-embed-text". You can install it via command:
```sh
ollama pull nomic-embed-text
```

## Frontend
To run the front end you need to set PYTHONPATH to the root of the project and run command:
```sh
streamlit run frontend/main.py
```

## Theory

This section covers quick overview of the theory behind the search engine.

### Theory of SVD

SVD stands for Singular Value Decomposition. It is a matrix factorization technique that decomposes a matrix into three matrices: $U$, $\Sigma$, and $V^T$. The original matrix can be reconstructed by multiplying these three matrices together: $A = U \Sigma V^T$. In the context of search engines, SVD is used to reduce the dimensionality of the term-document matrix, which can help to improve the performance of the search engine. By keeping only the top $k$ singular values and their corresponding singular vectors, we can create a lower-dimensional representation of the term-document matrix that captures the most important information while discarding noise and less important details.

### Theory of TF-IDF
TF-IDF stands for Term Frequency-Inverse Document Frequency.
For each term in the document we calculate two values:
IDF:
$$IDF(t) = log(\frac{N}{n_{t}})$$
where $N$ is the total number of documents and $n_{t}$ is the number of documents containing the term $t$. Later we transform matrix by multiplying each term vector by its IDF value. At last we normalize the matrix by dividing each term vector by its length. This way we get the TF-IDF matrix.

### Theory of BM25
BM25 is the next iteration of TF-IDF.
There are two main parameters in BM25:

- $k_1$ - term frequency saturation parameter, usually set to 1.2 or 2.0. It controls how much the term frequency contributes to the final score. Higher values of $k_1$ will give more weight to term frequency, while lower values will give less weight.

- $b$ - document length normalization parameter, usually set to 0.75. It controls how much the document length contributes to the final score. Higher values of $b$ will give more weight to document length, while lower values will give less weight.

First you calculate IDF but using different formula:
$$IDF(t) = log(\frac{N - n_{t} + 0.5}{n_{t} + 0.5} + 1.0)$$
You also calculate average document length:
$$AVG(d) = \frac{1}{N} \sum_{i=1}^{N} |d_{i}|$$
where $|d_{i}|$ is the length of the document $d_{i}$.

Later we similary to TF-IDF calculate norm factor:
$$norm(d) = (1 - b) + b \cdot \frac{|d|}{AVG(d)}$$

At last we calculate BM25 score for each term in the document:
$$BM25(t, d) = IDF(t) \cdot \frac{TF(t, d) \cdot (k_1 + 1)}{TF(t, d) + k_1 \cdot norm(d)}$$

### Theory of RAG
This technique is used to create vector database for all documents. It uses some AI model to create embeddings for each document. Effectively it is black box.

### Cosine similarity
Cosine similarity is a measure of similarity between two vectors. It is calculated as the cosine of the angle between the two vectors. The cosine similarity ranges from -1 to 1, where 1 means that the two vectors are identical, 0 means that the two vectors are orthogonal, and -1 means that the two vectors are opposite. In the context of search engines, cosine similarity is often used to measure the similarity between a query vector and a document vector. The higher the cosine similarity, the more relevant the document is to the query.

## Quick overview of the code structure

- `scrapper` - this module contains code for scrapping data from the Gutenberg project. It uses scrapy framework to do it.
- `search_engine` - this module contains code for data preprocessing and building the search engine. It contains code for building golden vocabulary, calculating TF-IDF and BM25 matrices, building vector database and searching for relevant documents.
- `utils` - this module contains BM25, SVD and TF-IDF implementations. They are represented as classes because before implementation of my own TF-IDF and SVD I used implementations from sklearn library. BM25 implementation in the same format was only natural option.
- `frontend` - this module contains code for the front end of the search engine. It uses streamlit framework to create a simple user interface for searching for relevant documents.
- `config.py` - this file contains all configuration needed to run the scrapper and search engine. It is used to avoid hardcoding values in the code and to make it easier to change configuration if needed.

## Quick overview of dependencies

- `scrapy` - framework for scrapping data from the web.
- `numpy` - library for numerical computing in Python.
- `scipy` - library for scientific computing in Python.
- `sklearn` - library for machine learning in Python. It is was used for implementation of TF-IDF and SVD before I implemented my own versions.
- `streamlit` - framework for creating web applications in Python. It is used for creating the front end of the search engine.
- `chromadb` - library for creating vector databases in Python. It is used for creating the vector database for all documents. It is already using **HNSW** graph for quick queryin of relevant documents.
- `peewee` - library for working with SQLite databases in Python. It is used for storing all documents in the database and for retrieving them when needed.
- `ollama` - library for working with Ollama models in Python. It is used for creating embeddings for all documents using the "nomic-embed-text" model.

# Enterprise-Search-Assistance-Retrival-Augmented-Generative-AI
## Abstract
Large Language Models (LLMs) represent a class of deep learning models pre-trained on vast datasets. These models, including popular examples like ChatGPT and Google Bard, utilize transformer neural networks to learn context by tracking relationships in sequential data, such as words in text. LLMs empower engineers to build solutions through semantic search capabilities, simplifying complex tasks like summarizing documents, answering questions, and generating new content—collectively known as Retrieval Augmented Generation Systems (RAG). This project combines retrieval capabilities and generative power to store, retrieve, and generate content for a mock technology company.
## Process
### Data Center
The goal of the RAG application is to simplify the retrieval of information from a data store. For this project, we created a mock company (Tech Innovators Inc.) using Atlassian’s Confluence as a virtual office for remote teams. The Confluence instance acts as a company wiki and was populated with fake company data to generate content for our application via API.
### Embeddings
Embeddings map high-dimensional data into lower-dimensional spaces, enabling easier machine learning applications. These embeddings are learned from data and capture semantic similarities. Key embedding techniques include:
- **Principal Component Analysis (PCA)**: Compresses variables but can lead to loss of information.
- **Singular Value Decomposition (SVD)**: Reduces data dimensionality but might omit important information.
- **Word2vec**: Creates word embeddings but struggles with different meanings for the same word.
- **Bidirectional Encoder Representations of Transformers (BERT)**: Solves Word2vec's context problem by training embeddings across a large dataset and fine-tuning based on specific use cases.
## Vectors and Indexing
Vectors, used to represent data in a machine-readable format, are critical in machine learning for tasks like similarity searches. In this project, we used a vector database to store and index embeddings for high-dimensional data. Here are some database options:
- **Pinecone**: Specializes in vector storage with fast and scalable similarity search.
- **Qdrant**: An open-source vector database that is flexible but can be challenging to implement.
- **MongoDB Atlas**: A universal database with vector storage and search capabilities, limited to 2048 dimensions. We opted to use MongoDB Atlas for its ease of use and advanced search capabilities.
## Querying
RAG systems enhance LLM performance by providing necessary context for queries. In our system, querying LLMs involves sending a prompt (e.g., a question or summary request) and receiving contextually relevant responses. More complex queries may involve chained prompts or reasoning loops to achieve optimal results. Proper "prompt engineering" ensures the query is informative enough to guide the LLM’s output.
LLMs, however, struggle with providing real-time information or answering very specific questions. RAG systems address this limitation by augmenting the query with additional context from external data sources.
## Implementing a Query Algorithm
Our RAG system uses the following techniques to optimize query handling:
- **Query Relaxation**: Drops query terms that yield no results to retrieve more general outputs.
- **Alternative Queries**: Replaces dropped terms with new query terms using fill-mask models like `distilbert-base-uncased`.
- **Query Intent Detection**: Identifies the user's underlying intent by classifying query tokens.
- **Query Expansion**: Expands short queries to improve the output quality by wrapping them in LLMs like ChatGPT.
## Evaluation
### LLM System Evaluation: Online and Offline
Evaluation is an ongoing process that ensures the LLM's performance meets the requirements of real-world applications. We used both online and offline evaluation techniques.
- **Offline Evaluation**: Tests LLMs against ground-truth datasets to verify performance standards before deployment. This method is ideal for entailment and factuality assessments.
- **Online Evaluation**: Uses real-world user data to assess live performance and satisfaction. This approach provides continuous feedback in a production environment.
### Evaluation Metrics
In evaluating our RAG system, we focused on key metrics such as precision, recall, and F1 score, tailoring these metrics to the system’s use case to provide a context-specific assessment.
## License
This project is licensed under the MIT License.

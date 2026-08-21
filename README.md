**Python Packages Required**
*1. Web Framework & API Layer**
fastapi version 0.115.0 - Framework used to build rest APIs.
uvicorn[standard] version 0.30.0 - ASGI webserver to run fastAPI.
pydantic version 2.7.0 - Data validation.

*2. RAG Search Pipeline**
llama-index version 0.11.0 - Core framwork for searching and managing context for LLMs.
llama-index-llms-openai version 0.2.0 - Connects LlamaIndex to OpenAI models.
llama-index-embeddings-huggingface version 0.3.0 - Runs local vector embedding models to convert text into searchable number.
llama-index-postprocessor-cohere-rerank version 0.2.0 - Uses Cohere to improve retrieval precision by reranking search reults.

*3. Machine Learning and Verification**
torch (PyTorch) version 2.2.0 - Deep learning engine required to run local AI models on CPU or GPU.
transformers version 4.40.0 - Hugging Face library used to load and run the DeBERTa model for Natural Language Inference (fact-checking claims against sources).
spacy version 3.7.0 - Natural Language Processing library used to split synthesized text into individual sentences for claim-by-claim verification.

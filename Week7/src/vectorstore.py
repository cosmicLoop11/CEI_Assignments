import cohere
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
import time

class VectorStore:
    def __init__(self, pdf_path: str, cohere_api_key: str, pinecone_api_key: str):
        self.pdf_path = pdf_path
        self.co = cohere.Client(api_key=cohere_api_key)
        self.pinecone_api_key = pinecone_api_key
        self.chunks = []
        self.embeddings = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        
        # Pipeline execution
        self.load_pdf()
        self.split_text()
        self.embed_chunks()
        self.index_chunks()

    def load_pdf(self):
        """Extracts text out of the provided PDF filepath."""
        text = ""
        with fitz.open(self.pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text("text") + "\n"
        self.pdf_text = text

    def split_text(self, chunk_size=800):
        """Chunks unstructured text into digestible segments based on sentence limits."""
        sentences = self.pdf_text.replace("\n", " ").split(". ")
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                self.chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            self.chunks.append(current_chunk.strip())

    def embed_chunks(self, batch_size=90):
        """Converts chunks into vector representations via Cohere V3 Embed."""
        total_chunks = len(self.chunks)
        if total_chunks == 0:
            raise ValueError("No text chunks found to embed. Please check your PDF input.")
            
        for i in range(0, total_chunks, batch_size):
            batch = self.chunks[i:min(i + batch_size, total_chunks)]
            batch_embeddings = self.co.embed(
                texts=batch, 
                input_type="search_document", 
                model="embed-english-v3.0"
            ).embeddings
            self.embeddings.extend(batch_embeddings)

    def index_chunks(self):
        """Initializes and upserts elements securely into a Pinecone Serverless Index."""
        pc = Pinecone(api_key=self.pinecone_api_key)
        index_name = 'rag-qa-bot'
        
        # Fixed bug: Safely retrieve the embedding size dimension
        embedding_dim = len(self.embeddings[0])

        # Safely create index if not existing
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            # Give the serverless index a brief moment to initialize spin up
            time.sleep(2)
            
        self.index = pc.Index(index_name)
        
        # Formulate and pack vectors for upserting
        chunks_metadata = [{'text': chunk} for chunk in self.chunks]
        ids = [str(i) for i in range(len(self.chunks))]
        
        vectors_to_upsert = list(zip(ids, self.embeddings, chunks_metadata))
        self.index.upsert(vectors=vectors_to_upsert)
        # Give Pinecone an absolute brief instant to reflect upserted elements
        time.sleep(1)

    def retrieve(self, query: str) -> list:
        """Queries vector database and filters through a smart semantic Cohere Reranker."""
        query_emb = self.co.embed(
            texts=[query], 
            model="embed-english-v3.0", 
            input_type="search_query"
        ).embeddings[0] # Fixed bug: extract list out of container matrix response
        
        res = self.index.query(vector=query_emb, top_k=self.retrieve_top_k, include_metadata=True)
        docs_to_rerank = [match['metadata']['text'] for match in res['matches'] if 'metadata' in match]
        
        if not docs_to_rerank:
            return []

        # Multi-stage Search: Rerank step
        rerank_results = self.co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0"
        )
        
        # Format explicitly into Cohere context structure requirements: {"text": "..."}
        final_docs = []
        for result in rerank_results.results:
            final_docs.append({"text": docs_to_rerank[result.index]})
            
        return final_docs
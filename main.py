from utils.pdf_loader import extract_text
from chunking.fixed_chunk import fixed_chunk
from embedding.openai_embed import get_embedding
from vectorstore.faiss_store import FAISSStore

# 1. Load PDF
text = extract_text(r"C:\MY_Folder\Github\RAG_DIfferent_Chunks_Usage\data\Sample_pdf.pdf")

# 2. Chunk
chunks = fixed_chunk(text, chunk_size=500, overlap=50)

print(f"Total chunks created: {len(chunks)}")

# 3. Embed chunks
embeddings = []
for chunk in chunks:
    embeddings.append(get_embedding(chunk))

# 4. Create FAISS store
dimension = len(embeddings[0])
store = FAISSStore(dimension)

store.add_embeddings(embeddings, chunks)

# 5. Query
query = "What is the main topic of this document?"
query_embedding = get_embedding(query)

results = store.search(query_embedding, top_k=3)

print("\nTop Retrieved Chunks:\n")
for r in results:
    print("-----")
    print(r[:500])

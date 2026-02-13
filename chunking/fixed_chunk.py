from langchain_text_splitters import CharacterTextSplitter




def fixed_chunk(text: str, chunk_size=500, overlap=50):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = splitter.split_text(text)
    return chunks

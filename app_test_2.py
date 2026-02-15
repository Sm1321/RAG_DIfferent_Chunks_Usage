import streamlit as st
import PyPDF2
from typing import List, Dict, Any
import numpy as np
from openai import AzureOpenAI
import re
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import pandas as pd
import time

# Initialize Azure OpenAI client
def init_azure_client() -> AzureOpenAI:
    """Initializes the Azure OpenAI client using credentials from Streamlit secrets."""
    try:
        return AzureOpenAI(
            api_key=st.secrets["AZURE_API_KEY"],
            api_version=st.secrets.get("API_VERSION", "2025-01-01-preview"),
            azure_endpoint=st.secrets["AZURE_ENDPOINT"]
        )
    except (KeyError, FileNotFoundError):
        st.error("🚨 Azure OpenAI credentials not found in st.secrets. Please configure your .streamlit/secrets.toml file.")
        st.stop()

# PDF Extraction
def extract_text_from_pdf(pdf_file) -> str:
    """Extracts all text from an uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return "".join(page.extract_text() for page in pdf_reader.pages)

def get_embeddings_batch(texts: List[str], client: AzureOpenAI) -> List[List[float]]:
    """Gets embeddings for a batch of texts to reduce API calls."""
    if not texts:
        return []
    try:
        response = client.embeddings.create(
            input=texts,
            model=st.secrets["EMBEDDING_MODEL"]
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Embedding batch error: {e}")
        return [[0] * 1536 for _ in texts]  # Fallback

# 1. Fixed-Size Chunking
def fixed_size_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Splits text into fixed-size chunks with overlap."""
    chunks, start, text_length = [], 0, len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

# 2. Content-Aware Chunking (Paragraph-based)
def content_aware_chunking(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Splits text by paragraphs, merging them up to a max chunk size."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def recursive_chunking(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Splits text recursively using a list of separators."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return [doc.page_content for doc in text_splitter.create_documents([text])]

def semantic_chunking(text: str, client: AzureOpenAI, threshold: float = 0.8, max_chunk_size: int = 1000) -> List[str]:
    """Splits text by semantic similarity of sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1: return [text]

    embeddings = get_embeddings_batch(sentences, client)
    
    chunks, current_chunk = [], sentences[0]
    for i in range(1, len(sentences)):
        similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if similarity > threshold and len(current_chunk) + len(sentences[i]) <= max_chunk_size:
            current_chunk += " " + sentences[i]
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentences[i]
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def agentic_chunking(text: str, client: AzureOpenAI, target_chunks: int = 5) -> List[str]:
    """Uses an LLM to intelligently split text into logical chunks."""
    truncated_text = text[:8000]
    prompt = f"""Analyze the following text and divide it into {target_chunks} logical, meaningful chunks.
    Each chunk should contain related information and maintain context.
    Return the chunks separated by a "---CHUNK---" marker.

    Text:
    {truncated_text}

    Provide only the chunked text with the separator, no explanations."""
    try:
        response = client.chat.completions.create(
            model=st.secrets["CHAT_MODEL"],
            messages=[
                {"role": "system", "content": "You are an expert at analyzing and structuring text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        result = response.choices[0].message.content
        return [chunk.strip() for chunk in result.split("---CHUNK---") if chunk.strip()]
    except Exception as e:
        st.error(f"Agentic chunking error: {e}")
        return [truncated_text]

# Query Answering with Retrieval
def get_embedding(text: str, client: AzureOpenAI, embedding_model: str) -> List[float]:
    """Get embedding for text"""
    try:
        response = client.embeddings.create(
            input=text[:8000],
            model=embedding_model
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return [0] * 1536

def answer_question(question: str, client: AzureOpenAI, chunks_with_embeddings: List[Dict[str, Any]], top_k: int = 5) -> Dict:
    """Answers a question using pre-computed embeddings for retrieval."""
    if not chunks_with_embeddings:
        return {
            "answer": "No chunks available.", 
            "relevant_chunks": [], 
            "scores": [],
            "retrieval_time": 0,
            "generation_time": 0,
            "total_time": 0
        }
    
    total_start = time.time()
    
    # Time the retrieval process
    retrieval_start = time.time()
    question_embedding = get_embeddings_batch([question], client)[0]
    chunk_embeddings = [item['embedding'] for item in chunks_with_embeddings]
    
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    relevant_chunks_data = [chunks_with_embeddings[i] for i in top_indices]
    context = "\n\n".join([
        f"[Relevance Score: {similarities[i]:.2f}]\n{item['text']}" 
        for i, item in zip(top_indices, relevant_chunks_data)
    ])
    retrieval_time = time.time() - retrieval_start
    
    # Time the answer generation
    generation_start = time.time()
    prompt = f"""Based on the following context, answer the question in detail.
    Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
    try:
        response = client.chat.completions.create(
            model=st.secrets["CHAT_MODEL"],
            messages=[{"role": "system", "content": "You are a helpful Q&A assistant."}, {"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content
        generation_time = time.time() - generation_start
        total_time = time.time() - total_start
        
        return {
            "answer": answer,
            "relevant_chunks": [item['text'] for item in relevant_chunks_data],
            "scores": [similarities[i] for i in top_indices],
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {e}", 
            "relevant_chunks": [], 
            "scores": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0,
            "total_time": time.time() - total_start
        }

def evaluate_answer_quality(question: str, answer: str, context: str, client: AzureOpenAI, 
                           full_document: str = None, reference_answer: str = None, 
                           evaluation_mode: str = "context_based") -> Dict[str, Any]:
    """Uses LLM to evaluate the quality of an answer based on multiple criteria.
    
    Args:
        evaluation_mode: 
            - "context_based": Evaluate only against retrieved context
            - "document_based": Evaluate against full document (ground truth)
            - "reference_based": Compare against a reference answer
    """
    
    if evaluation_mode == "document_based" and full_document:
        # More accurate: LLM has access to full document to verify facts
        evaluation_prompt = f"""You are an expert evaluator with access to the FULL DOCUMENT.

FULL DOCUMENT (Ground Truth):
{full_document[:15000]}

Question: {question}

Context Retrieved by System:
{context}

Generated Answer:
{answer}

First, determine what the CORRECT answer should be based on the full document.
Then evaluate the generated answer on these criteria (score 0-10):

1. **Factual Accuracy**: Is the answer factually correct compared to the full document?
2. **Completeness**: Does it cover all relevant information from the document?
3. **Relevance**: Does it directly address the question?
4. **Coherence**: Is it well-structured and clear?
5. **Hallucination Check**: Does it include information NOT in the document?

Provide evaluation in JSON format:
{{
    "factual_accuracy": <score 0-10>,
    "completeness": <score 0-10>,
    "relevance": <score 0-10>,
    "coherence": <score 0-10>,
    "hallucination_score": <score 0-10, 10=no hallucinations>,
    "overall_score": <average of all scores>,
    "correct_answer_summary": "<what the answer SHOULD have been based on full document>",
    "strengths": "<what was correct>",
    "weaknesses": "<what was wrong or missing>",
    "retrieval_quality": "<did the system retrieve the right chunks? score 0-10>"
}}

Return ONLY the JSON."""

    elif evaluation_mode == "reference_based" and reference_answer:
        # Compare against human-provided reference answer
        evaluation_prompt = f"""You are an expert evaluator comparing a generated answer against a reference answer.

Question: {question}

Reference Answer (Ground Truth):
{reference_answer}

Generated Answer:
{answer}

Context Used:
{context}

Evaluate how well the generated answer matches the reference answer (score 0-10):

1. **Semantic Similarity**: Does it convey the same meaning?
2. **Factual Alignment**: Does it contain the same facts?
3. **Completeness**: Does it cover everything in the reference?
4. **Accuracy**: Is everything stated correct?
5. **Quality**: Is it as clear and coherent as the reference?

Provide evaluation in JSON format:
{{
    "semantic_similarity": <score 0-10>,
    "factual_alignment": <score 0-10>,
    "completeness": <score 0-10>,
    "accuracy": <score 0-10>,
    "quality": <score 0-10>,
    "overall_score": <average of all scores>,
    "strengths": "<what matches well>",
    "weaknesses": "<what's missing or wrong>",
    "missing_information": "<key points from reference that are missing>"
}}

Return ONLY the JSON."""
    
    else:  # context_based (default, but less reliable)
        evaluation_prompt = f"""You are an expert evaluator assessing answer quality based on retrieved context.

⚠️ WARNING: You only see the RETRIEVED CONTEXT, not the full document. 
You can only verify if the answer is consistent with the given context.

Question: {question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate on these criteria (score 0-10):

1. **Context Consistency**: Is the answer consistent with the given context?
2. **Relevance**: Does it address the question?
3. **Coherence**: Is it well-structured?
4. **Context Coverage**: Does it use the context effectively?
5. **Confidence**: Based on context alone, how confident are you this is correct?

⚠️ Note: You CANNOT verify factual accuracy without the full document.

Provide evaluation in JSON format:
{{
    "context_consistency": <score 0-10>,
    "relevance": <score 0-10>,
    "coherence": <score 0-10>,
    "context_coverage": <score 0-10>,
    "confidence": <score 0-10>,
    "overall_score": <average of all scores>,
    "strengths": "<what seems good>",
    "weaknesses": "<potential issues>",
    "caveat": "Cannot verify factual accuracy without full document"
}}

Return ONLY the JSON."""

    try:
        response = client.chat.completions.create(
            model=st.secrets["CHAT_MODEL"],
            messages=[
                {"role": "system", "content": "You are an expert answer quality evaluator. Return only valid JSON."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0
        )
        
        result_text = response.choices[0].message.content.strip()
        # Extract JSON if wrapped in markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
            
        evaluation = json.loads(result_text)
        evaluation["evaluation_mode"] = evaluation_mode
        return evaluation
    except Exception as e:
        st.error(f"Evaluation error: {e}")
        return {
            "overall_score": 0,
            "strengths": "Error in evaluation",
            "weaknesses": str(e),
            "evaluation_mode": evaluation_mode
        }

# Streamlit App
def main():
    st.set_page_config(page_title="PDF Chunking Evaluator", layout="wide")
    st.title("📄 PDF Chunking Methods Evaluator")

    def clear_state():
        if 'chunking_results' in st.session_state:
            del st.session_state.chunking_results
            st.info("Parameters changed. Please re-generate chunks.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("Chunking Parameters")
        fixed_chunk_size = st.slider("Fixed Chunk Size", 200, 2000, 500, on_change=clear_state)
        overlap = st.slider("Overlap Size", 0, 500, 50, on_change=clear_state)
        semantic_threshold = st.slider("Semantic Threshold", 0.5, 0.95, 0.8, on_change=clear_state)

    client = init_azure_client()
    
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        text = st.session_state.get('extracted_text')
        if not text:
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                st.session_state.extracted_text = text
        st.success(f"✅ Extracted {len(text)} characters.")

        if st.button("Generate & Embed All Chunks", type="primary"):
            st.session_state.chunking_results = {}
            st.session_state.timing_results = {}
            
            with st.spinner("Running all chunking methods and generating embeddings..."):
                methods = {
                    "Fixed-Size": lambda: fixed_size_chunking(text, fixed_chunk_size, overlap),
                    "Content-Aware": lambda: content_aware_chunking(text),
                    "Recursive": lambda: recursive_chunking(text, fixed_chunk_size, overlap),
                    "Semantic": lambda: semantic_chunking(text, client, semantic_threshold),
                    "Agentic": lambda: agentic_chunking(text, client)
                }
                
                total_start_time = time.time()
                
                for name, chunk_func in methods.items():
                    method_start = time.time()
                    
                    # Time the chunking process
                    chunking_start = time.time()
                    chunks = chunk_func()
                    chunking_time = time.time() - chunking_start
                    
                    # Time the embedding process
                    embedding_start = time.time()
                    embeddings = get_embeddings_batch(chunks, client)
                    embedding_time = time.time() - embedding_start
                    
                    method_total_time = time.time() - method_start
                    
                    # Store chunks with embeddings
                    st.session_state.chunking_results[name] = [
                        {"text": chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)
                    ]
                    
                    # Store timing information
                    st.session_state.timing_results[name] = {
                        "chunking_time": chunking_time,
                        "embedding_time": embedding_time,
                        "total_time": method_total_time,
                        "num_chunks": len(chunks),
                        "avg_chunk_size": np.mean([len(chunk) for chunk in chunks]) if chunks else 0
                    }
                
                total_time = time.time() - total_start_time
                st.session_state.total_processing_time = total_time
                
            st.success(f"✅ All chunks generated and embedded in {total_time:.2f} seconds!")

        if 'chunking_results' in st.session_state:
            st.subheader("📊 Chunking Statistics")
            results = st.session_state.chunking_results
            timing_results = st.session_state.get('timing_results', {})
            
            # Display metrics with timing
            cols = st.columns(len(results))
            for idx, (method, data) in enumerate(results.items()):
                with cols[idx]:
                    st.metric(method, f"{len(data)} chunks")
                    avg_size = np.mean([len(item['text']) for item in data]) if data else 0
                    st.caption(f"Avg: {int(avg_size)} chars")
                    
                    # Display timing info
                    if method in timing_results:
                        timing = timing_results[method]
                        st.caption(f"⏱️ Chunking: {timing['chunking_time']:.2f}s")
                        st.caption(f"⏱️ Embedding: {timing['embedding_time']:.2f}s")
                        st.caption(f"⏱️ Total: {timing['total_time']:.2f}s")
            
            # Display detailed timing table
            if timing_results:
                st.markdown("### ⏱️ Detailed Timing Breakdown")
                timing_df_data = []
                for method, timing in timing_results.items():
                    timing_df_data.append({
                        "Method": method,
                        "Chunks": timing['num_chunks'],
                        "Avg Size (chars)": f"{int(timing['avg_chunk_size'])}",
                        "Chunking Time (s)": f"{timing['chunking_time']:.3f}",
                        "Embedding Time (s)": f"{timing['embedding_time']:.3f}",
                        "Total Time (s)": f"{timing['total_time']:.3f}",
                        "Time per Chunk (ms)": f"{(timing['total_time'] / timing['num_chunks'] * 1000):.1f}" if timing['num_chunks'] > 0 else "0"
                    })
                
                timing_df = pd.DataFrame(timing_df_data)
                st.dataframe(timing_df, use_container_width=True, hide_index=True)
                
                total_time = st.session_state.get('total_processing_time', 0)
                st.info(f"📊 Total Processing Time: {total_time:.2f} seconds")
            
            st.header("❓ Question Answering Evaluation")
            
            # Evaluation mode selection
            eval_mode = st.radio(
                "📋 Evaluation Mode:",
                ["context_based", "document_based", "reference_based"],
                format_func=lambda x: {
                    "context_based": "Context Only (Fast, Less Accurate) ⚠️",
                    "document_based": "Full Document (Slower, More Accurate) ✅",
                    "reference_based": "Reference Answer (Most Accurate) 🎯"
                }[x],
                help="""
                - **Context Only**: Evaluates based only on retrieved chunks (can't verify factual accuracy)
                - **Full Document**: LLM checks against entire document (can verify facts, but slower)
                - **Reference Answer**: Compare against your provided correct answer (most accurate)
                """
            )
            
            reference_answer = None
            if eval_mode == "reference_based":
                reference_answer = st.text_area(
                    "📝 Enter the reference (correct) answer:",
                    placeholder="Provide the ground truth answer to compare against...",
                    height=100
                )
            
            question = st.text_input("Enter your question about the document:")
            top_k = st.number_input("Top chunks to use", 1, 10, 3)

            if question and (eval_mode != "reference_based" or reference_answer):
                st.markdown("---")
                st.subheader("🎯 Evaluation Results")
                
                # Store evaluations for comparison
                evaluations = {}
                qa_timings = {}
                
                for method, chunks_with_embeddings in results.items():
                    st.markdown(f"### 📝 {method} Method")
                    
                    result = answer_question(question, client, chunks_with_embeddings, top_k)
                    
                    # Store QA timing
                    qa_timings[method] = {
                        "retrieval_time": result.get("retrieval_time", 0),
                        "generation_time": result.get("generation_time", 0),
                        "total_qa_time": result.get("total_time", 0)
                    }
                    
                    # Get context for evaluation
                    context = "\n\n".join(result['relevant_chunks'][:3])
                    
                    # Evaluate the answer based on selected mode
                    eval_start = time.time()
                    with st.spinner(f"Evaluating {method} answer..."):
                        evaluation = evaluate_answer_quality(
                            question=question,
                            answer=result["answer"],
                            context=context,
                            client=client,
                            full_document=text if eval_mode == "document_based" else None,
                            reference_answer=reference_answer if eval_mode == "reference_based" else None,
                            evaluation_mode=eval_mode
                        )
                        evaluations[method] = evaluation
                    eval_time = time.time() - eval_start
                    qa_timings[method]["evaluation_time"] = eval_time
                    qa_timings[method]["total_pipeline_time"] = result.get("total_time", 0) + eval_time
                    
                    # Display evaluation metrics
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### 💡 Generated Answer")
                        st.write(result["answer"])
                        
                        # Show additional evaluation insights (NOT nested in expander)
                        if eval_mode == "document_based" and "correct_answer_summary" in evaluation:
                            st.markdown("**🎯 What the answer SHOULD have been:**")
                            st.info(evaluation["correct_answer_summary"])
                            if "retrieval_quality" in evaluation:
                                st.metric("Retrieval Quality", f"{evaluation['retrieval_quality']:.1f}/10")
                        
                        if eval_mode == "reference_based" and "missing_information" in evaluation:
                            st.markdown("**❌ Missing Information:**")
                            st.warning(evaluation["missing_information"])
                    
                    with col2:
                        st.markdown("#### 📊 Quality Scores")
                        overall = evaluation.get('overall_score', 0)
                        
                        # Color-code the overall score
                        if overall >= 8:
                            score_color = "🟢"
                        elif overall >= 6:
                            score_color = "🟡"
                        else:
                            score_color = "🔴"
                        
                        st.metric("Overall Score", f"{score_color} {overall:.1f}/10")
                        
                        # Display timing metrics
                        if method in qa_timings:
                            timing = qa_timings[method]
                            st.markdown("**⏱️ Performance Metrics:**")
                            st.caption(f"Retrieval: {timing['retrieval_time']:.3f}s")
                            st.caption(f"Generation: {timing['generation_time']:.3f}s")
                            st.caption(f"Evaluation: {timing['evaluation_time']:.3f}s")
                            st.caption(f"**Total: {timing['total_pipeline_time']:.3f}s**")
                        
                        # Get metric keys dynamically based on evaluation mode
                        metric_keys = [k for k in evaluation.keys() if k.endswith('_score') == False and 
                                     isinstance(evaluation[k], (int, float)) and k != 'overall_score']
                        
                        # Display all available metrics
                        for metric_key in metric_keys:
                            score = evaluation[metric_key]
                            metric_name = metric_key.replace('_', ' ').title()
                            st.progress(score / 10, text=f"{metric_name}: {score:.1f}/10")
                        
                        # Show caveat for context-based mode
                        if eval_mode == "context_based":
                            st.caption("⚠️ " + evaluation.get("caveat", "Limited accuracy"))
                    
                    # Strengths and Weaknesses
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown("**✅ Strengths:**")
                        st.info(evaluation.get('strengths', 'N/A'))
                    with col4:
                        st.markdown("**⚠️ Weaknesses:**")
                        st.warning(evaluation.get('weaknesses', 'N/A'))
                    
                    # Show relevant chunks
                    with st.expander("🔍 View Retrieved Chunks"):
                        for i, (chunk, score) in enumerate(zip(result['relevant_chunks'], result['scores'])):
                            st.markdown(f"**Chunk {i+1}** (Similarity: {score:.2%})")
                            st.text_area(f"Content", chunk, height=150, key=f"{method}_{i}_content", label_visibility="collapsed")
                    
                    st.markdown("---")  # Separator between methods
                
                # Comparison table
                if evaluations:
                    st.markdown("---")
                    st.subheader("📈 Method Comparison")
                    
                    comparison_data = []
                    for method, eval_data in evaluations.items():
                        row = {"Method": method, "Overall": f"{eval_data.get('overall_score', 0):.1f}"}
                        
                        # Add all numeric scores dynamically
                        for key, value in eval_data.items():
                            if isinstance(value, (int, float)) and key != 'overall_score':
                                metric_name = key.replace('_', ' ').title()
                                row[metric_name] = f"{value:.1f}"
                        
                        comparison_data.append(row)
                    
                    df = pd.DataFrame(comparison_data)
                    
                    # Style the dataframe
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Find best method
                    best_method = max(evaluations.items(), key=lambda x: x[1].get('overall_score', 0))
                    st.success(f"🏆 **Best Performing Method:** {best_method[0]} with an overall score of {best_method[1].get('overall_score', 0):.1f}/10")
                    
                    # Evaluation mode indicator
                    mode_emoji = {"context_based": "⚠️", "document_based": "✅", "reference_based": "🎯"}
                    st.caption(f"{mode_emoji[eval_mode]} Evaluation Mode: {eval_mode.replace('_', ' ').title()}")
                    
                    # QA Performance Comparison Table
                    st.markdown("---")
                    st.subheader("⚡ Q&A Performance Comparison")
                    
                    qa_comparison_data = []
                    for method, timing in qa_timings.items():
                        chunking_timing = timing_results.get(method, {})
                        qa_comparison_data.append({
                            "Method": method,
                            "Chunking (s)": f"{chunking_timing.get('chunking_time', 0):.3f}",
                            "Embedding (s)": f"{chunking_timing.get('embedding_time', 0):.3f}",
                            "Retrieval (s)": f"{timing['retrieval_time']:.3f}",
                            "Generation (s)": f"{timing['generation_time']:.3f}",
                            "Evaluation (s)": f"{timing['evaluation_time']:.3f}",
                            "Total Q&A (s)": f"{timing['total_pipeline_time']:.3f}",
                            "End-to-End (s)": f"{chunking_timing.get('total_time', 0) + timing['total_pipeline_time']:.3f}"
                        })
                    
                    qa_df = pd.DataFrame(qa_comparison_data)
                    st.dataframe(qa_df, use_container_width=True, hide_index=True)
                    
                    # Find fastest method
                    fastest_method = min(qa_timings.items(), key=lambda x: x[1]['total_pipeline_time'])
                    st.info(f"⚡ **Fastest Q&A Method:** {fastest_method[0]} with {fastest_method[1]['total_pipeline_time']:.3f}s total time")

if __name__ == "__main__":
    main()
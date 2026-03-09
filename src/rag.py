"""
Interactive RAG Pipeline with Hallucination Detection - Streamlit Web UI
Upload PDF → Ask Questions → Get Answers with Hallucination Reports
"""

import re
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# --------------------------------------------------
# Document Processing Functions
# --------------------------------------------------
def load_document(doc_path):
    try:
        loader = PyPDFLoader(doc_path)
        pages = loader.load()
        return pages 
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return None

def clean_pages(pages):
    for p in pages:
        text = p.page_content
        
        # Remove PDF metadata
        text = re.sub(r'producer:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'creator:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'creationdate:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'ptex\.fullbanner:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'trapped:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'arxivid:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'doi:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'license:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'title:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'author:\s*.*?(?=\n\n|\Z)', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'arXiv:\d+\.\d+v\d+\s+\[.*?\]\s+\d+\s+\w+\s+\d{4}', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r"Metadata:\s*\{[^}]*\}", '', text)
        text = re.sub(r'-{40,}', '', text)
        text = re.sub(r'={40,}', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' \n', '\n', text)
        
        p.page_content = text.strip()
    
    pages = [p for p in pages if len(p.page_content) > 100]
    return pages

def clean_chunk_content(text):
    text = re.sub(r'CHUNK\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'chunk_id[:\s]+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'char_len[:\s]+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(page|page_label|source|total_pages)[:\s]+[^\s,}]+', '', text)
    text = re.sub(r'[-=]{4,}', '', text)
    text = re.sub(r'\[\d+\]\s*$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def chunk_text(pages, chunk_size=800, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(pages)
    
    for i, c in enumerate(chunks):
        c.page_content = clean_chunk_content(c.page_content)
        c.metadata["chunk_id"] = i
        c.metadata["char_len"] = len(c.page_content)
        c.metadata["word_count"] = len(c.page_content.split())
    
    chunks = [c for c in chunks if 
              len(c.page_content) > 50 and 
              not c.page_content.lower().startswith('metadata') and
              len(c.page_content.split()) > 5]
    
    return chunks

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore

def is_junk_chunk(text: str) -> bool:
    if len(text) < 100:
        return True
    if text.count("@") > 5:
        return True
    if text.count("http") > 5:
        return True
    return False

def retrieve_context(vectorstore, question, k=8, max_distance=2.0):
    results = vectorstore.similarity_search_with_score(question, k=k)
    
    filtered = []
    for doc, distance in results:
        if distance > max_distance:
            continue
        if is_junk_chunk(doc.page_content):
            continue
        filtered.append((doc, distance))
    
    filtered.sort(key=lambda x: x[1])
    selected = filtered[:4]
    
    return selected

def build_context(selected_chunks):
    context_blocks = []
    for i, (doc, distance) in enumerate(selected_chunks, 1):
        context_blocks.append(f"[DOC {i}]\n{doc.page_content.strip()}")
    return "\n\n".join(context_blocks)

def generate_answer(question, context):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    messages = [
        (
            "system",
            "You answer questions using the provided documents. "
            "If a statement is supported by a document, cite it using [DOC X]. "
            "If the documents do not clearly support a statement, "
            "still answer but mark it as [UNSUPPORTED]."
        ),
        (
            "user",
            f"""Documents:
{context}

Question:
{question}

Rules:
- Answer in plain language
- Every factual statement must end with either:
  • a citation like [DOC 1]
  • or the tag [UNSUPPORTED]
- Do not refuse to answer
- Do not explain your reasoning
- Do not summarize documents

Answer:"""
        )
    ]
    
    response = llm.invoke(messages)
    return response.content

def extract_claims(text):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Extract ONLY factual claims about the world or the document content from the text below.

Definition:
A valid claim is a declarative statement that asserts a fact about:
- entities, events, dates, locations, or properties described in the document

Do NOT extract:
- statements about missing information
- statements about lack of evidence
- statements explaining why the answer is unsupported
- meta statements like "the answer is unsupported" or "the documents do not provide information"

Instructions:
- Return each valid claim on a separate line
- Keep claims atomic
- If there are NO valid factual claims, return ONLY the word: NONE

Text:
\"\"\"{text}\"\"\"
"""
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    output = response.content.strip()
    
    if output == "NONE":
        return []
    
    claims = [c.strip() for c in output.split("\n") if c.strip()]
    return claims

def fact_check_claim(claim, context):
    verifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""You are a strict fact-checker.

Claim:
{claim}

Context:
{context}

Task:
Determine whether the claim is supported by the context.

Rules:
- Use ONLY the information explicitly stated or clearly implied in the context
- Do NOT use outside knowledge
- Do NOT assume missing facts
- If the context is related but does not directly support the claim, return 0.5
- If the context clearly supports the claim, return 1.0
- If the context does not support the claim, return 0.0

Return ONLY a single number: 1.0, 0.5, or 0.0
"""
    
    response = verifier_llm.invoke([{"role": "user", "content": prompt}])
    
    try:
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.0

def verify_claims(claims, vectorstore):
    results = []
    
    for claim in claims:
        search_results = vectorstore.similarity_search_with_score(claim, k=3)
        
        scores = []
        for doc, _ in search_results:
            score = fact_check_claim(claim, doc.page_content)
            scores.append(score)
        
        max_score = max(scores) if scores else 0.0
        
        if max_score >= 1.0:
            status = "SUPPORTED"
        elif max_score >= 0.5:
            status = "PARTIAL"
        else:
            status = "HALLUCINATED"
        
        results.append({
            "claim": claim,
            "score": max_score,
            "status": status
        })
    
    return results

def generate_hallucination_report(results):
    if not results:
        return {
            "total_claims": 0,
            "supported": 0,
            "partial": 0,
            "hallucinated": 0,
            "accuracy_score": 0.0,
            "reliability": "NO CLAIMS"
        }
    
    total = len(results)
    supported = sum(1 for r in results if r["status"] == "SUPPORTED")
    partial = sum(1 for r in results if r["status"] == "PARTIAL")
    hallucinated = sum(1 for r in results if r["status"] == "HALLUCINATED")
    
    avg_score = sum(r["score"] for r in results) / total
    
    if avg_score >= 0.9:
        reliability = "EXCELLENT"
    elif avg_score >= 0.7:
        reliability = "GOOD"
    elif avg_score >= 0.5:
        reliability = "FAIR"
    else:
        reliability = "POOR"
    
    return {
        "total_claims": total,
        "supported": supported,
        "partial": partial,
        "hallucinated": hallucinated,
        "accuracy_score": avg_score,
        "reliability": reliability,
        "details": results
    }

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="RAG with Hallucination Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 RAG System with Hallucination Detection")
    st.markdown("Upload a PDF and ask questions. Get answers with built-in fact-checking!")
    
    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📄 Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            if not st.session_state.pdf_processed or st.button("🔄 Reprocess PDF"):
                with st.spinner("Processing PDF..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process document
                    pages = load_document(tmp_path)
                    
                    if pages:
                        pages = clean_pages(pages)
                        st.success(f"✓ Loaded {len(pages)} pages")
                        
                        chunks = chunk_text(pages)
                        st.success(f"✓ Created {len(chunks)} chunks")
                        
                        st.session_state.vectorstore = create_vectorstore(chunks)
                        st.session_state.pdf_processed = True
                        st.success("✓ Vector store ready!")
                    else:
                        st.error("Failed to load PDF")
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
            
            if st.session_state.pdf_processed:
                st.success(f"📚 Loaded: {uploaded_file.name}")
                
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
    
    # Main chat interface
    if not st.session_state.pdf_processed:
        st.info("👈 Upload a PDF from the sidebar to get started!")
        return
    
    # Display chat history
    for item in st.session_state.chat_history:
        # User question
        with st.chat_message("user"):
            st.write(item["question"])
        
        # Assistant answer with hallucination report
        with st.chat_message("assistant"):
            st.markdown("### Answer")
            st.write(item["answer"])
            
            # Hallucination Report
            st.markdown("---")
            st.markdown("### 📊 Hallucination Report")
            
            report = item["report"]
            
            if report["total_claims"] == 0:
                st.info("No factual claims detected in the answer.")
            else:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Claims", report["total_claims"])
                
                with col2:
                    supported_pct = (report["supported"] / report["total_claims"]) * 100
                    st.metric("✓ Supported", f"{report['supported']} ({supported_pct:.0f}%)")
                
                with col3:
                    partial_pct = (report["partial"] / report["total_claims"]) * 100
                    st.metric("⚠ Partial", f"{report['partial']} ({partial_pct:.0f}%)")
                
                with col4:
                    hall_pct = (report["hallucinated"] / report["total_claims"]) * 100
                    st.metric("✗ Hallucinated", f"{report['hallucinated']} ({hall_pct:.0f}%)")
                
                # Accuracy score and reliability
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy Score", f"{report['accuracy_score']:.2f}/1.00")
                with col2:
                    reliability_color = {
                        "EXCELLENT": "🟢",
                        "GOOD": "🟡",
                        "FAIR": "🟠",
                        "POOR": "🔴"
                    }
                    st.metric("Reliability", f"{reliability_color.get(report['reliability'], '')} {report['reliability']}")
                
                # Detailed claims
                with st.expander("📋 View Detailed Claim Analysis"):
                    for i, detail in enumerate(report["details"], 1):
                        status_icon = {
                            "SUPPORTED": "✓",
                            "PARTIAL": "⚠",
                            "HALLUCINATED": "✗"
                        }[detail["status"]]
                        
                        st.markdown(f"**{i}. {detail['claim']}**")
                        st.markdown(f"{status_icon} {detail['status']} (confidence: {detail['score']:.2f})")
                        st.markdown("---")
    
    # Question input
    question = st.chat_input("Ask a question about the document...")
    
    if question:
        # Add user question to chat
        with st.chat_message("user"):
            st.write(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve context
                selected_chunks = retrieve_context(
                    st.session_state.vectorstore, 
                    question, 
                    k=8, 
                    max_distance=2.0
                )
                
                if not selected_chunks:
                    st.warning("Could not retrieve relevant context. Try rephrasing your question.")
                    return
                
                context = build_context(selected_chunks)
                answer = generate_answer(question, context)
                
                # Display answer
                st.markdown("### Answer")
                st.write(answer)
                
                # Extract and verify claims
                claims = extract_claims(answer)
                results = verify_claims(claims, st.session_state.vectorstore)
                report = generate_hallucination_report(results)
                
                # Display hallucination report
                st.markdown("---")
                st.markdown("### 📊 Hallucination Report")
                
                if report["total_claims"] == 0:
                    st.info("No factual claims detected in the answer.")
                else:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Claims", report["total_claims"])
                    
                    with col2:
                        supported_pct = (report["supported"] / report["total_claims"]) * 100
                        st.metric("✓ Supported", f"{report['supported']} ({supported_pct:.0f}%)")
                    
                    with col3:
                        partial_pct = (report["partial"] / report["total_claims"]) * 100
                        st.metric("⚠ Partial", f"{report['partial']} ({partial_pct:.0f}%)")
                    
                    with col4:
                        hall_pct = (report["hallucinated"] / report["total_claims"]) * 100
                        st.metric("✗ Hallucinated", f"{report['hallucinated']} ({hall_pct:.0f}%)")
                    
                    # Accuracy score and reliability
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy Score", f"{report['accuracy_score']:.2f}/1.00")
                    with col2:
                        reliability_color = {
                            "EXCELLENT": "🟢",
                            "GOOD": "🟡",
                            "FAIR": "🟠",
                            "POOR": "🔴"
                        }
                        st.metric("Reliability", f"{reliability_color.get(report['reliability'], '')} {report['reliability']}")
                    
                    # Detailed claims
                    with st.expander("📋 View Detailed Claim Analysis"):
                        for i, detail in enumerate(report["details"], 1):
                            status_icon = {
                                "SUPPORTED": "✓",
                                "PARTIAL": "⚠",
                                "HALLUCINATED": "✗"
                            }[detail["status"]]
                            
                            st.markdown(f"**{i}. {detail['claim']}**")
                            st.markdown(f"{status_icon} {detail['status']} (confidence: {detail['score']:.2f})")
                            st.markdown("---")
                
                # Save to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "report": report
                })

if __name__ == "__main__":
    main()
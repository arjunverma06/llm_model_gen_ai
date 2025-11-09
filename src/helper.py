import os
import re
from typing import List, Tuple

from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA

# ---------- PDF Loader (robust with fallback) ----------
# Prefer PyMuPDFLoader (better spacing). If not installed, fallback to PyPDFLoader.
try:
    from langchain_community.document_loaders import PyMuPDFLoader as PDFLoader
    _LOADER_NAME = "PyMuPDFLoader"
except Exception:
    from langchain_community.document_loaders import PyPDFLoader as PDFLoader
    _LOADER_NAME = "PyPDFLoader"

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from src.prompt import prompt_template, refine_template

# ---------- ENV ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment (.env). Please set GROQ_API_KEY=...")

# ---------- Helpers ----------
def clean_lines_to_questions(text: str) -> List[str]:
    """
    Return a clean list of questions from raw model output.
    - split by newline
    - trim
    - drop empty lines
    - remove leading numbering
    - remove MCQ choices (a), b), etc.) and commentary
    - de-duplicate while preserving order
    """
    qs: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # remove numbering like "1. ", "2) ", "- " etc.
        s = re.sub(r"^\s*(?:\d+[\.)-]\s*|-+\s*)", "", s)
        # drop MCQ choice lines
        if re.match(r"^[A-Da-d][\).\s-]\s*", s):
            continue
        # drop commentary lines
        lower = s.lower()
        if lower.startswith("that's correct") or lower.startswith("that's incorrect"):
            continue
        if lower.startswith("answer:") or lower.startswith("explanation:") or lower.startswith("i'm ready"):
            continue
        qs.append(s)

    seen = set()
    out: List[str] = []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def file_processing(file_path: str) -> Tuple[List[Document], List[Document]]:
    """Load PDF -> split -> return (for questions, for answers)."""
    loader = PDFLoader(file_path)
    pages = loader.load()

    # Concatenate all pages to one big string for question generation
    full_text = "\n\n".join(p.page_content for p in pages)

    # Chunking for question generation (bigger chunks)
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=300,
    )
    chunks_ques_gen = splitter_ques_gen.split_text(full_text)
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    # Chunking for answer retrieval (smaller chunks)
    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path: str, model_name: str = "llama-3.3-70b-versatile"):
    """
    Build the pipeline:
      - Generate questions with Groq model via refine summarize chain
      - Build FAISS index with HF embeddings
      - Create RetrievalQA chain for concise answers
    Returns: (answer_generation_chain, filtered_ques_list)
    """
    # 1) Load & split
    document_ques_gen, document_answer_gen = file_processing(file_path)

    # 2) Groq LLM for question generation
    llm_ques_gen = ChatGroq(
        temperature=0.3,
        model=model_name,
        # Groq clients now use max_completion_tokens (not max_tokens)
        max_completion_tokens=2048,
    )

    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template,
        input_variables=["text"],
    )
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        template=refine_template,
        input_variables=["existing_answer", "text"],
    )

    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen,
        chain_type="refine",
        verbose=False,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

    # IMPORTANT: use invoke() instead of deprecated run()
    res = ques_gen_chain.invoke({"input_documents": document_ques_gen})
    raw_questions = res["output_text"] if isinstance(res, dict) else str(res)
    filtered_ques_list = clean_lines_to_questions(raw_questions)

    # 3) Embeddings + VectorStore (HF + FAISS)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # 4) Groq LLM for answer generation
    llm_answer_gen = ChatGroq(
        temperature=0.15,
        model=model_name,
        max_completion_tokens=512,
    )

    # ✅ FIX: StuffDocumentsChain expects {context}; tell chain to use that variable name
    qa_prompt = PromptTemplate.from_template(
        """You are answering strictly from the provided context.
Return a concise answer (2–4 sentences), no bullet points, no preambles.

Context:
{context}

Question:
{question}
"""
    )

    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        verbose=False,
        chain_type_kwargs={
            "prompt": qa_prompt,
            "document_variable_name": "context",  # <-- IMPORTANT: matches {context} above
        },
    )

    return answer_generation_chain, filtered_ques_list

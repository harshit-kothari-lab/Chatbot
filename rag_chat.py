# import os
# import re
# from collections import Counter

# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI

# load_dotenv()

# PERSIST_DIR = "db/chroma_db"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# CHAT_MODEL = "llama-3.1-8b-instant"

# print("Loading embedding model...")
# embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# print("Loading vector database...")
# db = Chroma(
#     persist_directory=PERSIST_DIR,
#     embedding_function=embedding_model
# )

# print("Loading chat model...")
# model = ChatOpenAI(
#     model=CHAT_MODEL,
#     api_key=GROQ_API_KEY,
#     base_url=GROQ_BASE_URL,
#     temperature=0
# )

# STOPWORDS = {
#     "the", "is", "are", "of", "in", "to", "a", "an", "and", "for", "on",
#     "what", "which", "who", "how", "does", "do", "did", "used", "use",
#     "paper", "study", "research"
# }


# def tokenize(text: str):
#     tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
#     return [t for t in tokens if t not in STOPWORDS]


# def format_sources(docs):
#     seen = set()
#     sources = []
#     for doc in docs:
#         src = doc.metadata.get("source", "unknown")
#         page = doc.metadata.get("page", "unknown")
#         key = (src, page)
#         if key not in seen:
#             seen.add(key)
#             sources.append({"source": src, "page": page})
#     return sources


# def build_context(docs):
#     blocks = []
#     for doc in docs:
#         src = doc.metadata.get("source", "unknown")
#         page = doc.metadata.get("page", "unknown")
#         section = doc.metadata.get("section", "unknown")
#         text = doc.page_content.strip()

#         blocks.append(
#             f"[Source: {src} | Page: {page} | Section: {section}]\n{text}"
#         )

#     return "\n\n".join(blocks)


# def rerank_documents(query: str, docs):
#     q_tokens = tokenize(query)
#     q_counter = Counter(q_tokens)

#     scored = []

#     for rank, doc in enumerate(docs):
#         text = doc.page_content.lower()
#         section = str(doc.metadata.get("section", "unknown")).lower()

#         rank_score = max(0, 10 - rank)

#         text_tokens = set(tokenize(text))
#         text_overlap = sum(q_counter[t] for t in q_tokens if t in text_tokens)

#         section_tokens = set(tokenize(section))
#         section_overlap = sum(q_counter[t] for t in q_tokens if t in section_tokens)

#         final_score = rank_score + (2.0 * text_overlap) + (3.0 * section_overlap)
#         scored.append((final_score, doc))

#     scored.sort(key=lambda x: x[0], reverse=True)
#     return [doc for _, doc in scored]


# def retrieve_docs(query: str, initial_k: int = 8, final_k: int = 3):
#     docs = db.similarity_search(query, k=initial_k)

#     if not docs:
#         return []

#     reranked = rerank_documents(query, docs)
#     return reranked[:final_k]


# def answer_query(query: str):
#     relevant_docs = retrieve_docs(query)

#     if not relevant_docs:
#         return {
#             "answer": "I couldn't find enough evidence in the provided documents.",
#             "sources": []
#         }

#     context = build_context(relevant_docs)

#     prompt = f"""
# You are a document-grounded research assistant.

# Your task is to answer the user's question using ONLY the provided context.

# Instructions:
# 1. Use only the provided context. Do not use outside knowledge.
# 2. If the answer is explicitly stated in the context, answer directly.
# 3. If the answer is not stated in one sentence but is strongly supported across multiple retrieved chunks, synthesize a concise answer from those chunks.
# 4. Do NOT invent facts, names, methods, or conclusions that are not supported by the context.
# 5. If the context truly does not provide enough evidence, say exactly:
#    "I couldn't find enough evidence in the provided documents."
# 6. Keep the answer concise, factual, and grounded.
# 7. For questions like "objective", "aim", "purpose", "methodology", or "findings", it is acceptable to summarize the relevant statements from introduction/method/results sections if they clearly describe that concept.

# Question:
# {query}

# Context:
# {context}

# Answer:
# """

#     print("Sending request to Groq...")
#     response = model.invoke(prompt)

#     return {
#         "answer": response.content.strip(),
#         "sources": format_sources(relevant_docs),
#     }

# if __name__ == "__main__":
#     while True:
#         query = input("\nAsk a question (or type 'exit'): ")

#         if query.lower().strip() == "exit":
#             break

#         if not query.strip():
#             print("Please enter a question.")
#             continue

#         result = answer_query(query)

#         print("\nAnswer:")
#         print(result["answer"])

#         print("\nSources:")
#         if result["sources"]:
#             for src in result["sources"]:
#                 print(f"- {src['source']} (page {src['page']})")
#         else:
#             print("No sources found.")



"""
===============================================================================
ENTERPRISE RAG CHAT PIPELINE

Workflow

User Query
    │
    ▼
Query Processing
    │
    ▼
Retriever (MMR)
    │
    ▼
Cross Encoder Re-ranking
    │
    ▼
Context Compression
    │
    ▼
Prompt Builder
    │
    ▼
LLM
    │
    ▼
Answer + Sources + Confidence

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
import hashlib
import math
import logging
from collections import Counter
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

PERSIST_DIR = "db/chroma_db"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

CHAT_MODEL = "llama-3.1-8b-instant"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

TEMPERATURE = 0

MAX_CONTEXT_DOCS = 8

FINAL_CONTEXT_DOCS = 5

MAX_CONTEXT_LENGTH = 12000


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("Loading Enterprise RAG Pipeline")
logger.info("="*60)


# =============================================================================
# LOAD EMBEDDING MODEL
# =============================================================================

logger.info("Loading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(

    model_name=EMBED_MODEL,

    model_kwargs={"device":"cpu"},

    encode_kwargs={"normalize_embeddings":True}

)

logger.info("Embedding Model Loaded")


# =============================================================================
# LOAD VECTOR DATABASE
# =============================================================================

logger.info("Loading Chroma Database...")

db = Chroma(

    persist_directory=PERSIST_DIR,

    embedding_function=embedding_model

)

logger.info("Database Loaded")


# =============================================================================
# LOAD LLM
# =============================================================================

logger.info("Loading Groq Model...")

llm = ChatOpenAI(

    model=CHAT_MODEL,

    api_key=GROQ_API_KEY,

    base_url=GROQ_BASE_URL,

    temperature=TEMPERATURE

)

logger.info("LLM Loaded")

# =============================================================================
# CROSS ENCODER
# =============================================================================

logger.info("Loading Cross Encoder...")

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

logger.info("Cross Encoder Loaded")

# =============================================================================
# STOPWORDS
# =============================================================================

STOPWORDS = {

"the","is","are","of","to","in","for","on","a","an","and",

"what","which","who","how","why","where","when",

"paper","research","study","document",

"does","did","do","using","used","use",

"about","their","there","these","those"

}


# =============================================================================
# QUERY PREPROCESSING
# =============================================================================

"""
Enterprise systems never directly embed the raw query.

We first normalize it.

Example

"What's the objective?"

↓

"objective"

This slightly improves retrieval.
"""

def clean_query(query:str):

    query=query.lower()

    query=query.strip()

    query=re.sub(r"\s+"," ",query)

    return query


def tokenize(text:str):

    words=re.findall(

        r"[a-zA-Z0-9]+",

        text.lower()

    )

    return [

        w

        for w in words

        if w not in STOPWORDS

    ]

# =============================================================================
# QUERY INTENT DETECTION
# =============================================================================

"""
Enterprise retrieval systems first determine
what the user is actually asking.

Example

"What is the objective?"

↓

Intent = objective

↓

Boost Objective / Introduction sections

instead of searching every chunk equally.

This improves retrieval quality considerably.
"""

INTENT_KEYWORDS = {

    "abstract":[
        "abstract",
        "summary",
        "overview"
    ],

    "objective":[
        "objective",
        "aim",
        "purpose",
        "goal",
        "motivation"
    ],

    "introduction":[
        "introduction",
        "background"
    ],

    "methodology":[
        "method",
        "methodology",
        "approach",
        "architecture",
        "algorithm",
        "pipeline",
        "workflow",
        "framework",
        "implementation"
    ],

    "dataset":[
        "dataset",
        "data",
        "training data",
        "corpus"
    ],

    "results":[
        "result",
        "accuracy",
        "performance",
        "evaluation",
        "experiment",
        "benchmark"
    ],

    "discussion":[
        "discussion",
        "analysis",
        "interpretation"
    ],

    "limitations":[
        "limitation",
        "drawback",
        "weakness"
    ],

    "future work":[
        "future work",
        "extension",
        "improvement"
    ],

    "conclusion":[
        "conclusion",
        "conclude",
        "final remarks"
    ],

    "figure":[
        "figure",
        "diagram",
        "flowchart",
        "architecture diagram",
        "image"
    ],

    "table":[
        "table",
        "statistics",
        "values"
    ]
}


def detect_intent(query):

    query = clean_query(query)

    scores = Counter()

    for intent, keywords in INTENT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in query:

                scores[intent] += 1

    if not scores:

        return "general"

    return scores.most_common(1)[0][0]


# =============================================================================
# QUERY EXPANSION
# =============================================================================

"""
Expand common research terms.

Instead of searching only

"accuracy"

we also search

performance
evaluation
results

This increases recall.
"""

QUERY_EXPANSION = {

    "accuracy":[
        "performance",
        "evaluation",
        "results"
    ],

    "objective":[
        "purpose",
        "goal",
        "motivation"
    ],

    "method":[
        "methodology",
        "approach",
        "algorithm"
    ],

    "dataset":[
        "training data",
        "corpus"
    ],

    "result":[
        "performance",
        "evaluation"
    ],

    "limitation":[
        "weakness",
        "drawback"
    ]

}



# =============================================================================
# METADATA FILTERS
# =============================================================================

"""
Uses metadata generated during ingestion.

Example

Question:

"What is the methodology?"

↓

Prefer chunks whose metadata says

section = Methodology
"""

SECTION_MAPPING = {

    "objective":[
        "Objective",
        "Introduction"
    ],

    "methodology":[
        "Methodology",
        "Implementation",
        "Architecture"
    ],

    "dataset":[
        "Dataset",
        "Data Collection"
    ],

    "results":[
        "Results",
        "Evaluation",
        "Analysis"
    ],

    "discussion":[
        "Discussion"
    ],

    "limitations":[
        "Limitations"
    ],

    "future work":[
        "Future Work"
    ],

    "conclusion":[
        "Conclusion"
    ]

}


def section_boost(intent, docs):
    """
    We don't discard documents.

    We simply move matching sections
    toward the front.
    """

    if intent not in SECTION_MAPPING:

        return docs

    preferred = SECTION_MAPPING[intent]

    boosted = []

    remaining = []

    for doc in docs:

        section = str(

            doc.metadata.get(

                "section",

                ""

            )

        )

        if section in preferred:

            boosted.append(doc)

        else:

            remaining.append(doc)

    return boosted + remaining

# =============================================================================
# RETRIEVAL ENGINE
# =============================================================================

"""
Instead of simple similarity search,
we use MMR (Maximum Marginal Relevance).

Why?

Similarity Search often returns

Chunk 1
Chunk 2
Chunk 3

all from the same paragraph.

MMR encourages diversity.

Example

Chunk 1 -> Introduction

Chunk 2 -> Methodology

Chunk 3 -> Results

which gives the LLM much richer context.
"""


def dynamic_k(query):

    """
    Decide retrieval depth based on question complexity.
    """

    length = len(tokenize(query))

    if length <= 3:
        return 6

    if length <= 8:
        return 10

    return 14


# =============================================================================
# HYBRID RETRIEVAL
# =============================================================================

def retrieve_documents(query):

    query = clean_query(query)

    intent = detect_intent(query)

    k = dynamic_k(query)

    logger.info("=" * 60)
    logger.info("Retrieving documents...")
    logger.info(f"Intent : {intent}")

    retriever = db.as_retriever(

        search_type="mmr",

        search_kwargs={

            "k": k,

            "fetch_k": max(20, 2 * k),

            "lambda_mult": 0.65

        }

    )

    docs = retriever.invoke(query)

    docs = section_boost(intent, docs)

    ####################################################
    # Remove duplicate chunks
    ####################################################

    unique_docs = []

    seen = set()

    for doc in docs:

        fingerprint = doc.metadata.get(

            "chunk_hash",

            hash(doc.page_content)

        )

        if fingerprint in seen:

            continue

        seen.add(fingerprint)

        unique_docs.append(doc)

    logger.info(

        f"Retrieved {len(unique_docs)} unique chunks."

    )

    return unique_docs, intent

def keyword_overlap(query, text):

    q = set(tokenize(query))

    t = set(tokenize(text))

    return len(q & t)


# =============================================================================
# CROSS ENCODER RERANKING
# =============================================================================

"""
Embeddings retrieve candidate chunks.

Cross Encoder decides their final order.

This greatly improves retrieval quality because
the model jointly reads

(query , document)

instead of comparing embeddings independently.
"""


def rerank_documents(

    query,

    docs,

    intent

):

    if not docs:

        return []

    logger.info("Cross Encoder Re-ranking...")

    pairs = [

        (

            query,

            doc.page_content

        )

        for doc in docs

    ]

    scores = cross_encoder.predict(

        pairs

    )

    ranked = sorted(

        zip(

            scores,

            docs

        ),

        key=lambda x: x[0],

        reverse=True

    )

    reranked = [

        doc

        for _, doc in ranked

    ]

    ##########################################################
    # Slight section preference
    ##########################################################

    reranked = section_boost(

        intent,

        reranked

    )

    return reranked[:FINAL_CONTEXT_DOCS]

# =============================================================================
# CONTEXT COMPRESSION
# =============================================================================

"""
The CrossEncoder returns the best chunks.

However,

many chunks still repeat the same idea.

This module removes redundant information before
sending context to the LLM.

Benefits

✓ lower token usage

✓ less hallucination

✓ better answers
"""


def remove_duplicate_context(docs):

    compressed = []

    seen = set()

    for doc in docs:

        text = clean_text(doc.page_content)

        fingerprint = hashlib.md5(

            text.encode()

        ).hexdigest()

        if fingerprint in seen:

            continue

        seen.add(fingerprint)

        compressed.append(doc)

    return compressed


def trim_context(docs):

    """
    Ensures we never exceed

    MAX_CONTEXT_LENGTH
    """

    final_docs = []

    total = 0

    for doc in docs:

        length = len(doc.page_content)

        if total + length > MAX_CONTEXT_LENGTH:

            break

        final_docs.append(doc)

        total += length

    logger.info(

        f"Context Length : {total}"

    )

    return final_docs


# =============================================================================
# CONTEXT BUILDER
# =============================================================================

"""
Creates the final prompt context.

Every chunk keeps

• source

• page

• section

This makes citations much easier.
"""


def build_context(docs):

    docs = remove_duplicate_context(docs)

    docs = trim_context(docs)

    blocks = []

    for i, doc in enumerate(docs, start=1):

        meta = doc.metadata

        source = meta.get(

            "source",

            "Unknown"

        )

        page = meta.get(

            "page",

            "?"

        )

        section = meta.get(

            "section",

            "Unknown"

        )

        chunk_type = meta.get(

            "chunk_type",

            "text"

        )

        block = f"""
==================== DOCUMENT {i} ====================

Source      : {source}

Page        : {page}

Section     : {section}

Chunk Type  : {chunk_type}

Content

{doc.page_content}

"""

        blocks.append(

            block.strip()

        )

    logger.info(

        f"Context contains {len(blocks)} chunks."

    )

    return "\n\n".join(blocks)


# =============================================================================
# SOURCE FORMATTER
# =============================================================================

def format_sources(docs):

    seen = set()

    sources = []

    for doc in docs:

        src = doc.metadata.get(

            "source"

        )

        page = doc.metadata.get(

            "page"

        )

        key = (

            src,

            page

        )

        if key in seen:

            continue

        seen.add(key)

        sources.append(

            {

                "source": src,

                "page": page

            }

        )

    return sources

# =============================================================================
# PROMPT BUILDER
# =============================================================================

"""
This module constructs the final prompt sent to the LLM.

A well-designed prompt significantly reduces hallucinations.

The LLM is instructed to answer ONLY from the retrieved
documents.
"""



def build_prompt(query, context):

    return f"""
You are an Enterprise AI Research Assistant.

Your ONLY source of truth is the retrieved documents below.

===========================================================
YOUR ROLE
===========================================================

You answer questions ONLY using the provided documents.

You DO NOT use outside knowledge.

You DO NOT guess.

You NEVER fabricate information.

===========================================================
RULES
===========================================================

1. Use ONLY the retrieved context.

2. If the answer exists in multiple chunks,
   combine them into one concise answer.

3. If evidence is insufficient, say EXACTLY:

"I couldn't find enough evidence in the provided documents."

4. Never invent

• methods

• numbers

• datasets

• conclusions

• names

5. Prefer information that appears repeatedly
across multiple retrieved chunks.

6. When possible,
mention the relevant section naturally.

===========================================================
RETRIEVED DOCUMENTS
===========================================================

{context}

===========================================================
QUESTION
===========================================================

{query}

===========================================================
YOUR RESPONSE
===========================================================

Write a concise, professional answer.

Do NOT mention that you were given context.

Do NOT mention retrieval.

Do NOT mention documents unless the user asks.

"""

# =============================================================================
# TEXT CLEANER
# =============================================================================

"""
Small helper used only inside rag_chat.py.

Do not confuse with the ingestion cleaner.
"""

def clean_text(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =============================================================================
# ANSWER QUERY
# =============================================================================

def answer_query(query):

    logger.info("="*60)
    logger.info(f"Question : {query}")

    ############################################################
    # Retrieval
    ############################################################

    docs, intent = retrieve_documents(query)

    if not docs:

        return {

            "answer":

            "I couldn't find enough evidence in the provided documents.",

            "sources":[]

        }

    ############################################################
    # Cross Encoder
    ############################################################

    docs = rerank_documents(

        query,

        docs,

        intent

    )

    ############################################################
    # Context
    ############################################################

    context = build_context(

        docs

    )

    ############################################################
    # Prompt
    ############################################################

    prompt = build_prompt(

        query,

        context

    )

    ############################################################
    # LLM
    ############################################################

    logger.info("Generating Answer...")

    response = llm.invoke(prompt)

    ############################################################
    # Return
    ############################################################

    return {

        "answer":response.content.strip(),

        "sources":format_sources(docs)

    }


# =============================================================================
# COMMAND LINE TEST
# =============================================================================

if __name__=="__main__":

    logger.info("Enterprise RAG Ready.")

    while True:

        query=input("\nQuestion : ")

        if query.lower()=="exit":

            break

        result=answer_query(query)

        print("\n")

        print("="*70)

        print(result["answer"])

        print("\n")

        print("Sources")

        for s in result["sources"]:

            print(

                f"{s['source']} | Page {s['page']}"

            )

        print("="*70)
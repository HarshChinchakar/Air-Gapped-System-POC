#!/usr/bin/env python3
"""
Streamlit RAG Interface (Tables / Text + Tables selector)
Final Streamlit-ready version — Creator: Harsh Chinchakar
"""

import os, sys, json, re, subprocess, traceback
from pathlib import Path
from datetime import datetime
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv

# ---------------- ENV + PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_RETRIEVAL_SCRIPT = str(BASE_DIR / "Retrival" / "retrieval_combined_v2.py")
TABLES_ONLY_SCRIPT = str(BASE_DIR / "Retrival" / "retrival_tables.py")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"
SEMANTIC_TOP = 8
KEYWORD_TOP = 8
CHUNK_CHAR_LIMIT = 2500
TIMEOUT_SECONDS = 120

# ---------------- HELPERS ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def run_retrieval(query: str, retrieval_script: str):
    """Executes retrieval script via subprocess (same env as Streamlit)."""
    try:
        if not os.path.exists(retrieval_script):
            raise FileNotFoundError(f"Retrieval script not found: {retrieval_script}")

        cmd = [
            sys.executable,  # use same python env
            retrieval_script,
            "--query", query,
            "--top_k", str(max(SEMANTIC_TOP, KEYWORD_TOP))
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(BASE_DIR)
        )

        st.text_area("📜 Retrieval STDOUT", proc.stdout or "No output", height=200)
        if proc.stderr.strip():
            st.text_area("⚠️ Retrieval STDERR", proc.stderr, height=200)

        if proc.returncode != 0:
            raise RuntimeError(f"Retrieval failed (exit {proc.returncode})")

        match = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", proc.stdout)
        if not match:
            raise RuntimeError("No valid JSON found in retrieval output.")

        return json.loads(match.group(1))
    except Exception as e:
        st.error(f"❌ Retrieval error: {e}")
        st.text(traceback.format_exc())
        raise

def is_table_chunk(item: Dict[str, Any]) -> bool:
    stype = (item.get("section_type") or "").lower()
    if "table" in stype or "table_summary" in stype:
        return True
    if item.get("table_id") or item.get("table_part_index"):
        return True
    pdf = (item.get("pdf_name") or "").lower()
    if pdf.endswith((".xlsx", ".xls")):
        return True
    content = (item.get("content") or "")[:500]
    if "\t" in content or " | " in content or content.strip().startswith("|"):
        return True
    return False

def dedupe_and_merge(semantic, keyword):
    combined, seen = [], set()
    for src in (semantic or [])[:SEMANTIC_TOP] + (keyword or [])[:KEYWORD_TOP]:
        cid = src.get("chunk_id") or f"{src.get('pdf_name')}::p{src.get('page')}"
        if cid in seen:
            continue
        seen.add(cid)
        content = (src.get("content") or "")[:CHUNK_CHAR_LIMIT]
        combined.append({
            "chunk_id": cid,
            "pdf_name": src.get("pdf_name") or "<unknown>",
            "page": src.get("page"),
            "section_type": src.get("section_type"),
            "score": src.get("score", None),
            "content": content
        })
    return combined

# ---------------- RAG PIPELINE ----------------
def run_rag_pipeline(query, scope="tables"):
    """Runs retrieval and displays retrieved chunks for debug."""
    if scope == "tables":
        retrieval_script = TABLES_ONLY_SCRIPT
        apply_filter = False
    else:
        retrieval_script = DEFAULT_RETRIEVAL_SCRIPT
        apply_filter = True

    retrieval = run_retrieval(query, retrieval_script)

    sem_results = retrieval.get("semantic", {}).get("results", [])
    kw_results = retrieval.get("keyword", {}).get("results", [])

    def filter_scope(items):
        if not apply_filter:
            return items
        if scope == "text":
            return [i for i in items if not is_table_chunk(i)]
        elif scope == "tables":
            return [i for i in items if is_table_chunk(i)]
        return items

    sem_filtered, kw_filtered = filter_scope(sem_results), filter_scope(kw_results)
    merged_chunks = dedupe_and_merge(sem_filtered, kw_filtered)

    # Display retrieved chunks directly
    st.markdown("### 📚 Retrieved Chunks")
    if not merged_chunks:
        st.warning("⚠️ No chunks retrieved.")
    else:
        for c in merged_chunks[:10]:  # show top 10
            st.markdown(
                f"**Chunk ID:** `{c['chunk_id']}`  \n"
                f"**PDF:** `{c['pdf_name']}`  \n"
                f"**Page:** {c.get('page', '?')}  \n"
                f"**Content:** {c['content'][:500]}..."
            )
            st.markdown("---")

    return merged_chunks

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

st.title("🧠 RAG Assistant — Retrieval Debug Mode")
st.markdown("Enter a query below and view retrieved chunks directly from the retrieval pipeline.")

query = st.text_area("🔍 Enter Query:", placeholder="e.g., Summarize company performance for FY21")
scope_option = st.selectbox("Retrieval Scope", ["Text + Tables", "Tables Only"], index=0)
run_button = st.button("Run Retrieval")

if run_button and query.strip():
    with st.spinner("Running retrieval..."):
        scope = "tables" if scope_option == "Tables Only" else "all"
        try:
            chunks = run_rag_pipeline(query.strip(), scope=scope)
            if chunks:
                st.success(f"✅ Retrieved {len(chunks)} chunks successfully.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.text(traceback.format_exc())

elif run_button:
    st.warning("Please enter a query first.")

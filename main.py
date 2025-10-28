#!/usr/bin/env python3
"""
Streamlit RAG Interface (Tables / Text + Tables selector)
Creator: Harsh Chinchakar
Robust Hosted Version — Handles missing paths & silent subprocess failures gracefully.
"""

import os, sys, json, re, shlex, subprocess, traceback
from pathlib import Path
from datetime import datetime
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv

# ---------------- Load ENV ----------------
try:
    load_dotenv("./.env")
except Exception as e:
    st.error(f"⚠️ Failed to load .env: {e}")

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RETRIEVAL_SCRIPT = str(BASE_DIR / "Retrival" / "retrieval_combined_v2.py")
TABLES_ONLY_SCRIPT = str(BASE_DIR / "Retrival" / "retrival_tables.py")

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
SEMANTIC_TOP = 12
KEYWORD_TOP = 12
CHUNK_CHAR_LIMIT = 2500
TIMEOUT_SECONDS = 120

# ---------------- Helper ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def run_retrieval(query: str, retrieval_script: str):
    """Execute retrieval script and return parsed JSON output, with real-time debug."""
    try:
        if not os.path.exists(retrieval_script):
            raise FileNotFoundError(f"Retrieval script missing: {retrieval_script}")

        st.info(f"🔍 Running retrieval from: `{retrieval_script}`")

        cmd = [
            "python3",
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

        # Show debug info to Streamlit always
        st.text_area("📜 Retrieval Output (stdout)", proc.stdout or "No STDOUT output", height=200)
        if proc.stderr.strip():
            st.text_area("⚠️ Retrieval Errors (stderr)", proc.stderr, height=200)

        if proc.returncode != 0:
            raise RuntimeError(f"Retrieval failed (code {proc.returncode}).")

        match = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", proc.stdout)
        if not match:
            raise RuntimeError("No valid JSON found in retrieval script output.")

        return json.loads(match.group(1))

    except Exception as e:
        st.error(f"❌ Retrieval stage failed: {e}")
        st.text(traceback.format_exc())
        raise

def is_table_chunk(item: Dict[str, Any]) -> bool:
    stype = (item.get("section_type") or "").lower()
    if "table" in stype or "table_summary" in stype:
        return True
    if item.get("table_id") or item.get("table_part_index"):
        return True
    pdf = (item.get("pdf_name") or "").lower()
    if pdf.endswith(".xlsx") or pdf.endswith(".xls"):
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

# ---------------- Core RAG ----------------
def run_rag_pipeline(query, scope="tables"):
    """Main pipeline — safe for hosted execution."""
    try:
        retrieval_script = TABLES_ONLY_SCRIPT if scope == "tables" else DEFAULT_RETRIEVAL_SCRIPT
        retrieval = run_retrieval(query, retrieval_script)

        sem_results = retrieval.get("semantic", {}).get("results", [])
        kw_results = retrieval.get("keyword", {}).get("results", [])

        def filter_scope(items):
            if scope == "tables":
                return [i for i in items if is_table_chunk(i)]
            elif scope == "text":
                return [i for i in items if not is_table_chunk(i)]
            return items

        merged_chunks = dedupe_and_merge(
            filter_scope(sem_results),
            filter_scope(kw_results)
        )

        if not merged_chunks:
            return {"answer": "No relevant chunks found for this query."}

        # --- OpenAI Call ---
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        system_prompt = (
            "You are a formal company assistant. "
            "Use only provided chunks, cite as [chunk_id | pdf_name]."
        )
        chunk_context = "\n".join([
            json.dumps({
                "chunk_id": c["chunk_id"],
                "pdf_name": c["pdf_name"],
                "page": c["page"],
                "content": c["content"]
            }, ensure_ascii=False)
            for c in merged_chunks
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: {query}\n\nContext:\n{chunk_context}"}
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=700
        )

        answer = response.choices[0].message.content
        pdf_names = sorted({Path(c["pdf_name"]).stem for c in merged_chunks if c.get("pdf_name")})
        if pdf_names:
            answer += "\n\n**Sources Referenced:** " + ", ".join(pdf_names)

        return answer

    except Exception as e:
        st.error(f"❌ RAG pipeline error: {e}")
        st.text(traceback.format_exc())
        raise

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

tabs = st.tabs(["📂 Processed Files", "💬 Query Interface"])

# --- Tab 1 ---
with tabs[0]:
    st.header("📂 Processed Files")
    st.markdown("""
    These are the processed files currently available for querying.
    > ⚙️ *Pre-embedded for retrieval.*
    """)
    st.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

# --- Tab 2 ---
with tabs[1]:
    st.header("💬 RAG Query Interface")

    col1, col2 = st.columns([4, 2])
    with col1:
        query = st.text_area("Enter your query:")
    with col2:
        scope_option = st.selectbox("Retrieval Scope", ["Text + Tables", "Tables Only"])
        run_button = st.button("Run Query")

    if run_button:
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Running RAG pipeline..."):
                try:
                    scope = "tables" if scope_option == "Tables Only" else "all"
                    answer = run_rag_pipeline(query.strip(), scope=scope)
                    st.success("✅ Query processed successfully.")
                    st.markdown("### 🧠 Response")
                    st.write(answer)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.text(traceback.format_exc())

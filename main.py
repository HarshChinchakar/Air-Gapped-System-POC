#!/usr/bin/env python3
"""
Streamlit RAG Interface (Tables / Text + Tables selector)
Final Version — Creator: Harsh Chinchakar
(With retrieval script path and runtime error tracing)
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
    st.error(f"⚠️ Failed to load .env file: {e}")

# ---------------- CONFIG ----------------
DEFAULT_RETRIEVAL_SCRIPT = "Retrival/retrieval_combined_v2.py"
TABLES_ONLY_SCRIPT = "Retrival/retrival_tables.py"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
SEMANTIC_TOP = 12
KEYWORD_TOP = 12
CHUNK_CHAR_LIMIT = 2500
TIMEOUT_SECONDS = 120

# ---------------- Helper Functions ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def run_retrieval(query: str, retrieval_script: str):
    """Executes the retrieval script via subprocess and returns parsed JSON output, with debug tracing."""
    try:
        # Check if the retrieval script path exists before running
        if not os.path.exists(retrieval_script):
            raise FileNotFoundError(f"Retrieval script not found at path: {retrieval_script}")

        cmd = f"python3 {shlex.quote(retrieval_script)} --query {shlex.quote(query)} --top_k {max(SEMANTIC_TOP, KEYWORD_TOP)}"
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )

        if proc.returncode != 0:
            # Show both stdout and stderr for better visibility
            raise RuntimeError(
                f"Retrieval script failed (exit code {proc.returncode}).\n\n"
                f"STDERR:\n{proc.stderr}\n\nSTDOUT:\n{proc.stdout}"
            )

        match = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", proc.stdout)
        if not match:
            raise RuntimeError(
                f"No valid JSON found in retrieval output.\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )

        return json.loads(match.group(1))

    except FileNotFoundError as fnf:
        st.error(f"❌ Retrieval script missing: {fnf}")
        st.text(traceback.format_exc())
        raise
    except subprocess.TimeoutExpired:
        st.error("❌ Retrieval script timed out.")
        raise
    except Exception as e:
        st.error(f"❌ Retrieval script error: {e}")
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

# ---------------- Core RAG Logic ----------------
def run_rag_pipeline(query, scope="tables"):
    """Main RAG logic with retrieval error tracing."""
    try:
        if scope == "tables":
            retrieval_script = TABLES_ONLY_SCRIPT
            apply_filter = False
        else:
            retrieval_script = DEFAULT_RETRIEVAL_SCRIPT
            apply_filter = True

        st.info(f"🧠 Running retrieval using: {retrieval_script}")
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

        if not merged_chunks:
            return {"answer": "No relevant chunks found for this query."}

        # --- OpenAI section ---
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        system_prompt = (
            "Rules (obey strictly):\n"
            "1) Use ONLY information present in provided chunks.\n"
            "2) If inferred, mark as 'Inferred' and cite chunks.\n"
            "3) If nothing relevant found, respond with 'Information not found in retrieved dataset.'\n"
            "4) Cite with [chunk_id | pdf_name].\n"
            "5) Be concise, formal, include 'Sources' and 'Limitations' sections.\n"
            "6) Never fabricate numeric values."
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

        pdf_names = sorted(set([
            Path(c["pdf_name"]).stem for c in merged_chunks if c.get("pdf_name")
        ]))
        if pdf_names:
            citations = "\n\n**Sources Referenced:** " + ", ".join(pdf_names)
            answer = answer.strip() + citations

        return answer

    except Exception as e:
        st.error(f"❌ run_rag_pipeline() failed: {e}")
        st.text(traceback.format_exc())
        raise

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

tabs = st.tabs(["📂 Processed Files", "💬 Query Interface"])

# --- Tab 1: Processed Files ---
with tabs[0]:
    st.header("📂 Processed Files (Available for Querying)")
    st.markdown("""
    These are the **processed files** currently available for querying through the RAG system.
    > ⚙️ *Only these datasets are pre-embedded for now.*
    """)

    st.subheader("📘 Source Files (Text Content)")
    st.markdown("""
    - APR & MAY  
    - APR & MAY__dup1  
    - APR TO AUG  
    - Apr to June 20  
    - April 2020 To March 2021  
    """)

    st.subheader("📊 Source Files (Table Summaries)")
    st.markdown("""
    - Axis Bank Statement-April-20 to Oct-20  
    - CHEMBUR FY 20-21  
    - CIBIL Score_PMIPL_As on Nov 2020  
    - Final rejection Summary sheet SPC VSM  
    - MIS-Poshs Metal-Mar-21
    """)

    st.markdown("---")
    st.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

# --- Tab 2: Main Query Interface ---
with tabs[1]:
    st.header("💬 RAG Query Interface")
    st.sidebar.title("Guidelines")

    st.sidebar.subheader("[*] Do’s")
    st.sidebar.markdown("""
    - Keep queries **specific**.  
    - Choose correct retrieval scope.  
    - Verify results using sources.
    """)

    st.sidebar.subheader("[*] Don’ts")
    st.sidebar.markdown("""
    - Don’t query unprocessed files.  
    - Don’t expect live data or updates.  
    - Avoid vague questions.
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

    col1, col2 = st.columns([4, 2])
    with col1:
        query = st.text_area(
            "Enter your query:",
            placeholder="e.g. Provide a summary of outstanding loans and their repayment schedules."
        )
    with col2:
        scope_option = st.selectbox("Retrieval Scope", ["Text + Tables", "Tables Only"], index=0)
        run_button = st.button("Run Query")

    if run_button and query.strip():
        with st.spinner("Running RAG pipeline..."):
            scope = "tables" if scope_option == "Tables Only" else "all"
            try:
                answer = run_rag_pipeline(query.strip(), scope=scope)
                st.success("✅ Query processed successfully.")
                st.markdown("### 🧠 Response")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.text(traceback.format_exc())

    elif run_button:
        st.warning("Please enter a query before running.")

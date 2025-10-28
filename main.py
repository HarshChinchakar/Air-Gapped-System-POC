#!/usr/bin/env python3
"""
Streamlit RAG Interface — Verbose step-by-step debug version
Creator: Harsh Chinchakar

Behavior:
- Prints explicit progress logs for every step of the pipeline.
- Uses sys.executable for subprocess calls (same venv as Streamlit).
- Shows stdout/stderr and tracebacks inline in the UI.
- Parses and displays retrieved chunks.
- Optionally runs the OpenAI generation step (toggleable).
"""

import os
import sys
import json
import re
import subprocess
import traceback
import time
from pathlib import Path
from datetime import datetime
import streamlit as st
from typing import Dict, Any, List
from dotenv import load_dotenv

# ---------------- Setup ----------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # no error if missing

# Retrieval scripts (absolute)
DEFAULT_RETRIEVAL_SCRIPT = str(BASE_DIR / "Retrival" / "retrieval_combined_v2.py")
TABLES_ONLY_SCRIPT = str(BASE_DIR / "Retrival" / "retrival_tables.py")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("RAG_MODEL", "gpt-4o-mini")
SEMANTIC_TOP = int(os.getenv("SEMANTIC_TOP", 12))
KEYWORD_TOP = int(os.getenv("KEYWORD_TOP", 12))
CHUNK_CHAR_LIMIT = int(os.getenv("CHUNK_CHAR_LIMIT", 2500))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 120))

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Assistant — Verbose Debug", page_icon="🧪", layout="wide")
st.title("🧪 RAG Assistant — Verbose Debug Mode")
st.markdown("This page runs the RAG retrieval pipeline with **step-by-step** log updates so you always know what's happening.")

# Left column: controls, Right column: log & outputs
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Controls")
    query_input = st.text_area("Enter query (or leave blank to use a test query):", height=120)
    if not query_input:
        query_input = "Summarize the available documents and list any numeric figures mentioned."

    scope_option = st.selectbox("Retrieval Scope", ["Text + Tables", "Tables Only"], index=0)
    run_openai = st.checkbox("Run OpenAI generation after retrieval (requires OPENAI_API_KEY)", value=False)
    run_button = st.button("Run Pipeline (verbose)")

with col_right:
    st.subheader("Step-by-step Log")
    log_box = st.empty()   # will render logs as we go
    stdout_box = st.empty()  # retrieval stdout
    stderr_box = st.empty()  # retrieval stderr
    chunks_box = st.empty()  # show retrieved chunks
    answer_box = st.empty()  # final answer or OpenAI output

# Helper for logging to the UI with timestamps
logs: List[str] = []
def log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = f"[{ts}] {msg}"
    logs.append(entry)
    # show as code (monospace) so newlines preserved
    log_box.code("\n".join(logs), language="text")

# ---------------- Utility functions (unchanged logic) ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def is_table_chunk(item: Dict[str, Any]) -> bool:
    try:
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
    except Exception:
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

# ---------------- Core: run_retrieval with verbose logging ----------------
def run_retrieval_verbose(query: str, retrieval_script: str):
    """Run retrieval script using same python executable, and log every step."""
    log(f"STEP 1: Verify retrieval script path: {retrieval_script}")
    if not Path(retrieval_script).exists():
        log(f"ERROR: Retrieval script not found at `{retrieval_script}`")
        raise FileNotFoundError(f"Retrieval script not found: {retrieval_script}")
    log("OK: Retrieval script exists.")

    log(f"STEP 2: Show python executable used by Streamlit: `{sys.executable}`")
    # Optionally verify that numpy is importable in this interpreter
    try:
        import importlib
        numpy_spec = importlib.util.find_spec("numpy")
        log(f"STEP 3: numpy importable? {'Yes' if numpy_spec else 'No'} (spec: {numpy_spec})")
    except Exception as e:
        log(f"STEP 3: numpy check raised exception: {e}")

    log("STEP 4: Prepare subprocess command (using sys.executable).")
    cmd = [
        sys.executable,
        retrieval_script,
        "--query", query,
        "--top_k", str(max(SEMANTIC_TOP, KEYWORD_TOP))
    ]
    log(f"CMD: {' '.join(shlex_quote(a) for a in cmd)}")

    log("STEP 5: Running retrieval subprocess (this will capture stdout/stderr).")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(BASE_DIR)
        )
        log(f"STEP 5b: Subprocess finished with exit code {proc.returncode}")
        # show stdout and stderr in the UI separately
        stdout_box.text_area("Retrieval STDOUT", proc.stdout or "<no stdout>", height=240)
        if proc.stderr and proc.stderr.strip():
            stderr_box.text_area("Retrieval STDERR", proc.stderr, height=240)
            log("STEP 5c: Retrieval STDERR printed above.")
        else:
            log("STEP 5c: No STDERR from retrieval script.")
        # If non-zero exit, raise (but still show outputs)
        if proc.returncode != 0:
            log("STEP 5d: Subprocess exit code non-zero — raising error to caller.")
            raise RuntimeError(f"Retrieval script failed (exit {proc.returncode}). See STDOUT/STDERR above.")
    except subprocess.TimeoutExpired as te:
        log("ERROR: Retrieval subprocess timed out.")
        log(traceback.format_exc())
        raise
    except Exception:
        log("ERROR: Exception while running retrieval subprocess.")
        log(traceback.format_exc())
        raise

    log("STEP 6: Parsing retrieval output for JSON marker `RETRIEVAL_JSON_OUTPUT:`")
    # Try to extract JSON first by specified marker, otherwise try to load full stdout as JSON
    try:
        stdout = proc.stdout or ""
        m = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", stdout)
        if m:
            payload = json.loads(m.group(1))
            log("OK: Found JSON with RETRIEVAL_JSON_OUTPUT marker.")
        else:
            # fallback: find first JSON object in stdout
            m2 = re.search(r"(\{[\s\S]+\})", stdout)
            if m2:
                payload = json.loads(m2.group(1))
                log("OK: Found a JSON object in STDOUT (fallback).")
            else:
                log("ERROR: No JSON object found in retrieval stdout.")
                raise RuntimeError("No JSON found in retrieval stdout.")
    except Exception:
        log("ERROR: Failed to parse retrieval output as JSON.")
        log(traceback.format_exc())
        raise

    log("STEP 7: Extract semantic & keyword results from payload.")
    sem_results = payload.get("semantic", {}).get("results", [])
    kw_results = payload.get("keyword", {}).get("results", [])
    log(f"Found semantic results: {len(sem_results)}, keyword results: {len(kw_results)}")

    log("STEP 8: Apply scope filter (tables/text/all).")
    # apply filter logic as in original
    def filter_scope(items):
        if scope_option == "Tables Only":
            return [i for i in items if is_table_chunk(i)]
        elif scope_option == "Text + Tables":
            return items
        return items

    sem_filtered = filter_scope(sem_results)
    kw_filtered = filter_scope(kw_results)
    log(f"After filtering: semantic {len(sem_filtered)}, keyword {len(kw_filtered)}")

    log("STEP 9: Deduplicate and merge top-k results.")
    merged = dedupe_and_merge(sem_filtered, kw_filtered)
    log(f"STEP 9 DONE: Merged chunk count = {len(merged)}")

    return merged

# helper to create safe command display
def shlex_quote(s: str) -> str:
    # simple quoting for display (avoid importing shlex to compress)
    if " " in s or '"' in s:
        return f'"{s}"'
    return s

# ---------------- Run button action ----------------
if run_button:
    logs.clear()
    log("=== RUN STARTED ===")
    log(f"Base dir: {BASE_DIR}")
    log(f"Retrieval scripts:\n - Combined: {DEFAULT_RETRIEVAL_SCRIPT}\n - Tables: {TABLES_ONLY_SCRIPT}")

    # Step: Check files exist
    try:
        log("STEP 0: Checking existence of retrieval scripts and .env")
        for name, p in [("combined", DEFAULT_RETRIEVAL_SCRIPT), ("tables", TABLES_ONLY_SCRIPT), (".env", str(BASE_DIR / ".env"))]:
            exists = Path(p).exists()
            log(f" - {name}: {'FOUND' if exists else 'MISSING'} -> {p}")
    except Exception:
        log("ERROR during file existence checks.")
        log(traceback.format_exc())

    # Step: run retrieval (with explicit logs for every sub-step)
    try:
        log("STEP A: Starting retrieval stage")
        # choose script
        script_to_use = TABLES_ONLY_SCRIPT if scope_option == "Tables Only" else DEFAULT_RETRIEVAL_SCRIPT
        log(f"Selected retrieval script: {script_to_use}")
        # Run retrieval and get merged chunks
        merged_chunks = run_retrieval_verbose(query_input, script_to_use)
        log("STEP A DONE: Retrieval completed and parsed.")
    except Exception as e:
        log(f"STEP A FAILED: {e}")
        # Already printed traceback in run_retrieval_verbose, so stop the pipeline
        log("=== RUN ABORTED DUE TO ERROR ===")
        raise

    # Step: display chunks (and explicit marking of completion)
    try:
        log("STEP B: Displaying retrieved chunks (top 50).")
        if not merged_chunks:
            log("No chunks to display.")
            chunks_box.warning("⚠️ No chunks retrieved.")
        else:
            # build a textual summary
            txt = []
            for i, c in enumerate(merged_chunks[:50], start=1):
                ci = c.copy()
                content = ci.get("content", "")
                snippet = content if len(content) < 800 else content[:800] + " ... [truncated]"
                txt.append(f"=== CHUNK {i} ===\nchunk_id: {ci.get('chunk_id')}\npdf_name: {ci.get('pdf_name')}\npage: {ci.get('page')}\nscore: {ci.get('score')}\ncontent:\n{snippet}\n")
            chunks_box.code("\n\n".join(txt), language="text")
            log(f"STEP B DONE: Displayed {min(len(merged_chunks),50)} chunks.")
    except Exception:
        log("ERROR while displaying chunks.")
        log(traceback.format_exc())

    # Step: optionally run OpenAI generation (explicit step logs)
    if run_openai:
        log("STEP C: OpenAI generation requested.")
        if not OPENAI_API_KEY:
            log("ERROR: OPENAI_API_KEY not set in environment — skipping OpenAI generation.")
            answer_box.error("OpenAI generation skipped: OPENAI_API_KEY not found.")
        else:
            try:
                log("STEP C.1: Preparing system/user prompts.")
                system_prompt = (
                    "You are a formal company assistant answering from retrieved document chunks.\n"
                    "Use only provided data. If inferred, mark as 'Inferred'. Cite chunks as [chunk_id | pdf_name]."
                )
                chunk_context = "\n".join([
                    json.dumps({
                        "chunk_id": c["chunk_id"],
                        "pdf_name": c["pdf_name"],
                        "page": c["page"],
                        "content": c["content"]
                    }, ensure_ascii=False)
                    for c in merged_chunks
                ])[:16000]  # small guard

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {query_input}\n\nContext:\n{chunk_context}"}
                ]
                log("STEP C.2: Calling OpenAI client (this may take a few seconds).")
                # import and call OpenAI client (preserve original behavior)
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=700
                )
                answer_text = response.choices[0].message.content
                # append citations
                pdf_names = sorted({Path(c["pdf_name"]).stem for c in merged_chunks if c.get("pdf_name")})
                if pdf_names:
                    answer_text = answer_text.strip() + "\n\n**Sources Referenced:** " + ", ".join(pdf_names)
                answer_box.markdown("### 🧠 OpenAI Answer")
                answer_box.code(answer_text, language="text")
                log("STEP C DONE: OpenAI generation completed and displayed.")
            except Exception:
                log("ERROR during OpenAI generation step.")
                log(traceback.format_exc())
                answer_box.error("OpenAI generation failed — see logs.")

    # Finalize
    log("=== RUN COMPLETE ===")
    st.success("Pipeline run finished — check logs and output panels on the right.")
else:
    st.info("Ready. Enter a query and press **Run Pipeline (verbose)** to start. Logs will appear here step-by-step.")

# #!/usr/bin/env python3
# """
# main.py

# Streamlit diagnostic runner — runs two retrieval scripts with a hardcoded query,
# prints step-by-step progress, stdout/stderr, exit codes, and attempts to parse JSON output.

# Drop this file at your repo root (same level as `Retrival/`) and run:
#     streamlit run main.py

# Author: Harsh Chinchakar
# """

# import sys
# import os
# import json
# import re
# import subprocess
# import traceback
# from pathlib import Path
# from datetime import datetime
# import streamlit as st
# from dotenv import load_dotenv
# import importlib

# # ---------------- Config ----------------
# BASE_DIR = Path(__file__).resolve().parent
# load_dotenv(BASE_DIR / ".env")  # silent if missing

# RETRIEVAL_COMBINED = BASE_DIR / "Retrival" / "retrieval_combined_v2.py"
# RETRIEVAL_TABLES = BASE_DIR / "Retrival" / "retrival_tables.py"

# HARD_CODED_QUERY = (
#     "Hardcoded test query: Summarize the key financial items and any numeric figures present in the documents."
# )

# # Timeout in seconds for subprocess runs
# SUBPROCESS_TIMEOUT = 180

# # ---------------- Streamlit UI ----------------
# st.set_page_config(page_title="Retrieval Runner — Hardcoded Query", page_icon="🧪", layout="wide")
# st.title("Retrieval Runner — Hardcoded Query")
# st.markdown("This page will run the two retrieval scripts (combined + tables) using the same Python interpreter as Streamlit and print everything that happens.")

# col_controls, col_output = st.columns([1, 2])

# with col_controls:
#     st.subheader("Controls")
#     st.markdown(f"**Base dir:** `{BASE_DIR}`")
#     run_button = st.button("Run retrieval scripts (hardcoded query)")
#     st.markdown("---")
#     st.write("Hardcoded query that will be used:")
#     st.code(HARD_CODED_QUERY)

# with col_output:
#     st.subheader("Step-by-step log")
#     log_box = st.empty()
#     details_expander = st.expander("Detailed output (stdout/stderr/parsed JSON)", expanded=True)
#     stdout_area = details_expander.empty()
#     stderr_area = details_expander.empty()
#     parsed_area = details_expander.empty()

# # small helper to timestamp logs
# logs = []
# def add_log(msg: str):
#     ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
#     logs.append(f"[{ts}] {msg}")
#     log_box.code("\n".join(logs), language="text")

# def run_script_and_capture(script_path: Path, query: str) -> dict:
#     """
#     Run the given script via the same python executable used by Streamlit.
#     Returns a dictionary with keys: exists, cmd, returncode, stdout, stderr, parsed_json (or None), error
#     """
#     result = {
#         "script": str(script_path),
#         "exists": script_path.exists(),
#         "cmd": None,
#         "returncode": None,
#         "stdout": None,
#         "stderr": None,
#         "parsed_json": None,
#         "error": None,
#     }

#     add_log(f"Checking script path: {script_path}")
#     if not script_path.exists():
#         result["error"] = f"Script not found at {script_path}"
#         add_log("ERROR: " + result["error"])
#         return result

#     # Use the same python interpreter as Streamlit
#     python_exec = sys.executable
#     cmd = [
#         python_exec,
#         str(script_path),
#         "--query", query,
#         "--top_k", "5"
#     ]
#     result["cmd"] = " ".join(cmd)
#     add_log(f"Preparing to run: {result['cmd']}")
#     add_log(f"Working directory: {BASE_DIR}")
#     add_log(f"Using Python executable: {python_exec}")

#     # show quick availability of key packages
#     try:
#         np_spec = importlib.util.find_spec("numpy")
#         st_spec = importlib.util.find_spec("sentence_transformers")
#         faiss_spec = importlib.util.find_spec("faiss")
#         add_log(f"Package check -> numpy: {'FOUND' if np_spec else 'MISSING'}, sentence_transformers: {'FOUND' if st_spec else 'MISSING'}, faiss: {'FOUND' if faiss_spec else 'MISSING'}")
#     except Exception as e:
#         add_log(f"Package check error: {e}")

#     try:
#         proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT, cwd=str(BASE_DIR))
#         result["returncode"] = proc.returncode
#         result["stdout"] = proc.stdout
#         result["stderr"] = proc.stderr
#         add_log(f"Process finished with exit code {proc.returncode}")

#         # Display raw outputs in UI
#         stdout_area.text_area(f"STDOUT for {script_path.name}", proc.stdout or "<no stdout>", height=300)
#         if proc.stderr and proc.stderr.strip():
#             stderr_area.text_area(f"STDERR for {script_path.name}", proc.stderr, height=300)
#         else:
#             stderr_area.text_area(f"STDERR for {script_path.name}", "<no stderr>", height=120)

#         # Try to parse JSON from stdout
#         try:
#             m = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", proc.stdout)
#             if m:
#                 payload = json.loads(m.group(1))
#                 result["parsed_json"] = payload
#                 parsed_area.json(payload)
#                 add_log("Parsed JSON found via RETRIEVAL_JSON_OUTPUT marker.")
#             else:
#                 # fallback: find first JSON object in stdout
#                 m2 = re.search(r"(\{[\s\S]+\})", proc.stdout)
#                 if m2:
#                     payload = json.loads(m2.group(1))
#                     result["parsed_json"] = payload
#                     parsed_area.json(payload)
#                     add_log("Parsed JSON found via fallback (first JSON object in stdout).")
#                 else:
#                     add_log("No JSON found in stdout.")
#         except Exception as e:
#             result["error"] = f"JSON parse error: {e}"
#             add_log("ERROR parsing JSON: " + str(e))
#             parsed_area.text("Failed to parse JSON from stdout. See stdout/stderr above.")
#     except subprocess.TimeoutExpired:
#         result["error"] = f"Timeout expired after {SUBPROCESS_TIMEOUT} seconds."
#         add_log("ERROR: " + result["error"])
#         stderr_area.text(result["error"])
#     except Exception as e:
#         result["error"] = f"Exception while running script: {e}\n{traceback.format_exc()}"
#         add_log("ERROR: Exception while running script.")
#         stderr_area.text(result["error"])

#     return result

# # Run both scripts when button pressed
# if run_button:
#     logs.clear()
#     add_log("=== RUN STARTED ===")
#     add_log(f"Base directory: {BASE_DIR}")
#     add_log(f"Combined retrieval script path: {RETRIEVAL_COMBINED}")
#     add_log(f"Tables retrieval script path: {RETRIEVAL_TABLES}")
#     add_log("Hardcoded query will be used (displayed on left).")

#     # Run combined retrieval
#     add_log(">>> Running combined retrieval script")
#     res_combined = run_script_and_capture(RETRIEVAL_COMBINED, HARD_CODED_QUERY)
#     add_log(f"Combined retrieval finished. Exit code: {res_combined.get('returncode')}, error: {res_combined.get('error')}")

#     add_log("----- Pause (small) -----")
#     # optional short sleep to ensure UI updates
#     # time.sleep(0.5)

#     # Run tables-only retrieval
#     add_log(">>> Running tables-only retrieval script")
#     res_tables = run_script_and_capture(RETRIEVAL_TABLES, HARD_CODED_QUERY)
#     add_log(f"Tables retrieval finished. Exit code: {res_tables.get('returncode')}, error: {res_tables.get('error')}")

#     add_log("=== RUN COMPLETE ===")
#     st.success("Run complete — scroll through the Step-by-step log and the Detailed output panels for full stdout/stderr/JSON.")

# else:
#     st.info("Click 'Run retrieval scripts (hardcoded query)' to execute both retrieval scripts and show all outputs.")

#!/usr/bin/env python3
"""
main.py

Retrieval runner — robust version with defensive error handling.
Runs two retrieval scripts (combined = text+tables, and tables-only)
with a hardcoded query, prints step-by-step progress, stdout/stderr,
attempts to parse JSON, and displays deduped retrieved chunks.

Author: Harsh Chinchakar
"""

import sys
import os
import json
import re
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import importlib
from typing import Dict, Any, List

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # silent if missing

RETRIEVAL_COMBINED = BASE_DIR / "Retrival" / "retrieval_combined_v2.py"   # text + tables (combined)
RETRIEVAL_TABLES = BASE_DIR / "Retrival" / "retrival_tables.py"          # tables-only

HARD_CODED_QUERY = (
    "Hardcoded test query: Summarize the key financial items and any numeric figures present in the documents."
)

SUBPROCESS_TIMEOUT = 180  # seconds
TOP_K = 8  # top results for merging
CHUNK_CHAR_LIMIT = 2500

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Retrieval Runner — Robust", page_icon="🧪", layout="wide")
st.title("Retrieval Runner — Robust (with full error tracing)")
st.markdown("Runs both retrieval scripts and shows detailed stdout/stderr, parsed JSON and merged chunks. Any error will be printed in the logs and the stderr panel.")

# Left pane: controls & secrets
col_controls, col_output = st.columns([1, 2])

with col_controls:
    st.subheader("Controls")
    st.markdown(f"**Base dir:** `{BASE_DIR}`")
    st.markdown("**Scripts to run**:")
    st.code(f"1) {RETRIEVAL_COMBINED}\n2) {RETRIEVAL_TABLES}")

    st.markdown("---")
    st.write("Hardcoded query (used for both retrieval scripts):")
    st.code(HARD_CODED_QUERY)

    st.markdown("---")
    # Fetch OpenAI key from multiple places (safe)
    openai_env = os.getenv("OPENAI_API_KEY") or ""
    openai_secret_top = ""
    try:
        # safe access to st.secrets
        if hasattr(st, "secrets"):
            if "OPENAI_API_KEY" in st.secrets:
                openai_secret_top = st.secrets.get("OPENAI_API_KEY", "")
            elif "openai" in st.secrets and isinstance(st.secrets["openai"], dict):
                openai_secret_top = st.secrets["openai"].get("api_key", "")
    except Exception:
        openai_secret_top = ""

    openai_key_effective = openai_env or openai_secret_top or ""

    def mask_key(k: str) -> str:
        if not k:
            return "<NOT SET>"
        if len(k) <= 8:
            return k[0:1] + "*"*(len(k)-2) + k[-1:]
        return k[:4] + "*"*(max(6, len(k)-8)) + k[-4:]

    st.subheader("OpenAI Key (masked)")
    st.markdown(f"- env OPENAI_API_KEY present: `{bool(openai_env)}`")
    st.markdown(f"- st.secrets OPENAI_API_KEY present: `{bool(openai_secret_top and 'OPENAI_API_KEY' in getattr(st, 'secrets', {}))}`")
    st.markdown(f"- st.secrets.openai.api_key present: `{bool(openai_secret_top and 'openai' in getattr(st, 'secrets', {}))}`")
    st.code(mask_key(openai_key_effective))
    reveal_key = st.checkbox("Reveal raw OpenAI key (risky)", value=False)

    st.markdown("---")
    run_button = st.button("Run retrieval scripts (hardcoded query)")

# Right pane: logs and detailed outputs
with col_output:
    st.subheader("Step-by-step log")
    log_box = st.empty()
    detail_expander = st.expander("Detailed outputs (stdout / stderr / parsed JSON / chunks)", expanded=True)
    stdout_area = detail_expander.empty()
    stderr_area = detail_expander.empty()
    parsed_area = detail_expander.empty()
    chunks_area = detail_expander.empty()
    traceback_area = detail_expander.empty()

# small helper to timestamp logs
logs: List[str] = []
def add_log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    logs.append(f"[{ts}] {msg}")
    # use code block so newlines kept
    log_box.code("\n".join(logs), language="text")

# ---------------- Helpers ----------------
def check_packages() -> Dict[str, bool]:
    pkgs = {}
    for pkg in ("numpy", "sentence_transformers", "faiss", "faiss_cpu", "faiss-cpu"):
        try:
            spec = importlib.util.find_spec(pkg)
            pkgs[pkg] = bool(spec)
        except Exception:
            pkgs[pkg] = False
    return pkgs

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

def dedupe_and_merge(semantic: List[Dict[str, Any]], keyword: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    combined, seen = [], set()
    for src in (semantic or [])[:TOP_K] + (keyword or [])[:TOP_K]:
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

# ---------------- Core runner ----------------
def run_script_and_capture(script_path: Path, query: str, force_table_filter: bool = False) -> Dict[str, Any]:
    result = {
        "script": str(script_path),
        "exists": script_path.exists(),
        "cmd": None,
        "returncode": None,
        "stdout": None,
        "stderr": None,
        "parsed_json": None,
        "merged_chunks": [],
        "error": None,
    }

    try:
        add_log(f"Checking script path: {script_path}")
        if not script_path.exists():
            result["error"] = f"Script not found at {script_path}"
            add_log("ERROR: " + result["error"])
            return result

        python_exec = sys.executable
        cmd = [python_exec, str(script_path), "--query", query, "--top_k", str(TOP_K)]
        result["cmd"] = " ".join(cmd)
        add_log(f"Running command: {result['cmd']}")
        add_log(f"Working dir: {BASE_DIR}")
        add_log(f"Python exec: {python_exec}")

        # package check
        try:
            pkg_status = check_packages()
            add_log("Package availability: " + ", ".join([f"{k}:{'Y' if v else 'N'}" for k,v in pkg_status.items()]))
        except Exception as e:
            add_log(f"Package check failed: {e}")

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT, cwd=str(BASE_DIR))
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr

        # show raw outputs
        stdout_area.text_area(f"STDOUT for {script_path.name}", proc.stdout or "<no stdout>", height=300)
        stderr_area.text_area(f"STDERR for {script_path.name}", proc.stderr or "<no stderr>", height=200)

        add_log(f"Process exit code: {proc.returncode}")

        # parse JSON robustly: try marker, then last JSON blob
        payload = None
        try:
            if proc.stdout:
                m = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", proc.stdout)
                if m:
                    payload = json.loads(m.group(1))
                    add_log("Parsed JSON via RETRIEVAL_JSON_OUTPUT marker.")
                else:
                    # try to find the last JSON object in stdout to avoid partial logs earlier
                    matches = list(re.finditer(r"(\{[\s\S]*?\})", proc.stdout))
                    if matches:
                        last = matches[-1].group(1)
                        payload = json.loads(last)
                        add_log("Parsed JSON via fallback (last JSON blob in stdout).")
        except Exception as je:
            add_log("JSON parse exception: " + str(je))
            # include traceback in stderr area as well
            stderr_area.text(proc.stderr or "" + "\n\nJSON parse exception:\n" + traceback.format_exc())

        result["parsed_json"] = payload

        if payload:
            sem_results = []
            kw_results = []
            try:
                sem_obj = payload.get("semantic", {})
                if isinstance(sem_obj, dict):
                    sem_results = sem_obj.get("results", []) or []
                kw_obj = payload.get("keyword", {})
                if isinstance(kw_obj, dict):
                    kw_results = kw_obj.get("results", []) or []
                add_log(f"Found semantic: {len(sem_results)}, keyword: {len(kw_results)}")
            except Exception:
                add_log("Error extracting semantic/keyword arrays.")
                stderr_area.text("Error extracting semantic/keyword arrays:\n" + traceback.format_exc())

            # apply table-only filter if requested
            if force_table_filter:
                try:
                    sem_results = [i for i in sem_results if is_table_chunk(i)]
                    kw_results = [i for i in kw_results if is_table_chunk(i)]
                    add_log(f"Applied table filter -> semantic: {len(sem_results)}, keyword: {len(kw_results)}")
                except Exception:
                    add_log("Error applying table filter.")
                    stderr_area.text("Error applying table filter:\n" + traceback.format_exc())

            try:
                merged = dedupe_and_merge(sem_results, kw_results)
                result["merged_chunks"] = merged
                add_log(f"Merged deduped chunk count: {len(merged)}")
            except Exception:
                add_log("Error during dedupe_and_merge.")
                stderr_area.text("Error during dedupe_and_merge:\n" + traceback.format_exc())
        else:
            add_log("No JSON payload parsed from stdout; merged_chunks will be empty.")

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout expired after {SUBPROCESS_TIMEOUT} s."
        add_log("ERROR: " + result["error"])
        stderr_area.text(result["error"])
    except Exception as e:
        result["error"] = f"Exception running script: {e}"
        add_log("UNEXPECTED ERROR: " + str(e))
        traceback_area.text(traceback.format_exc())
        stderr_area.text(str(e) + "\n\n" + traceback.format_exc())

    return result

# ---------------- Main action ----------------
if run_button:
    logs.clear()
    add_log("=== RUN STARTED ===")
    add_log(f"Base dir: {BASE_DIR}")
    add_log(f"Combined script: {RETRIEVAL_COMBINED}")
    add_log(f"Tables script: {RETRIEVAL_TABLES}")

    # reveal key optionally
    if reveal_key and openai_key_effective:
        add_log(f"OpenAI key (REVEALED): {openai_key_effective}")
    else:
        add_log(f"OpenAI key present: {bool(openai_key_effective)} (masked in UI)")

    # Run combined retrieval
    add_log(">>> Running combined retrieval (text + tables)")
    try:
        res_combined = run_script_and_capture(RETRIEVAL_COMBINED, HARD_CODED_QUERY, force_table_filter=False)
        add_log(f"Combined exit: {res_combined.get('returncode')}, error: {res_combined.get('error')}")
    except Exception:
        add_log("Combined retrieval raised an unhandled exception.")
        traceback_area.text(traceback.format_exc())
        st.error("Combined retrieval failed — see logs and stderr above.")
        raise  # re-raise so dev can see in logs

    # Display combined chunks
    try:
        merged_combined = res_combined.get("merged_chunks") or []
        if merged_combined:
            add_log(f"Displaying top {min(50, len(merged_combined))} combined chunks.")
            lines = []
            for i, c in enumerate(merged_combined[:50], start=1):
                snippet = c.get("content", "")
                if len(snippet) > 800:
                    snippet = snippet[:800] + " ... [truncated]"
                lines.append(
                    f"--- CHUNK {i} ---\nchunk_id: {c.get('chunk_id')}\npdf_name: {c.get('pdf_name')}\npage: {c.get('page')}\nscore: {c.get('score')}\ncontent:\n{snippet}\n"
                )
            chunks_area.code("\n".join(lines), language="text")
        else:
            add_log("No merged chunks from combined retrieval.")
            chunks_area.info("No merged chunks from combined retrieval.")
    except Exception:
        add_log("Error while rendering combined chunks.")
        traceback_area.text(traceback.format_exc())
        chunks_area.error("Failed to render combined chunks — see traceback.")

    # Run tables-only retrieval
    add_log(">>> Running tables-only retrieval")
    try:
        res_tables = run_script_and_capture(RETRIEVAL_TABLES, HARD_CODED_QUERY, force_table_filter=True)
        add_log(f"Tables exit: {res_tables.get('returncode')}, error: {res_tables.get('error')}")
    except Exception:
        add_log("Tables retrieval raised an unhandled exception.")
        traceback_area.text(traceback.format_exc())
        st.error("Tables retrieval failed — see logs and stderr above.")
        raise

    # Display tables chunks (append below or replace if none)
    try:
        merged_tables = res_tables.get("merged_chunks") or []
        if merged_tables:
            add_log(f"Displaying top {min(50, len(merged_tables))} table chunks.")
            lines = []
            for i, c in enumerate(merged_tables[:50], start=1):
                snippet = c.get("content", "")
                if len(snippet) > 800:
                    snippet = snippet[:800] + " ... [truncated]"
                lines.append(
                    f"--- TABLE CHUNK {i} ---\nchunk_id: {c.get('chunk_id')}\npdf_name: {c.get('pdf_name')}\npage: {c.get('page')}\nscore: {c.get('score')}\ncontent:\n{snippet}\n"
                )
            # append to existing chunks display safely
            try:
                prev_text = ""
                try:
                    # try reading current value of chunks_area by rendering then reusing logs content
                    # fallback: just render both lists combined
                    chunks_area.code("\n\n".join([chunks_area._rendered_value if hasattr(chunks_area, "_rendered_value") else "", "\n".join(lines)]), language="text")
                except Exception:
                    # safe fallback to simply render lines
                    chunks_area.code("\n".join(lines), language="text")
            except Exception:
                # last fallback: show in a new area
                st.code("\n".join(lines), language="text")
        else:
            add_log("No merged chunks from tables retrieval.")
            # do not overwrite combined display; show info
            st.info("No merged chunks from tables retrieval.")
    except Exception:
        add_log("Error while rendering table chunks.")
        traceback_area.text(traceback.format_exc())
        st.error("Failed to render table chunks — see traceback.")

    add_log("=== RUN COMPLETE ===")
    st.success("Run complete — check logs, STDOUT/STERR, parsed JSON and chunks above.")
else:
    st.info("Click 'Run retrieval scripts (hardcoded query)' to execute both retrieval scripts and show all outputs and merged retrieved chunks.")

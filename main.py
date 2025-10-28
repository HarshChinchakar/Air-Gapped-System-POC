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

Streamlit retrieval tester — runs two retrieval scripts (combined + tables)
with a user-provided query, prints step-by-step progress, stdout/stderr,
and displays parsed retrieved chunks directly below.

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

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # silent if missing

RETRIEVAL_COMBINED = BASE_DIR / "Retrival" / "retrieval_combined_v2.py"
RETRIEVAL_TABLES = BASE_DIR / "Retrival" / "retrival_tables.py"

SUBPROCESS_TIMEOUT = 180

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Retrieval Runner — Query Tester", page_icon="🧠", layout="wide")
st.title("Retrieval Runner — Text + Tables Search")
st.markdown(
    "Enter a query below and run both retrieval pipelines (combined text+tables and tables-only). "
    "Outputs, parsed JSON, and retrieved chunks will be displayed."
)

col_controls, col_output = st.columns([1, 2])

# Left pane — query input & controls
with col_controls:
    st.subheader("Controls")
    st.markdown(f"**Base dir:** `{BASE_DIR}`")
    query_input = st.text_area("Enter your query:", "", height=100, placeholder="e.g. Summarize revenue trends for FY2024...")
    run_button = st.button("Run retrievals")
    st.markdown("---")
    st.markdown("**Scripts to run:**")
    st.code(f"1️⃣ {RETRIEVAL_COMBINED}\n2️⃣ {RETRIEVAL_TABLES}")

# Right pane — logs and detailed output
with col_output:
    st.subheader("Logs & Outputs")
    log_box = st.empty()
    details_expander = st.expander("Detailed output (stdout / stderr / parsed JSON)", expanded=True)
    stdout_area = details_expander.empty()
    stderr_area = details_expander.empty()
    parsed_area = details_expander.empty()
    chunks_area = st.expander("Retrieved Chunks (Parsed from JSON)", expanded=True)

# --------------- Helper Functions ---------------
logs = []
def add_log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    logs.append(f"[{ts}] {msg}")
    log_box.code("\n".join(logs), language="text")

def run_script_and_capture(script_path: Path, query: str) -> dict:
    """Runs a retrieval script and captures stdout, stderr, and parsed JSON."""
    result = {
        "script": str(script_path),
        "exists": script_path.exists(),
        "cmd": None,
        "returncode": None,
        "stdout": None,
        "stderr": None,
        "parsed_json": None,
        "error": None,
    }

    add_log(f"Checking script path: {script_path}")
    if not script_path.exists():
        result["error"] = f"Script not found at {script_path}"
        add_log("ERROR: " + result["error"])
        return result

    python_exec = sys.executable
    cmd = [python_exec, str(script_path), "--query", query, "--top_k", "5"]
    result["cmd"] = " ".join(cmd)
    add_log(f"Preparing to run: {result['cmd']}")

    # Quick package check
    try:
        np_spec = importlib.util.find_spec("numpy")
        st_spec = importlib.util.find_spec("sentence_transformers")
        faiss_spec = importlib.util.find_spec("faiss")
        add_log(
            f"Package check → numpy: {'FOUND' if np_spec else 'MISSING'}, "
            f"sentence_transformers: {'FOUND' if st_spec else 'MISSING'}, "
            f"faiss: {'FOUND' if faiss_spec else 'MISSING'}"
        )
    except Exception as e:
        add_log(f"Package check error: {e}")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=SUBPROCESS_TIMEOUT, cwd=str(BASE_DIR)
        )
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        add_log(f"Process finished with exit code {proc.returncode}")

        # Display stdout/stderr
        stdout_area.text_area(f"STDOUT ({script_path.name})", proc.stdout or "<no stdout>", height=250)
        stderr_area.text_area(f"STDERR ({script_path.name})", proc.stderr or "<no stderr>", height=150)

        # Try to parse JSON output
        try:
            m = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", proc.stdout)
            if m:
                payload = json.loads(m.group(1))
                result["parsed_json"] = payload
                parsed_area.json(payload)
                add_log("Parsed JSON found via marker.")
            else:
                m2 = re.search(r"(\{[\s\S]+\})", proc.stdout)
                if m2:
                    payload = json.loads(m2.group(1))
                    result["parsed_json"] = payload
                    parsed_area.json(payload)
                    add_log("Parsed JSON found via fallback (first JSON object).")
                else:
                    add_log("No JSON found in stdout.")
        except Exception as e:
            result["error"] = f"JSON parse error: {e}"
            add_log("ERROR parsing JSON: " + str(e))
            parsed_area.text("Failed to parse JSON from stdout.")
    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout expired after {SUBPROCESS_TIMEOUT}s."
        add_log(result["error"])
        stderr_area.text(result["error"])
    except Exception as e:
        result["error"] = f"Exception while running: {e}"
        add_log(result["error"])
        stderr_area.text(traceback.format_exc())

    return result


def display_retrieved_chunks(title: str, parsed_json: dict):
    """Display retrieved chunks (semantic + keyword) if available."""
    if not parsed_json:
        chunks_area.warning(f"No parsed JSON for {title}")
        return

    semantic = parsed_json.get("semantic", {}).get("results", [])
    keyword = parsed_json.get("keyword", {}).get("results", [])

    st.subheader(f"🔹 {title} — Retrieved Chunks")
    st.write(f"Semantic results: {len(semantic)} | Keyword results: {len(keyword)}")

    def render_chunks(label, items):
        if not items:
            st.info(f"No {label} results found.")
            return
        for i, chunk in enumerate(items[:15]):  # limit for readability
            st.markdown(
                f"**{label.upper()} #{i+1}**  "
                f"(Score: `{chunk.get('score', 'N/A')}`, "
                f"Page: `{chunk.get('page', 'N/A')}`, "
                f"PDF: `{chunk.get('pdf_name', 'N/A')}`)"
            )
            st.text_area(
                "Content",
                chunk.get("content", "")[:1500],
                height=150,
                key=f"{label}_{i}"
            )

    render_chunks("semantic", semantic)
    render_chunks("keyword", keyword)

# ---------------- Run ----------------
if run_button:
    if not query_input.strip():
        st.warning("⚠️ Please enter a query before running.")
    else:
        logs.clear()
        add_log("=== RUN STARTED ===")
        add_log(f"Query: {query_input}")
        add_log(f"Combined script: {RETRIEVAL_COMBINED}")
        add_log(f"Tables script: {RETRIEVAL_TABLES}")

        # Run combined
        add_log(">>> Running combined (text+tables) retrieval")
        res_combined = run_script_and_capture(RETRIEVAL_COMBINED, query_input)
        add_log(f"Combined retrieval finished. Exit code: {res_combined.get('returncode')}")

        # Run tables-only
        add_log(">>> Running tables-only retrieval")
        res_tables = run_script_and_capture(RETRIEVAL_TABLES, query_input)
        add_log(f"Tables retrieval finished. Exit code: {res_tables.get('returncode')}")

        # Display retrieved chunks below
        st.markdown("---")
        display_retrieved_chunks("Combined (Text + Tables)", res_combined.get("parsed_json"))
        display_retrieved_chunks("Tables Only", res_tables.get("parsed_json"))

        add_log("=== RUN COMPLETE ===")
        st.success("Retrievals complete — scroll down for retrieved chunks.")
else:
    st.info("Enter a query and click 'Run retrievals' to execute both pipelines and view retrieved chunks.")

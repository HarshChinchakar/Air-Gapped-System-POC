#!/usr/bin/env python3
"""
main.py

Streamlit diagnostic runner — runs two retrieval scripts with a hardcoded query,
prints step-by-step progress, stdout/stderr, exit codes, and attempts to parse JSON output.

Drop this file at your repo root (same level as `Retrival/`) and run:
    streamlit run main.py

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

HARD_CODED_QUERY = (
    "Hardcoded test query: Summarize the key financial items and any numeric figures present in the documents."
)

# Timeout in seconds for subprocess runs
SUBPROCESS_TIMEOUT = 180

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Retrieval Runner — Hardcoded Query", page_icon="🧪", layout="wide")
st.title("Retrieval Runner — Hardcoded Query")
st.markdown("This page will run the two retrieval scripts (combined + tables) using the same Python interpreter as Streamlit and print everything that happens.")

col_controls, col_output = st.columns([1, 2])

with col_controls:
    st.subheader("Controls")
    st.markdown(f"**Base dir:** `{BASE_DIR}`")
    run_button = st.button("Run retrieval scripts (hardcoded query)")
    st.markdown("---")
    st.write("Hardcoded query that will be used:")
    st.code(HARD_CODED_QUERY)

with col_output:
    st.subheader("Step-by-step log")
    log_box = st.empty()
    details_expander = st.expander("Detailed output (stdout/stderr/parsed JSON)", expanded=True)
    stdout_area = details_expander.empty()
    stderr_area = details_expander.empty()
    parsed_area = details_expander.empty()

# small helper to timestamp logs
logs = []
def add_log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    logs.append(f"[{ts}] {msg}")
    log_box.code("\n".join(logs), language="text")

def run_script_and_capture(script_path: Path, query: str) -> dict:
    """
    Run the given script via the same python executable used by Streamlit.
    Returns a dictionary with keys: exists, cmd, returncode, stdout, stderr, parsed_json (or None), error
    """
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

    # Use the same python interpreter as Streamlit
    python_exec = sys.executable
    cmd = [
        python_exec,
        str(script_path),
        "--query", query,
        "--top_k", "5"
    ]
    result["cmd"] = " ".join(cmd)
    add_log(f"Preparing to run: {result['cmd']}")
    add_log(f"Working directory: {BASE_DIR}")
    add_log(f"Using Python executable: {python_exec}")

    # show quick availability of key packages
    try:
        np_spec = importlib.util.find_spec("numpy")
        st_spec = importlib.util.find_spec("sentence_transformers")
        faiss_spec = importlib.util.find_spec("faiss")
        add_log(f"Package check -> numpy: {'FOUND' if np_spec else 'MISSING'}, sentence_transformers: {'FOUND' if st_spec else 'MISSING'}, faiss: {'FOUND' if faiss_spec else 'MISSING'}")
    except Exception as e:
        add_log(f"Package check error: {e}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT, cwd=str(BASE_DIR))
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        add_log(f"Process finished with exit code {proc.returncode}")

        # Display raw outputs in UI
        stdout_area.text_area(f"STDOUT for {script_path.name}", proc.stdout or "<no stdout>", height=300)
        if proc.stderr and proc.stderr.strip():
            stderr_area.text_area(f"STDERR for {script_path.name}", proc.stderr, height=300)
        else:
            stderr_area.text_area(f"STDERR for {script_path.name}", "<no stderr>", height=120)

        # Try to parse JSON from stdout
        try:
            m = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", proc.stdout)
            if m:
                payload = json.loads(m.group(1))
                result["parsed_json"] = payload
                parsed_area.json(payload)
                add_log("Parsed JSON found via RETRIEVAL_JSON_OUTPUT marker.")
            else:
                # fallback: find first JSON object in stdout
                m2 = re.search(r"(\{[\s\S]+\})", proc.stdout)
                if m2:
                    payload = json.loads(m2.group(1))
                    result["parsed_json"] = payload
                    parsed_area.json(payload)
                    add_log("Parsed JSON found via fallback (first JSON object in stdout).")
                else:
                    add_log("No JSON found in stdout.")
        except Exception as e:
            result["error"] = f"JSON parse error: {e}"
            add_log("ERROR parsing JSON: " + str(e))
            parsed_area.text("Failed to parse JSON from stdout. See stdout/stderr above.")
    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout expired after {SUBPROCESS_TIMEOUT} seconds."
        add_log("ERROR: " + result["error"])
        stderr_area.text(result["error"])
    except Exception as e:
        result["error"] = f"Exception while running script: {e}\n{traceback.format_exc()}"
        add_log("ERROR: Exception while running script.")
        stderr_area.text(result["error"])

    return result

# Run both scripts when button pressed
if run_button:
    logs.clear()
    add_log("=== RUN STARTED ===")
    add_log(f"Base directory: {BASE_DIR}")
    add_log(f"Combined retrieval script path: {RETRIEVAL_COMBINED}")
    add_log(f"Tables retrieval script path: {RETRIEVAL_TABLES}")
    add_log("Hardcoded query will be used (displayed on left).")

    # Run combined retrieval
    add_log(">>> Running combined retrieval script")
    res_combined = run_script_and_capture(RETRIEVAL_COMBINED, HARD_CODED_QUERY)
    add_log(f"Combined retrieval finished. Exit code: {res_combined.get('returncode')}, error: {res_combined.get('error')}")

    add_log("----- Pause (small) -----")
    # optional short sleep to ensure UI updates
    # time.sleep(0.5)

    # Run tables-only retrieval
    add_log(">>> Running tables-only retrieval script")
    res_tables = run_script_and_capture(RETRIEVAL_TABLES, HARD_CODED_QUERY)
    add_log(f"Tables retrieval finished. Exit code: {res_tables.get('returncode')}, error: {res_tables.get('error')}")

    add_log("=== RUN COMPLETE ===")
    st.success("Run complete — scroll through the Step-by-step log and the Detailed output panels for full stdout/stderr/JSON.")

else:
    st.info("Click 'Run retrieval scripts (hardcoded query)' to execute both retrieval scripts and show all outputs.")


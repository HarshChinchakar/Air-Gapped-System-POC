#!/usr/bin/env python3
"""
Streamlit Diagnostic Script — Creator: Harsh Chinchakar
Checks all retrieval paths and tests if retrieval scripts execute correctly.
"""

import os, subprocess, traceback
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
RETRIEVAL_COMBINED = BASE_DIR / "Retrival" / "retrieval_combined_v2.py"
RETRIEVAL_TABLES = BASE_DIR / "Retrival" / "retrival_tables.py"
ENV_FILE = BASE_DIR / ".env"

# ---------------- UI HEADER ----------------
st.set_page_config(page_title="RAG System Diagnostic", page_icon="🧠", layout="centered")
st.title("🔍 RAG Diagnostic Tool")
st.markdown("Use this tool to verify if your environment and retrieval scripts work correctly.")

# ---------------- STEP 1: CHECK FILES ----------------
st.header("📂 File & Path Checks")

files_to_check = {
    "Base Directory": BASE_DIR,
    ".env File": ENV_FILE,
    "retrieval_combined_v2.py": RETRIEVAL_COMBINED,
    "retrival_tables.py": RETRIEVAL_TABLES,
}

for name, path in files_to_check.items():
    if path.exists():
        st.success(f"✅ {name} found → `{path}`")
    else:
        st.error(f"❌ {name} not found → Expected at `{path}`")

# ---------------- STEP 2: LOAD ENV (optional) ----------------
if ENV_FILE.exists():
    try:
        load_dotenv(str(ENV_FILE))
        st.info("✅ .env file loaded successfully.")
    except Exception as e:
        st.error(f"⚠️ Failed to load .env: {e}")
        st.text(traceback.format_exc())
else:
    st.warning("⚠️ .env file missing — skipping environment variable load.")

# ---------------- STEP 3: TEST RETRIEVAL EXECUTION ----------------
st.header("🧠 Retrieval Script Execution Test")

test_query = "Summarize the data available in the sample documents."
st.markdown(f"**Hardcoded Test Query:** `{test_query}`")

def run_retrieval_test(script_path: Path):
    """Run a retrieval script and return stdout/stderr."""
    if not script_path.exists():
        st.error(f"❌ Script not found: {script_path}")
        return

    st.write(f"▶️ Running `{script_path.name}` ...")
    try:
        cmd = ["python3", str(script_path), "--query", test_query, "--top_k", "5"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90, cwd=str(BASE_DIR))

        st.markdown("**🔹 STDOUT:**")
        st.text_area(f"stdout_{script_path.name}", proc.stdout or "No output", height=200)

        if proc.stderr.strip():
            st.markdown("**🔸 STDERR:**")
            st.text_area(f"stderr_{script_path.name}", proc.stderr, height=200)

        if proc.returncode == 0:
            st.success(f"✅ `{script_path.name}` executed successfully.")
        else:
            st.error(f"❌ `{script_path.name}` failed (exit code {proc.returncode}).")

    except subprocess.TimeoutExpired:
        st.error(f"⚠️ `{script_path.name}` timed out after 90s.")
    except Exception as e:
        st.error(f"❌ Exception while running `{script_path.name}`: {e}")
        st.text(traceback.format_exc())

# --- Buttons ---
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Test retrieval_combined_v2.py"):
        run_retrieval_test(RETRIEVAL_COMBINED)

with col2:
    if st.button("▶️ Test retrival_tables.py"):
        run_retrieval_test(RETRIEVAL_TABLES)

st.markdown("---")
st.markdown("👨‍💻 **Diagnostic created by Harsh Chinchakar**")


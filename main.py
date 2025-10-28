#!/usr/bin/env python3
"""
Streamlit Diagnostic Script — Creator: Harsh Chinchakar
Checks all retrieval paths, verifies environment, and executes retrieval scripts
using the same Python interpreter as Streamlit (fix for missing modules).
"""

import os, sys, subprocess, traceback
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
RETRIEVAL_COMBINED = BASE_DIR / "Retrival" / "retrieval_combined_v2.py"
RETRIEVAL_TABLES = BASE_DIR / "Retrival" / "retrival_tables.py"
ENV_FILE = BASE_DIR / ".env"
TEST_QUERY = "Summarize the data available in the sample documents."

# ---------------- UI HEADER ----------------
st.set_page_config(page_title="RAG System Diagnostic", page_icon="🧠", layout="centered")
st.title("🔍 RAG Diagnostic Tool")
st.markdown("""
Use this diagnostic interface to verify that your environment and retrieval scripts 
are functioning correctly on Streamlit Cloud.
""")

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

# ---------------- STEP 2: ENV + PYTHON INFO ----------------
st.header("🐍 Environment Information")

st.write("**Current Python executable (Streamlit runtime):**", sys.executable)
try:
    import numpy, torch, faiss, sentence_transformers
    st.success("✅ Core modules are available in current environment.")
    st.write("NumPy version:", numpy.__version__)
    st.write("Torch version:", torch.__version__)
except Exception as e:
    st.error("⚠️ Some required modules are not importable in Streamlit runtime.")
    st.text(traceback.format_exc())

# Try to load .env if exists
if ENV_FILE.exists():
    try:
        load_dotenv(str(ENV_FILE))
        st.info("✅ .env file loaded successfully.")
    except Exception as e:
        st.error(f"⚠️ Failed to load .env: {e}")
        st.text(traceback.format_exc())
else:
    st.warning("⚠️ .env file missing — skipping environment variable load.")

# ---------------- STEP 3: RETRIEVAL EXECUTION ----------------
st.header("🧠 Retrieval Script Execution Test")

st.markdown(f"**Hardcoded Test Query:** `{TEST_QUERY}`")

def run_retrieval_test(script_path: Path):
    """Run a retrieval script and return stdout/stderr using same interpreter as Streamlit."""
    if not script_path.exists():
        st.error(f"❌ Script not found: {script_path}")
        return

    st.write(f"▶️ Running `{script_path.name}` using `{sys.executable}` ...")

    try:
        cmd = [
            sys.executable,  # ✅ Use same interpreter as Streamlit
            str(script_path),
            "--query", TEST_QUERY,
            "--top_k", "3"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90, cwd=str(BASE_DIR))

        st.markdown("**🔹 STDOUT:**")
        st.text_area(f"stdout_{script_path.name}", proc.stdout or "No STDOUT output", height=200)

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


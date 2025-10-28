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

Retrieval runner — runs two retrieval scripts with a hardcoded query,
prints step-by-step progress, stdout/stderr, exit codes, attempts to parse JSON output,
and shows your OpenAI key (masked by default, optional reveal).

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

RETRIEVAL_COMBINED = BASE_DIR / "Retrival" / "retrieval_combined_v2.py"   # text + tables (combined)
RETRIEVAL_TABLES = BASE_DIR / "Retrival" / "retrival_tables.py"          # tables-only

HARD_CODED_QUERY = (
    "Hardcoded test query: Summarize the key financial items and any numeric figures present in the documents."
)

SUBPROCESS_TIMEOUT = 180  # seconds

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Retrieval Runner — Hardcoded Query + Secrets", page_icon="🧪", layout="wide")
st.title("Retrieval Runner — Hardcoded Query + Secrets")
st.markdown("This page will run the two retrieval scripts (combined = text+tables, and tables-only) using the same Python interpreter as Streamlit and print everything that happens. It will also display the OpenAI key (masked by default).")

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
    # Fetch OpenAI key from multiple places
    openai_env = os.getenv("OPENAI_API_KEY") or ""
    openai_secret_top = ""
    try:
        # Try common placements in st.secrets
        if "OPENAI_API_KEY" in st.secrets:
            openai_secret_top = st.secrets["OPENAI_API_KEY"]
        elif "openai" in st.secrets and isinstance(st.secrets["openai"], dict):
            openai_secret_top = st.secrets["openai"].get("api_key", "")
    except Exception:
        # st.secrets may not be available in certain contexts
        openai_secret_top = ""

    # Decide which key to show (env takes precedence)
    openai_key_effective = openai_env or openai_secret_top or ""

    def mask_key(k: str) -> str:
        if not k:
            return "<NOT SET>"
        if len(k) <= 8:
            return k[0:1] + "*"*(len(k)-2) + k[-1:]
        return k[:4] + "*"*(max(6, len(k)-8)) + k[-4:]

    st.subheader("OpenAI Key (Sources)")
    st.write("Effective key is chosen from (in order): `env OPENAI_API_KEY`, `st.secrets['OPENAI_API_KEY']`, `st.secrets['openai']['api_key']`")

    st.markdown(f"- `env OPENAI_API_KEY`: `{bool(openai_env)}`")
    st.markdown(f"- `st.secrets OPENAI_API_KEY`: `{bool(openai_secret_top and 'OPENAI_API_KEY' in st.secrets)}`")
    st.markdown(f"- `st.secrets.openai.api_key`: `{bool(openai_secret_top and 'openai' in st.secrets)}`")

    st.markdown("**Effective OpenAI key (masked):**")
    st.code(mask_key(openai_key_effective))

    reveal_key = st.checkbox("Reveal raw OpenAI key in logs (only enable if you understand the risk)", value=False)

    st.markdown("---")
    run_button = st.button("Run retrieval scripts (hardcoded query)")

# Right pane: logs and detailed outputs
with col_output:
    st.subheader("Step-by-step log")
    log_box = st.empty()
    detail_expander = st.expander("Detailed outputs (stdout / stderr / parsed JSON)", expanded=True)
    stdout_area = detail_expander.empty()
    stderr_area = detail_expander.empty()
    parsed_area = detail_expander.empty()

# small helper to timestamp logs
logs = []
def add_log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    logs.append(f"[{ts}] {msg}")
    log_box.code("\n".join(logs), language="text")

def check_packages():
    """Return a small dict marking presence of key packages"""
    pkgs = {}
    for pkg in ("numpy", "sentence_transformers", "faiss", "faiss_cpu", "faiss-cpu"):
        try:
            spec = importlib.util.find_spec(pkg)
            pkgs[pkg] = bool(spec)
        except Exception:
            pkgs[pkg] = False
    return pkgs

def run_script_and_capture(script_path: Path, query: str) -> dict:
    """
    Run the given script via the same python executable used by Streamlit.
    Returns dict with keys: exists, cmd, returncode, stdout, stderr, parsed_json, error
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

    # Use same python interpreter as Streamlit
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
        pkg_status = check_packages()
        add_log("Package availability: " + ", ".join([f"{k}:{'Y' if v else 'N'}" for k,v in pkg_status.items()]))
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
    add_log(f"Combined retrieval script: {RETRIEVAL_COMBINED}")
    add_log(f"Tables retrieval script: {RETRIEVAL_TABLES}")

    # Print effective OpenAI key info
    add_log("OpenAI key sources check:")
    add_log(f"- env OPENAI_API_KEY present: {bool(openai_env)}")
    add_log(f"- st.secrets OPENAI_API_KEY present: {bool(openai_secret_top and 'OPENAI_API_KEY' in st.secrets)}")
    add_log(f"- st.secrets.openai.api_key present: {bool(openai_secret_top and 'openai' in st.secrets)}")
    add_log(f"Masked effective key: {mask_key(openai_key_effective)}")
    if reveal_key:
        add_log(f"RAW effective OpenAI key (revealed by user): {openai_key_effective}")
    else:
        add_log("Raw OpenAI key not revealed (use the checkbox to reveal).")

    add_log("=== Running combined (text+tables) retrieval ===")
    res_combined = run_script_and_capture(RETRIEVAL_COMBINED, HARD_CODED_QUERY)
    add_log(f"Combined retrieval finished. Exit code: {res_combined.get('returncode')}, error: {res_combined.get('error')}")

    add_log("=== Running tables-only retrieval ===")
    res_tables = run_script_and_capture(RETRIEVAL_TABLES, HARD_CODED_QUERY)
    add_log(f"Tables retrieval finished. Exit code: {res_tables.get('returncode')}, error: {res_tables.get('error')}")

    # Show brief parsed summaries if present
    def summarize_parsed(parsed):
        if not parsed:
            return "No parsed JSON."
        # show counts if structure matches earlier expected format
        sem_count = len(parsed.get("semantic", {}).get("results", [])) if isinstance(parsed.get("semantic", {}), dict) else "?"
        kw_count  = len(parsed.get("keyword", {}).get("results", [])) if isinstance(parsed.get("keyword", {}), dict) else "?"
        return f"Parsed JSON — semantic.results: {sem_count}, keyword.results: {kw_count}"

    add_log("Summary of parsed outputs:")
    add_log(f"- Combined parsed: {summarize_parsed(res_combined.get('parsed_json'))}")
    add_log(f"- Tables parsed:   {summarize_parsed(res_tables.get('parsed_json'))}")

    add_log("=== RUN COMPLETE ===")
    st.success("Run complete — scroll through the Step-by-step log and the Detailed output panels for full stdout/stderr/JSON.")
else:
    st.info("Click 'Run retrieval scripts (hardcoded query)' to execute both retrieval scripts and show all outputs. Use the checkbox to reveal your OpenAI key if you want to display it raw.")

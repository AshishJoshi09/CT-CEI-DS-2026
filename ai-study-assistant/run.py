"""
Convenience launcher: runs the Streamlit app via `python run.py`
instead of typing the full `streamlit run streamlit_app.py` command.
"""
from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

"""
Hedge Fund AI — Environment Setup
Run once: python setup.py
After that, just double-click activate.bat to start working
"""

import subprocess
import sys
from pathlib import Path

# Hardcoded so it doesn't matter where this file gets run from
PROJECT_DIR = Path(r"D:\ary fund")
VENV_DIR = PROJECT_DIR / "hedgefund_ai"

def run(cmd, **kwargs):
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, **kwargs, capture_output=False)
    if result.returncode != 0:
        print(f"\n  ERROR: Command failed. Check the output above.")
        sys.exit(1)
    return result

def main():
    print("=" * 55)
    print("  Hedge Fund AI — Environment Setup")
    print("=" * 55)

    if not PROJECT_DIR.exists():
        print(f"\n  ERROR: {PROJECT_DIR} does not exist.")
        sys.exit(1)

    # Step 1: Create venv
    print("\n[1/5] Creating virtual environment...")
    if VENV_DIR.exists():
        print("  Virtual environment already exists, skipping.")
    else:
        run(f'python -m venv "{VENV_DIR}"')
        print("  Done.")

    # Step 2: Paths to pip and python inside the venv (robust quoting)
    pip_exe = VENV_DIR / "Scripts" / "pip.exe"
    pip = f'"{pip_exe.absolute()}"'  # Use absolute path for reliability
    python_exe = VENV_DIR / "Scripts" / "python.exe"

    # Step 2.5: Upgrade pip FIRST (fixes ModuleNotFoundError in subprocesses)
    print("\n[2/5] Upgrading pip...")
    run(f'{pip} install --upgrade pip --quiet')

    # Step 3: Install all packages
    print("\n[3/5] Installing packages (this takes 5-10 min)...")

    packages = [
        # PyTorch with CUDA 12.1
        ("torch torchvision torchaudio",
         "--index-url https://download.pytorch.org/whl/cu121"),

        # ML libraries
        ("transformers accelerate bitsandbytes peft trl datasets", ""),

        # Finance & data
        ("yfinance sec-edgar-downloader fredapi python-dotenv", ""),

        # Quant models
        ("hmmlearn arch scipy statsmodels scikit-learn", ""),

        # UI & visualization
        ("plotly streamlit pandas numpy", ""),

        # Infrastructure
        ("ollama fastapi uvicorn sqlite-utils requests beautifulsoup4 sqlalchemy", ""),

        # Testing
        ("pytest pytest-mock", ""),
    ]

    for pkg, flags in packages:
        pkg_name = pkg.split()[0]
        print(f"\n  Installing: {pkg_name}{'...' if len(pkg.split()) > 1 else ''}")
        run(f'{pip} install {pkg} {flags}')

    # Step 4: Install the project itself in editable mode (now works post-pip upgrade)
    print(f"\n[4/5] Installing hedgefund-ai in editable mode...")
    run(f'{pip} install -e "{PROJECT_DIR.absolute()}"')

    print("\nAll packages installed.")

    # Step 5: Create activate.bat for future sessions
    print("\n[5/5] Creating activate.bat for quick startup...")

    bat_content = f"""@echo off
echo.
echo  Hedge Fund AI — Activating environment...
cd /d "{PROJECT_DIR.absolute()}"
call "{VENV_DIR.absolute()}\\Scripts\\activate.bat"
echo  Done. You are now in (hedgefund_ai) at %CD%
echo  Run: python main.py
echo.
cmd /k
"""
    bat_path = PROJECT_DIR / "activate.bat"
    bat_path.write_text(bat_content, encoding='utf-8')

    print("\n" + "=" * 55)
    print("  Setup complete!")
    print("=" * 55)
    print(f"""
 NEXT STEPS:
 1. Double-click activate.bat any time you want to work
 2. Run: ollama serve       (in a separate terminal)
 3. Run: ollama pull phi3:3.8b
 4. Run: python main.py

 Your environment lives at:
 {VENV_DIR.absolute()}

 Verify CUDA: python -c "import torch; print('CUDA:', torch.cuda.is_available())"
""")

if __name__ == "__main__":
    main()
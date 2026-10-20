"""Download the public dataset through official GitHub endpoints.

Keeps TLS verification enabled and validates the ZIP before replacing the file.
Does not install or change Python packages.
"""
import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data" / "armors.zip"
REPO = "baiyeweiguang/Number-Classifier-for-RoboMaster"


def valid_zip(path):
    if not path.exists() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and any(
                n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
                for n in z.namelist()
            )
    except (OSError, zipfile.BadZipFile):
        return False


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if valid_zip(TARGET):
        print("Dataset already downloaded and verified:", TARGET)
        return
    candidates = [
        (f"https://raw.githubusercontent.com/{REPO}/main/dataset/armors.zip", False),
        (f"https://github.com/{REPO}/raw/refs/heads/main/dataset/armors.zip", False),
        (f"https://codeload.github.com/{REPO}/zip/refs/heads/main", True),
    ]
    with tempfile.TemporaryDirectory(prefix="armor-download-", dir=TARGET.parent) as tmp:
        tmp = Path(tmp)
        for url, is_repository in candidates:
            print("Trying official source:", url, flush=True)
            archive = tmp / "download.zip"
            result = subprocess.run([
                "curl", "--fail", "--location", "--retry", "1",
                "--connect-timeout", "12", "--max-time", "120",
                "--output", str(archive), url,
            ])
            if result.returncode != 0:
                continue
            candidate = archive
            if is_repository:
                if not zipfile.is_zipfile(archive):
                    continue
                with zipfile.ZipFile(archive) as z:
                    matches = [n for n in z.namelist() if n.endswith("/dataset/armors.zip")]
                    if len(matches) != 1:
                        continue
                    candidate = tmp / "armors.zip"
                    with z.open(matches[0]) as src, candidate.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            if valid_zip(candidate):
                candidate.replace(TARGET)
                print("DATASET READY:", TARGET)
                print("SHA256:", hashlib.sha256(TARGET.read_bytes()).hexdigest())
                return
            print("Downloaded content was not a valid armor dataset ZIP.", flush=True)
    raise SystemExit(
        "Download failed. Python packages are already installed. "
        "Please share this output; do not disable certificate verification."
    )


if __name__ == "__main__":
    main()

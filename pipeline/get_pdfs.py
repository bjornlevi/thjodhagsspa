# pipeline/get_pdfs.py
import csv
import re
import time
import unicodedata
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

CSV_PATH = Path("data/sources/thjodhagsspa_list.csv")
OUT_DIR = Path("data/spa_pdf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- settings ---
PER_FILE_DELAY_SEC = 0.5   # polite pause between files
MAX_REDIRECTS = 5

def slugify(text: str) -> str:
    mapping = str.maketrans({
        "þ": "th", "Þ": "th",
        "ð": "d",  "Ð": "d",
        "æ": "ae", "Æ": "ae",
        "ö": "o",  "Ö": "o",
        "–": "-", "—": "-",
    })
    text = text.translate(mapping)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")

def normalize_url(url: str) -> str:
    url = url.strip()
    # Fix common copy/paste issues
    url = url.replace(" ", "")
    # Remove accidental trailing chars like a final ')'
    url = re.sub(r"[)\]]+$", "", url)
    # Collapse accidental double .pdf extension
    url = re.sub(r"\.pdf(\.pdf)+$", ".pdf", url, flags=re.IGNORECASE)
    # Ensure scheme exists
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url

def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=7,                 # total retry budget
        connect=7,               # retry on DNS/connection errors
        read=7,
        backoff_factor=0.8,      # exponential backoff: 0.8, 1.6, 3.2, ...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": "thjodhagsspa-downloader/1.0 (+https://example.org)",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    })
    s.max_redirects = MAX_REDIRECTS
    return s

session = make_session()

with CSV_PATH.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        year = row["ar"].strip()
        utgafa = row["utgafa"].strip()
        url_raw = row["url"].strip()
        url = normalize_url(url_raw)

        fname = f"spa_{year}_{slugify(utgafa)}.pdf"
        out_path = OUT_DIR / fname

        if out_path.exists():
            print(f"✓ Already exists: {fname}")
            continue

        print(f"↓ Downloading {fname} …")
        try:
            # manual retry loop on top (covers DNS errors that can occur before adapter kicks in)
            attempts = 0
            while True:
                attempts += 1
                try:
                    with session.get(url, stream=True, timeout=60, allow_redirects=True) as r:
                        r.raise_for_status()
                        ctype = r.headers.get("Content-Type", "")
                        if "pdf" not in ctype.lower():
                            print(f"  ! Warning: unexpected content-type '{ctype}' from {url}")

                        with open(out_path, "wb") as fp:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    fp.write(chunk)
                    break  # success
                except requests.exceptions.RequestException as e:
                    if attempts >= 3:
                        raise
                    sleep_for = 1.5 * attempts
                    print(f"  … transient error ({e}). Retrying in {sleep_for:.1f}s")
                    time.sleep(sleep_for)

            print(f"  Saved → {out_path}")
        except Exception as e:
            print(f"✗ Failed to download {url} : {e}")

        # polite pause between files
        time.sleep(PER_FILE_DELAY_SEC)

print("✅ All downloads attempted. Check data/spa_pdf/")

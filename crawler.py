import os
import re
import time
import logging
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pathlib import Path

# ==============================
# Configuration
# ==============================
#Checkpoint
CHECKPOINT_FILE = Path("last_page.txt")
# Total number of pages (confirmed: 164)
TOTAL_PAGES = 8
ENTRIES_PER_PAGE = 50

# Base URL pattern for listing pages
BASE_URL_TEMPLATE = "https://www.supremecourt.gov.bd/web/index.php?page=judgments.php&menu=00&div_id=1&start={start}"

# Local directory to save PDFs
DOWNLOAD_DIR = Path("sc_judgments_Appellate Division")

# Request settings
REQUEST_DELAY = 2          # seconds between page requests
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30       # seconds

# Invalid filename characters (Windows + Unix safe)
INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'

# ==============================
# Logging Setup
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_log.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# Helper Functions
# ==============================

def sanitize_filename(name: str) -> str:
    """Remove or replace invalid filename characters and clean spacing."""
    sanitized = re.sub(INVALID_FILENAME_CHARS, '_', name)
    sanitized = re.sub(r'[_\s]+', '_', sanitized)  # collapse underscores/spaces
    sanitized = sanitized.strip('_ ')
    return sanitized if sanitized else "unnamed_document"

def extract_title_from_url(url: str) -> str:
    """
    Extract meaningful title from PDF URL and clean it.
    Example:
        Input: .../297027_Writ_petition6347of2010discharged_as_being_infructuous_dt.20.08.2025.pdf
        Output: Writ_petition_6347_of_2010_discharged_as_being_infructuous.pdf
    """
    filename = os.path.basename(urlparse(url).path)
    
    # Remove leading numeric ID (e.g., "297027_")
    title = re.sub(r'^\d+_', '', filename)
    
    # Insert underscores around "of" between numbers: "6347of2010" → "6347_of_2010"
    title = re.sub(r'(\d+)of(\d{4})', r'\1_of_\2', title, flags=re.IGNORECASE)
    
    # Standardize "discharged" and similar terms
    title = re.sub(r'discharged_as_being_infructuous', 'discharged_as_being_infructuous', title, flags=re.IGNORECASE)
    
    # Remove date suffix like "_dt.20.08.2025" or "_dt_20_08_2025"
    title = re.sub(r'_dt[._]\d{2}[._]\d{2}[._]\d{4}', '', title, flags=re.IGNORECASE)
    
    # Ensure .pdf extension
    if not title.lower().endswith('.pdf'):
        title += '.pdf'
    
    return sanitize_filename(title)

def download_file(url: str, filepath: Path) -> bool:
    """Download file with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Downloading: {url} (Attempt {attempt}/{MAX_RETRIES})")
            response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"✅ Saved: {filepath.name}")
            return True
            
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed for {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)
            else:
                logger.error(f"❌ Failed after {MAX_RETRIES} attempts: {url}")
                return False

def extract_document_links_from_page(page_url: str) -> list:
    """Parse HTML and extract all PDF links under /resources/documents/."""
    try:
        logger.info(f"Fetching page: {page_url}")
        response = requests.get(page_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if '/resources/documents/' in href and href.lower().endswith('.pdf'):
                full_url = urljoin(page_url, href)
                links.append(full_url)
        
        logger.info(f"📄 Found {len(links)} document(s) on this page.")
        return links
        
    except Exception as e:
        logger.error(f"Error parsing page {page_url}: {e}")
        return []

    def get_last_completed_page():
        if CHECKPOINT_FILE.exists():
            try:
                return int(CHECKPOINT_FILE.read_text().strip())
            except:
                pass
        return 0  # means start from page 1

# ==============================
# Main Function
# ==============================

def main():
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    # Read checkpoint: last fully completed page
    checkpoint_file = Path("last_page.txt")
    if checkpoint_file.exists():
        try:
            last_done = int(checkpoint_file.read_text().strip())
            start_page = last_done + 1
            logger.info(f"▶ Resuming from page {start_page} (last completed: {last_done})")
        except Exception as e:
            logger.warning(f"Invalid checkpoint file; starting from page 1. Error: {e}")
            start_page = 1
    else:
        start_page = 1
        logger.info("▶ Starting from page 1 (no checkpoint found)")

    if start_page > TOTAL_PAGES:
        logger.info("✅ All pages already completed!")
        return

    logger.info(
        f"Starting download from page {start_page} to {TOTAL_PAGES} (total ~{(TOTAL_PAGES - start_page + 1) * ENTRIES_PER_PAGE} documents)")

    for page_num in range(start_page, TOTAL_PAGES + 1):
        start_value = (page_num - 1) * ENTRIES_PER_PAGE
        page_url = BASE_URL_TEMPLATE.format(start=start_value)

        logger.info(f"➡️ Processing page {page_num}/{TOTAL_PAGES} (start={start_value})")

        doc_links = extract_document_links_from_page(page_url)

        for doc_url in doc_links:
            try:
                new_filename = extract_title_from_url(doc_url)
                filepath = DOWNLOAD_DIR / new_filename

                if filepath.exists():
                    logger.info(f"⏭️ Skipping (already exists): {new_filename}")
                    continue

                download_file(doc_url, filepath)

            except Exception as e:
                logger.exception(f"💥 Unexpected error processing {doc_url}: {e}")

        # ✅ CRITICAL: Save checkpoint AFTER finishing all downloads on this page
        checkpoint_file.write_text(str(page_num))

        # Delay before next page (except after last page)
        if page_num < TOTAL_PAGES:
            logger.info(f"⏳ Waiting {REQUEST_DELAY} seconds before next page...")
            time.sleep(REQUEST_DELAY)

    # Optional: clean up checkpoint when fully done
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("🧹 Checkpoint file removed — all pages completed.")

    logger.info("🎉 All pages processed. Download complete!")

# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Script interrupted by user.")
    except Exception as e:
        logger.exception(f"🔥 Critical error: {e}")
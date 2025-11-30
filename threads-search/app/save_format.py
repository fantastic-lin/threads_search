import os
import json
import csv
from typing import List, Dict, Optional, Any

import pandas as pd

# Optional Google Sheets support
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSHEETS = True
except Exception:
    HAS_GSHEETS = False


def _stable_keys(rows: List[Dict[str, Any]]) -> List[str]:
    """
    Build a stable, first-seen union of keys across all rows.
    This preserves a consistent column order in outputs.
    """
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def export_csv(rows: List[Dict[str, Any]], path: str) -> Optional[str]:
    """
    Write a list of dict rows to CSV file at `path`.
    Returns the path on success, or None if `rows` is empty.
    """
    if not rows:
        return None

    fieldnames = _stable_keys(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


def export_xlsx(rows: List[Dict[str, Any]], path: str) -> Optional[str]:
    """
    Write a list of dict rows to XLSX file at `path`.
    Returns the path on success, or None if `rows` is empty.
    """
    if not rows:
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    return path


def export_json(rows: List[Dict[str, Any]], path: str) -> Optional[str]:
    """
    Write a list of dict rows to JSON file at `path`.
    Returns the path on success, or None if `rows` is empty.
    """
    if not rows:
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return path


def export_to_gsheets(
    rows: List[Dict[str, Any]],
    stamp: str,
    chunk_size: int = 5000
) -> Optional[str]:
    """
    Write a list of dict rows to a Google Sheet.
    - Spreadsheet title: GOOGLE_SHEETS_DOC_NAME (default: "Threads_Exports")
    - Worksheet title:   GOOGLE_SHEETS_WORKSHEET (default: f"keyword_export_{stamp}")
    - Service account JSON file path: GOOGLE_SERVICE_ACCOUNT_JSON

    Returns a synthetic handle like "gsheets://<doc>/<worksheet>" on success,
    or None if rows are empty or Google Sheets libs/credentials are missing.
    """
    if not rows:
        return None
    if not HAS_GSHEETS:
        return None

    cred_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    doc_name = os.getenv("GOOGLE_SHEETS_DOC_NAME", "Threads_Exports")
    worksheet_title = os.getenv("GOOGLE_SHEETS_WORKSHEET", f"keyword_export_{stamp}")

    if not cred_path or not os.path.exists(cred_path):
        return None

    # Column order
    all_keys: List[str] = _stable_keys(rows)

    # Auth
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(cred_path, scopes=scopes)
    gc = gspread.authorize(creds)

    # Open or create spreadsheet
    try:
        sh = gc.open(doc_name)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(doc_name)

    # Open or create worksheet
    try:
        ws = sh.worksheet(worksheet_title)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=worksheet_title,
            rows="1",
            cols=max(1, len(all_keys))
        )

    # Write header
    ws.update([all_keys], value_input_option="RAW")

    # Convert rows to list-of-lists in stable column order
    def row_to_list(r: Dict[str, Any]) -> List[Any]:
        return [r.get(k, "") for k in all_keys]

    data_rows = [row_to_list(r) for r in rows]

    # Batch update in chunks
    start_row = 2  # Row 1 is header
    for i in range(0, len(data_rows), chunk_size):
        chunk = data_rows[i:i + chunk_size]
        end_row = start_row + len(chunk) - 1
        end_col = len(all_keys)

        start_cell = gspread.utils.rowcol_to_a1(start_row, 1)
        end_cell = gspread.utils.rowcol_to_a1(end_row, end_col)
        rng = f"{start_cell}:{end_cell}"

        ws.update(rng, chunk, value_input_option="RAW")
        start_row = end_row + 1

    return f"gsheets://{doc_name}/{worksheet_title}"

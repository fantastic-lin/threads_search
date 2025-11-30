import yagmail
import sys
import traceback
import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_HOST = "smtp.gmail.com"   # Adapt if you use a different provider
SMTP_PORT = 465                # SSL

def _join(items: List[str]) -> str:
    """
    Joins a list of strings into a single string, separated by commas (with Chinese punctuation).
    Filters out any empty or whitespace-only strings.
    """
    items = [str(x).strip() for x in (items or []) if str(x).strip()]
    return ",".join(items)

def build_description(meta: Dict[str, Any]) -> str:
    """
    Builds an email body description based on the provided meta data.

    meta expects:
      - mode: 'keyword' | 'account' | 'mixed'
      - keywords: List of keywords to search for
      - users: List of account usernames to search for
      - key_word_count: Number of articles found based on keywords
      - account_count: Number of articles found based on account usernames
      - date_range: A list with [start, end] date range (optional)
    """
    mode = (meta.get("mode") or "").lower()
    kws = meta.get("keywords") or []
    accts = meta.get("users") or []
    key_word_cnt = int(meta.get("key_word_count") or 0)
    account_cnt = int(meta.get("account_count") or 0)

    # Determine the introduction based on the mode
    if mode == "keyword" and kws and not accts:
        body_intro = f"Based on the keywords ({_join(kws)}), {key_word_cnt} articles were fetched."
    elif mode == "account" and accts and not kws:
        body_intro = f"Based on the accounts ({_join(accts)}), {account_cnt} articles were fetched."
    elif kws and accts:
        body_intro = f"Based on the keywords ({_join(kws)}) and accounts ({_join(accts)}), {key_word_cnt + account_cnt} articles were fetched."
    else:
        body_intro = "No results were fetched."

    ending = "Please find the complete data in the attachments."
    parts = ["Results of the crawl:", "", body_intro + ending]
    return "\n".join(parts)

def send_email(to_email: List[str], content: str, attachments: List[str]):
    """
    Sends an email with the specified content and attachments.

    Arguments:
    - to_email: List of recipient email addresses
    - content: Body of the email
    - attachments: List of file paths to attach to the email
    """
    try:
        yag = yagmail.SMTP(SMTP_USER, SMTP_PASS, host=SMTP_HOST, port=SMTP_PORT, smtp_ssl=True)
        yag.send(
            to=to_email,
            subject="Threads Data",
            contents=content,
            attachments=attachments  # can be a list or a single path
        )
        print("Email sent to:", to_email)
    except Exception:
        print("Failed to send email.")
        print(traceback.format_exc())
        sys.exit(1)

# Integrate with your CLI runner (after you compute `res` & `p`)
def maybe_send(to_list: List[str], dictory, main_file_name, tf_file_name, meta):
    """
    Determines whether to send an email based on the existence of attachments and sends it.

    Arguments:
    - to_list: List of recipient email addresses
    - dictory: Directory containing the files to attach
    - main_file_name: Main file to attach
    - tf_file_name: Term frequency file (optional, attached only if exists)
    - meta: Metadata for constructing the email content
    """
    attachments: List[str] = []
    if main_file_name: 
    # 1) Main export file
        main_file_path = os.path.join(dictory, main_file_name)
        if os.path.isfile(main_file_path):
            attachments.append(main_file_path)

    # 2) Term frequency file (only if term frequency is enabled and the file exists)
        tf_file_path = os.path.join(dictory, tf_file_name)
        if  os.path.isfile(tf_file_path):
            attachments.append(tf_file_path)
        
        content = build_description(meta)
    # 3) if main_file_name is None, return
    else: 
        print("No available attachments. Email will not be sent.")
        return

    # 4) Send the email with the attachments
    send_email(to_email=to_list, content=content, attachments=attachments)

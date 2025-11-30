import os
import sys
import json
import time
import csv
import re
import unicodedata
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple
from playwright.sync_api import sync_playwright

from dotenv import load_dotenv
load_dotenv()

import requests
import emoji
import pandas as pd
from bs4 import BeautifulSoup
from dateutil.parser import isoparse, parse as dateparse
from urllib.parse import (
    urlparse, urlencode, urlunparse, parse_qsl
)
from flask import Flask, request, redirect, url_for, session

# Local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import save_format
from app.termfreq import term_freq_from_rows
from app.likes import get_counts_from_feed_card, parse_threads_url
from app.sender import maybe_send

# Configuration
APP_ID = os.getenv("API_ID")
APP_SECRET = os.getenv("API_SECRET")
REDIRECT_BASE = os.getenv("REDIRECT_URI")

AUTHZ = "https://threads.net/oauth/authorize"
GRAPH = "https://graph.threads.net/v1.0"
REFRESH_URL = "https://graph.threads.net/v1.0/refresh_access_token"

TOKENS_FILE = ".tokens.json"  # local long-lived token cache

app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET", "dev")


# Token management
def _load_tokens() -> dict:
    """Load cached token payload from disk (if any)."""
    if not os.path.exists(TOKENS_FILE):
        return {}
    try:
        with open(TOKENS_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _save_tokens(token_dict: dict) -> None:
    """Persist token payload to disk."""
    with open(TOKENS_FILE, "w", encoding="utf-8") as f:
        json.dump(token_dict, f, ensure_ascii=False, indent=2)

def _now_utc() -> datetime:
    """Current UTC time."""
    return datetime.now(timezone.utc)

def _store_long_lived(access_token: str, expires_in: int) -> None:
    """Persist a new long-lived access token with its expiry."""
    exp = _now_utc() + timedelta(seconds=int(expires_in))
    _save_tokens({"access_token": access_token, "expires_at": exp.isoformat()})

def _exchange_long_lived(current_token: str) -> dict:
    """Exchange a short-lived token for a long-lived token."""
    r = requests.get(
        f"{GRAPH}/access_token",
        params={
            "grant_type": "th_exchange_token",
            "client_secret": APP_SECRET,
            "access_token": current_token,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def _refresh_long_lived(long_token: str) -> dict:
    """Refresh a long-lived token."""
    r = requests.get(
        REFRESH_URL,
        params={
            "grant_type": "th_refresh_token",
            "access_token": long_token,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def _get_access_token() -> str | None:
    """
    Return a valid access token.
    - If near/at expiry, attempt proactive refresh.
    - If expired and refresh fails, return None (caller should re-authorize).
    """
    t = _load_tokens()
    token = t.get("access_token")
    expires_at = t.get("expires_at")
    if not token or not expires_at:
        return None

    exp = datetime.fromisoformat(expires_at)
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)

    remaining = exp - _now_utc()

    # If already expired, try refresh. If it fails, return None.
    if remaining <= timedelta(seconds=0):
        try:
            refreshed = _refresh_long_lived(token)
            new_token = refreshed["access_token"]
            ttl = int(refreshed.get("expires_in", 0))
            new_exp = _now_utc() + timedelta(seconds=ttl)
            _save_tokens({"access_token": new_token, "expires_at": new_exp.isoformat()})
            return new_token
        except Exception:
            return None

    # If expiring within 7 days, attempt refresh but keep using current token on failure.
    if remaining <= timedelta(days=7):
        try:
            refreshed = _refresh_long_lived(token)
            new_token = refreshed["access_token"]
            ttl = int(refreshed.get("expires_in", 0))
            new_exp = _now_utc() + timedelta(seconds=ttl)
            _save_tokens({"access_token": new_token, "expires_at": new_exp.isoformat()})
            return new_token
        except Exception:
            return token

    return token

# Graph API helper
def api_get(path: str, params: dict | None = None) -> dict:
    """GET a Threads Graph API endpoint with the current access token."""
    token = _get_access_token()
    if not token:
        raise RuntimeError("NO_TOKEN")

    params = dict(params or {})
    params["access_token"] = token

    r = requests.get(f"{GRAPH}{path}", params=params, timeout=30)
    if not r.ok:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Graph API error {r.status_code} on {path}: {err}")
    return r.json()

# Text/URL normalization utilities
def strip_html(text: str) -> str:
    """Remove HTML markup and collapse text."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def normalize_text(text: str, emoji_format: str = "emoji") -> str:
    """NFKC normalize, handle emojis, remove zero-width chars, trim."""
    t = unicodedata.normalize("NFKC", text or "")
    t = emoji.emojize(t, language="alias") if emoji_format == "emoji" else emoji.demojize(t)
    t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
    return t.strip()

def is_valid_url(url: str) -> bool:
    """Basic URL validation: scheme http/https and non-empty netloc."""
    try:
        p = urlparse(url or "")
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")

def canonical_url(u: str) -> str:
    """Normalize permalink for dedupe: strip fragment, clean trackers, lowercase host."""
    if not u:
        return ""
    p = urlparse(u)
    q = [
        (k, v)
        for (k, v) in parse_qsl(p.query, keep_blank_values=True)
        if not k.lower().startswith(TRACKING_PREFIXES)
    ]
    clean = p._replace(
        scheme=p.scheme.lower(),
        netloc=p.netloc.lower(),
        path=re.sub(r"/+$", "", p.path),
        params="",
        query=urlencode(q, doseq=True),
        fragment="",
    )
    return urlunparse(clean)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize_regex(text: str) -> list[str]:
    return WORD_RE.findall(text) if text else []

def count_words_regex(text: str) -> int:
    return len(tokenize_regex(text))

def slugify_name(s: str) -> str:
    """Filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (s or "").strip()).strip("_")

# History IO / dedup
def build_store_path_for_kw(kw: str, out_dir: str = "data/keyword_store", ext: str = "json") -> Path:
    """
    Return cumulative storage path for a single keyword:
      data/keyword_store/kw_<kw>.json
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return Path(out_dir) / f"kw_{slugify_name(kw)}.{ext}"


def build_store_path_for_user(username: str, out_dir: str = "data/user_store", ext: str = "json") -> Path:
    """
    Return cumulative storage path for a single user:
      data/user_store/user_<handle>.json
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    uname = slugify_name(username.lstrip("@"))
    return Path(out_dir) / f"user_posts_{uname}.{ext}"

def load_history(path: Path) -> Tuple[pd.DataFrame | None, set]:
    """Load JSON history file; return (DataFrame or None, seen_url_set)."""
    if not path.exists():
        return None, set()
    df = pd.read_json(path)
    seen_urls = set()
    if "url" in df.columns and df["url"].notna().any():
        seen_urls = set(canonical_url(u) for u in df["url"].dropna().tolist())
    return df, seen_urls

def load_profile_history(path: str) -> list[dict]:
    """Load profile history from a JSON file, or return empty list."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Empty file → treat as no history
        if not content:
            return set()

        data = json.loads(content)
        if isinstance(data, list):
            return set(data)
        else:
            return set()
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable file → treat as no history
        return set()
    
def merge_and_save(history_df: pd.DataFrame | None, new_rows: list[dict], path: Path) -> pd.DataFrame:
    """
    Merge new rows into history, dedupe by URL, sort by timestamp, and write JSON.
    """
    new_df = pd.DataFrame(new_rows)
    out_df = new_df if history_df is None else pd.concat([history_df, new_df], ignore_index=True)

    if "timestamp" in out_df.columns:
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], errors="coerce", utc=True)

    if "url" in out_df.columns and out_df["url"].notna().any():
        out_df = out_df.sort_values(by="timestamp", ascending=True)
        out_df = out_df.drop_duplicates(subset=["url"], keep="first")

    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_json(path, orient="records", force_ascii=False)
    return out_df

def save_profile_history(names: set[str], json_path: str):
    """Save profile history to JSON (for reloading) and CSV (for analysis)."""
    # JSON for persistence

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(names)), f, ensure_ascii=False, indent=2)

def save_profile_if_new(profile: dict, profiles_dir: str):
    """
    Save a profile dict into profiles/{username}.json.
    If the file already exists, do nothing.
    """
    os.makedirs(profiles_dir, exist_ok=True)

    username = profile.get("username")
    if not username:
        return  # cannot save without username

    path = Path(profiles_dir) / f"{username}.json"

    # Skip if exists
    if path.exists():
        return

    # Save new profile
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

# OAuth redirect helper
def redirect_uri() -> str:
    """Construct redirect URI for OAuth flow."""
    if REDIRECT_BASE:
        return f"{REDIRECT_BASE}{url_for('callback')}"
    return url_for("callback", _external=True, _scheme="https")

# Flask routes (OAuth + admin)
@app.route("/")
def login():
    """Start OAuth flow: build and redirect to authorization URL."""
    scope = "threads_basic,threads_keyword_search,threads_profile_discovery"
    state = secrets.token_urlsafe(24)
    session["oauth_state"] = state

    params = dict(
        client_id=APP_ID,
        redirect_uri=redirect_uri(),
        response_type="code",
        scope=scope,
        state=state,
    )
    final_url = AUTHZ + "?" + urlencode(params)
    print("Generated login URL:", final_url)
    return redirect(final_url)

@app.route("/admin/refresh", methods=["POST", "GET"])
def admin_refresh():
    """Force-refresh long-lived token manually."""
    t = _load_tokens()
    token = t.get("access_token")
    if not token:
        return {"ok": False, "error": "NO_TOKEN"}, 400
    try:
        j = _refresh_long_lived(token)
        new_token = j["access_token"]
        ttl = int(j.get("expires_in", 0))
        new_exp = _now_utc() + timedelta(seconds=ttl)
        _save_tokens({"access_token": new_token, "expires_at": new_exp.isoformat()})
        return {"ok": True, "expires_at": new_exp.isoformat()}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

@app.route("/callback")
def callback():
    """OAuth callback: exchange code → short-lived → long-lived token."""
    code = request.args.get("code")
    returned_state = request.args.get("state")
    expected_state = session.pop("oauth_state", None)

    if not code or not returned_state or returned_state != expected_state:
        return {
            "ok": False,
            "error": "Invalid or missing code/state",
            "expected_state": expected_state,
            "returned_state": returned_state,
        }, 400

    # Prevent reuse of the same code
    if session.get("used_code") == code:
        return redirect(url_for("done"))

    # Exchange authorization code -> short-lived token
    r = requests.post(
        f"{GRAPH}/oauth/access_token",
        data={
            "client_id": APP_ID,
            "client_secret": APP_SECRET,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri(),
            "code": code,
        },
        timeout=30,
    )
    r.raise_for_status()
    short_token = r.json()["access_token"]
    session["used_code"] = code

    # Exchange short-lived -> long-lived token
    ll = _exchange_long_lived(short_token)
    _store_long_lived(ll["access_token"], ll.get("expires_in", 5184000))
    return redirect(url_for("done"), code=303)

@app.route("/done")
def done():
    return "Authorization successful. ✅ You can close this page now."

@app.route("/admin/token")
def show_token():
    """Return token status (no secret values)."""
    t = _load_tokens()
    if not t:
        return {"ok": False, "error": "NO_TOKEN"}, 404
    return {"ok": True, "expires_at": t.get("expires_at")}

@app.route("/healthz")
def healthz():
    """Liveness: show days remaining for token expiry."""
    t = _load_tokens()
    exp = t.get("expires_at")
    days_left = None
    if exp:
        try:
            days_left = (datetime.fromisoformat(exp) - _now_utc()).total_seconds() / 86400.0
        except Exception:
            pass
    return {
        "has_token": bool(t.get("access_token")),
        "expires_at": exp,
        "days_left": round(days_left, 2) if days_left is not None else None,
    }

# Keyword search + export + optional term frequency
def run_keyword_export(
    keywords,
    max_posts=50,                 # global cap or per-keyword cap (based on mode)
    limit_mode="total",           # "per" | "total"
    search_type="TOP",            # "TOP" | "RECENT"
    search_mode="KEYWORD",        # "KEYWORD" | "TAG"
    media_type="TEXT",            # "TEXT" | "IMAGE" | "VIDEO" | "ALL"
    since=None,
    until=None,
    out="csv",                    # "csv" | "xlsx" | "json" | "gsheets" | "preview"
    clean=True,
    time_format="local",          # "local" | "utc"
    emoji_format="emoji",         # "emoji" | "alias"
    persist_ext="json",
    keyword_store_dir="data/keyword_store",
    export_dir="data/keyword_store",
    output_basename=None,
    termfreq_mode=True,           # compute term frequency
    termfreq_scope="history",     # "new" | "history"
    termfreq_topn=50,
    termfreq_outdir="data/keyword_store",
    send_email=True,
    send_email_to=None
):
    total_cap_reached = False
    all_rows = []
    per_kw_summary = []
    history_total_all_keywords = 0
    

    for kw in keywords:
        if limit_mode == "total" and len(all_rows) >= max_posts:
            total_cap_reached = True
            break

        store_path = build_store_path_for_kw(kw, out_dir=keyword_store_dir, ext=persist_ext)
        history_df, seen_urls = load_history(store_path)

        history_total_before = len(history_df) if history_df is not None else 0
        rows_this_kw = []
        collected, after = [], None
        kw_cap = max_posts if limit_mode == "per" else max(0, max_posts - len(all_rows))

        while len(collected) < kw_cap:
            params = {
                "q": kw,
                "search_type": search_type,
                "search_mode": search_mode,
                "media_type": media_type,
                "fields": "id,text,media_type,permalink,timestamp,owner{username}",
                "limit": max_posts,
            }
            if since is not None:
                params["since"] = since
            if until is not None:
                params["until"] = until
            if after:
                params["after"] = after

            data = api_get("/keyword_search", params)
            items = data.get("data", []) or []
            if not items:
                break

            collected.extend(items)
            after = (data.get("paging") or {}).get("cursors", {}).get("after")
            if not after:
                break

        for m in collected[:kw_cap]:
            owner = (m.get("owner") or {})
            text_raw = m.get("text") or ""
            text_clean = strip_html(text_raw) if clean else text_raw
            text_norm = normalize_text(text_clean, emoji_format) if clean else text_clean
            url = m.get("permalink")
            url_nor = canonical_url(url)
            ts = standardize_ts(m.get("timestamp"), time_format)

            if not is_valid_url(url_nor):
                continue
            if url_nor in seen_urls:
                continue

            time.sleep(0.1)  # gentle pacing
            row = {
                "keyword": kw,
                "id": m.get("id"),
                "text": text_norm,
                "text_count": count_words_regex(text_norm),
                "url": url_nor,
                "author": ("@" + owner.get("username")) if owner.get("username") else "",
                "author_url": f"https://www.threads.net/@{owner.get('username')}" if owner.get("username") else "",
                "media_type": m.get("media_type"),
                "timestamp": ts,
            }
            seen_urls.add(url_nor)
            rows_this_kw.append(row)

        if limit_mode == "total" and len(all_rows) >= max_posts:
            total_cap_reached = True
            break

        out_df = merge_and_save(history_df, rows_this_kw, store_path)
        history_rows_this_kw = out_df.to_dict(orient="records") if isinstance(out_df, pd.DataFrame) else []
        history_total_after = len(out_df)
        had_history = history_total_after > 0 or history_total_before > 0
        history_total_all_keywords += history_total_after

        per_kw_summary.append({
            "keyword": kw,
            "added_this_run": len(rows_this_kw),
            "history_total_before": history_total_before,
            "history_total_after": history_total_after,
            "had_history": had_history,
            "store_path": str(store_path),
            "_history_rows_for_tf": history_rows_this_kw,
        })
        all_rows.extend(rows_this_kw)

    result = {
        "count": len(all_rows),
        "history_total_all_keywords": history_total_all_keywords,
        "keywords": keywords,
        "limit": max_posts,
        "limitMode": limit_mode,
        "out": out,
        "truncated_by_total_cap": total_cap_reached,
        "summary": per_kw_summary,
        "rows": all_rows,
    }

    # Export rows
    if out in ("csv", "xlsx", "json", "gsheets", "preview"):
        os.makedirs(export_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = (
            output_basename
            or (
                f"threads_keyword_{keywords[0].replace(' ', '_')}_{stamp}"
                if len(keywords) == 1
                else f"threads_keyword_{keywords[0].replace(' ', '_')}_plus{len(keywords) - 1}_{stamp}"
            )
        )
        base_path = os.path.join(export_dir, base)
        if all_rows:
            filter_rows = [d for d in all_rows if d.get("media_type") != "REPOST_FACADE"]
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                for row in filter_rows:
                    url=row['url']
                    try:
                        user, post_id = parse_threads_url(url)
                        counts = get_counts_from_feed_card(page, user=user, post_id=post_id)
                        row.update(counts)
                    except Exception as e:
                        counts_error={'like': None, 'reply': None, 'repost': None}
                        row.update(counts_error)
                browser.close()
   
            if out == "csv":
                result["file"] = save_format.export_csv(filter_rows, base_path + ".csv")
            elif out == "xlsx":
                result["file"] = save_format.export_xlsx(filter_rows, base_path + ".xlsx")
            elif out == "json":
                result["file"] = save_format.export_json(filter_rows, base_path + ".json")
            elif out == "gsheets":
                result["sheet"] = save_format.export_to_gsheets(filter_rows, stamp)
            elif out == "preview":
                result["preview"] = filter_rows[:5]
        else:
            # No new posts — show alert instead of creating empty CSV
            result["warning"] = "No new posts found for the given users."
            result["file"] = None

    # Optional term frequency (global across keywords)
    if termfreq_mode:
        os.makedirs(termfreq_outdir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        def _filter_rows_for_tf(rows):
            out_rows = []
            for r in rows:
                out_rows.append(r)
            return out_rows

        all_hist_rows = []
        for s in per_kw_summary:
            all_hist_rows.extend(s.get("_history_rows_for_tf", []))

        base_global = all_hist_rows or all_rows
        rows_for_tf_global = _filter_rows_for_tf(base_global)
        tf_global = term_freq_from_rows(rows_for_tf_global, text_field="text", topn=termfreq_topn)

        safe_keywords = [slugify_name(kw) for kw in keywords]
        keyword_part = (
            "_".join(safe_keywords[:3]) + f"_plus{len(safe_keywords) - 3}"
            if len(safe_keywords) > 3
            else "_".join(safe_keywords)
        )

        global_base = f"termfrequency_{keyword_part}_{termfreq_scope}"
        global_json = os.path.join(termfreq_outdir, global_base + ".json")

        try:
            save_format.export_json(tf_global, global_json)
        except Exception:
            with open(global_json, "w", encoding="utf-8") as f:
                json.dump(tf_global, f, ensure_ascii=False, indent=2)

        result["term_freq_file_name"] = global_base + '.json'
        result["tf_global"] = tf_global

    if send_email and send_email_to:
        meta={"mode": "keyword","keywords": result['keywords'], "users":None, "key_word_count": len(result['keywords']), "account_count": None}
        
        print("========keyword_store_dir", keyword_store_dir, "result[file]",result["file"])
        maybe_send(send_email_to, keyword_store_dir, result["file"], result["term_freq_file_name"],meta)
    return result

# Username fetch + export + optional term frequency
def get_profile_basic(handle: str):
    """Lookup a profile by handle; return (username, raw_response_or_error)."""
    handle = handle.lstrip("@").strip().lower()
    try:
        data = api_get("/profile_lookup", {"username": handle})
    except Exception as e:
        return None, {"username": handle, "error": f"{e}"}
    pid = data.get("username")  # API shape can vary; adjust if needed
    return pid, data

def standardize_ts(ts: str, time_format: str) -> str:
    """Convert Threads ISO8601 to local/UTC ISO8601."""
    if not ts:
        return ""
    try:
        dt = isoparse(ts)
    except Exception:
        dt = dateparse(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz=None if time_format == "local" else timezone.utc).isoformat()

def run_user_posts(
    handles,
    max_posts=5,
    per_max=2,
    out="csv",
    clean=True,
    time_format="local",
    emoji_format="emoji",
    sleep_s=0.15,
    user_store_dir="data/user_store",
    persist_ext="json",
    export_dir="data/user_store",
    output_basename=None,
    detail_sleep_s=0.1,
    termfreq_mode=True,
    termfreq_scope="history",    # "new" | "history"
    termfreq_topn=50,
    termfreq_outdir="data/user_store",
    send_email=True,
    send_email_to=None,
):
    """Fetch posts from one or more users and optionally compute term frequency."""
    if per_max is None:
        per_max = max_posts

    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(user_store_dir, exist_ok=True)
    all_rows, profiles, per_user_summary = [], [], []
    profile_history_json = os.path.join(user_store_dir, "profiles_history.json")
    profile_history_names = load_profile_history(profile_history_json)

    for handle in handles:
        norm_handle = handle.lstrip("@").lower()
        uname, profile_basic = get_profile_basic(norm_handle)
        

        if not uname:
            profiles.append({"username": norm_handle, "error": "Profile not found"})
            per_user_summary.append({
                "username": norm_handle,
                "added_this_run": 0,
                "total_after_merge": 0,
                "store_path": str(build_store_path_for_user(norm_handle, out_dir=user_store_dir, ext=persist_ext)),
            })
            continue

        profiles.append(profile_basic)
        profile_dir = os.path.join(user_store_dir, "profiles")
        save_profile_if_new(profile_basic, profile_dir)
        profile_history_names.add(profile_basic["username"])

        store_path = build_store_path_for_user(uname, out_dir=user_store_dir, ext=persist_ext)
        history_df, seen_urls = load_history(store_path)
        seen_urls = set(seen_urls or [])

        collected, after = [], None
        pulled = 0

        # 1) List posts for the user
        while pulled < per_max and len(all_rows) < max_posts:
            params = {
                "username": uname,
                "fields": "id,username,permalink,text,owner,media_type,timestamp",
                "limit": min(50, per_max - pulled),
            }
            if after:
                params["after"] = after

            j = api_get("/profile_posts", params)
            items = j.get("data", []) or []
            if not items:
                break

            collected.extend(items)
            pulled += len(items)

            if pulled >= per_max or len(all_rows) >= max_posts:
                break

            after = (j.get("paging") or {}).get("cursors", {}).get("after")
            if not after:
                break

        # 2) Build rows (optionally pace between detail calls)
        rows_this_user = []
        for m in collected[:per_max]:
            text_raw = m.get("text") or ""
            text_clean = strip_html(text_raw) if clean else text_raw
            text_norm = normalize_text(text_clean, emoji_format) if clean else text_clean

            url = m.get("permalink")
            url_nor = canonical_url(url)

            if not is_valid_url(url_nor):
                continue
            if url_nor in seen_urls:
                continue

            if detail_sleep_s:
                time.sleep(detail_sleep_s)

            ts = standardize_ts(m.get("timestamp"), time_format)

            row = {
                "source_username": "@" + uname,
                "id": m.get("id"),
                "text": text_norm,
                "url": url_nor,
                "media_type": m.get("media_type"),
                "timestamp": ts,
            }
            rows_this_user.append(row)
            seen_urls.add(url_nor)

            if len(all_rows) + len(rows_this_user) >= max_posts:
                break

        # 3) Merge with history
        out_df_user = merge_and_save(history_df, rows_this_user, store_path)
        per_user_summary.append({
            "username": uname,
            "added_this_run": len(rows_this_user),
            "total_after_merge": len(out_df_user),
            "store_path": str(store_path),
            "_history_rows_for_tf": out_df_user.to_dict(orient="records") if hasattr(out_df_user, "to_dict") else [],
        })
        all_rows.extend(rows_this_user)

        if sleep_s:
            time.sleep(sleep_s)
        if len(all_rows) >= max_posts:
            break

    result = {
        "users": handles,
        "profiles": profiles,
        "count": len(all_rows),
        "limit_total": max_posts,
        "limit_per_user": per_max,
        "out": out,
        "per_user_summary": per_user_summary,
        "rows": all_rows,
    }
    save_profile_history(
        profile_history_names,
        json_path=profile_history_json
    )

    # Export rows
    if out in ("csv", "xlsx", "json", "gsheets", "preview"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = output_basename or f"threads_users_{'_'.join([h.lstrip('@') for h in handles[:3]])}{'_more' if len(handles) > 3 else ''}_{stamp}"
        base_path = os.path.join(user_store_dir, base)

        if all_rows:
            filter_rows = [d for d in all_rows if d.get("media_type") != "REPOST_FACADE"]
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                for row in filter_rows:
                    url=row['url']
                    try:
                        user, post_id = parse_threads_url(url)
                        counts = get_counts_from_feed_card(page, user=user, post_id=post_id)
                        row.update(counts)
                    except Exception as e:
                        counts_error={'like': None, 'reply': None, 'repost': None}
                        row.update(counts_error)
                browser.close()
            if out == "csv":
                result["file"] = save_format.export_csv(filter_rows, base_path + ".csv")
            elif out == "xlsx":
                result["file"] = save_format.export_xlsx(filter_rows, base_path + ".xlsx")
            elif out == "json":
                result["file"] = save_format.export_json(filter_rows, base_path + ".json")
            elif out == "gsheets":
                result["sheet"] = save_format.export_to_gsheets(filter_rows, stamp)
            elif out == "preview":
                result["preview"] = filter_rows[:5]
        else:
            # No new posts — show alert instead of creating empty CSV
            result["warning"] = "No new posts found for the given users."
            result["file"] = None

    # Optional term frequency (global across users)
    if termfreq_mode:
        os.makedirs(termfreq_outdir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        def _filter_rows_for_tf(rows):
            out_rows = []
            for r in rows:
                out_rows.append(r)
            return out_rows

        all_hist_rows = []
        for s in per_user_summary:
            all_hist_rows.extend(s.get("_history_rows_for_tf", []))

        base_global = all_hist_rows or all_rows
        rows_for_tf_global = _filter_rows_for_tf(base_global)
        tf_global = term_freq_from_rows(rows_for_tf_global, text_field="text", topn=termfreq_topn)

        safe_users = [slugify_name(h) for h in handles]
        user_part = (
            "_".join(safe_users[:3]) + f"_plus{len(safe_users) - 3}"
            if len(safe_users) > 3
            else "_".join(safe_users)
        )

        global_base = f"termfrequency_{user_part}_{termfreq_scope}"
        global_json = os.path.join(termfreq_outdir, global_base + ".json")

        try:
            save_format.export_json(tf_global, global_json)
        except Exception:
            with open(global_json, "w", encoding="utf-8") as f:
                json.dump(tf_global, f, ensure_ascii=False, indent=2)

        result["term_freq_file_name"] = global_base
        result["tf_global"] = tf_global
    if send_email and send_email_to:
        meta={"mode": "keyword","keywords": None, "users": result["users"], "key_word_count": None, "account_count": len(result["users"])}
        maybe_send(send_email_to, user_store_dir, result["file"], result["term_freq_file_name"],meta)
    return result

# Entrypoint (Flask server for OAuth)
if __name__ == "__main__":
    # Expose a small admin/OAuth surface; the data exports are callable functions
    app.run(host="0.0.0.0", port=8443)
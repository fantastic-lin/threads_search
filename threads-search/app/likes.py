import re
from typing import Dict, List, Tuple

from playwright.sync_api import sync_playwright, Page


#  Utility Functions

def parse_count(text: str) -> int:
    """
    Convert strings like '161', '1.2K', '3,456', '1.1M' into integers.
    Returns 0 for invalid/empty strings.
    """
    if not text:
        return 0

    t = text.strip().upper()
    t = t.replace(",", "").replace(" ", "")

    if not t:
        return 0

    multiplier = 1
    if t.endswith("K"):
        multiplier = 1_000
        t = t[:-1]
    elif t.endswith("M"):
        multiplier = 1_000_000
        t = t[:-1]
    elif t.endswith("B"):
        multiplier = 1_000_000_000
        t = t[:-1]

    try:
        val = float(t)
    except ValueError:
        return 0

    return int(val * multiplier)


def last_non_empty_span(scope) -> int:
    """
    scope: Playwright Locator

    Inside `scope`, find the last span that looks like a numeric count
    (digits with optional K/M/B suffix). If none are found, fall back to
    the last non-empty span and try to parse a number from it.
    Returns 0 if nothing valid is found.
    """
    spans = scope.locator("span")

    # 1) Prefer spans that look like pure number / K / M / B
    filtered = spans.filter(
        has_text=re.compile(r"^\s*\d+(?:[.,]\d+)?\s*[KkMmBb]?\s*$")
    )
    if filtered.count() > 0:
        txt = filtered.last.inner_text()
        return parse_count(txt)

    # 2) Fallback: scan from the end and use the last non-empty span
    n = spans.count()
    for i in range(n - 1, -1, -1):
        txt = spans.nth(i).inner_text().strip()
        if txt:
            return parse_count(txt)

    return 0


def count_for_labels_in_card(card, labels: List[str]) -> int:
    """
    For a given card (data-pressable-container="true"):

    Find the button that contains svg[aria-label=<label>] and then
    extract the numeric count from spans inside that button.

    labels: e.g. ["Like", "Unlike"], ["Comment"], ["Repost"]
    """
    for label in labels:
        svgs = card.locator(f'svg[aria-label="{label}"]')
        if svgs.count() == 0:
            continue

        svg = svgs.first

        # Prefer the closest ancestor with role="button"
        wrapper = svg.locator('xpath=ancestor::*[@role="button"][1]')
        if wrapper.count() == 0:
            # Some structures may not have a button, fall back to nearest parent
            wrapper = svg.locator('xpath=ancestor::*[1]')

        wrapper = wrapper.first
        try:
            wrapper.wait_for(state="attached", timeout=5000)
        except Exception:
            pass

        return last_non_empty_span(wrapper)

    # No label was found
    return 0


def get_card_for_post(page: Page, user: str, post_id: str):
    """
    Find all [data-pressable-container="true"] cards,
    then select the one containing an anchor with
    href*="/@user/post/post_id".
    """
    card_selector = (
        f'[data-pressable-container="true"]:has(a[href*="/@{user}/post/{post_id}"])'
    )
    cards = page.locator(card_selector)
    n = cards.count()
    print(f"[DEBUG] cards found for {user}/{post_id}: {n}")

    if n == 0:
        return None

    card = cards.first
    try:
        card.wait_for(state="attached", timeout=5000)
    except Exception as e:
        print(f"[WARN] card.wait_for failed for {user}/{post_id}: {e}")

    return card


#  Main Logic: Extract like / comment / repost from card

def get_counts_from_feed_card(page: Page, user: str, post_id: str) -> Dict[str, int]:
    """
    Open the post detail page and:

    1. Find the [data-pressable-container="true"] card that contains
       a[href*="/@user/post/post_id"].
    2. Within this card, look for:
       - svg[aria-label="Like"] / svg[aria-label="Unlike"]
       - svg[aria-label="Comment"]
       - svg[aria-label="Repost"]
    3. For each svg, find its button wrapper and extract the numeric
       count from the spans inside.

    Returns a dict: {"like": int, "comment": int, "repost": int}
    """
    url = f"https://www.threads.net/@{user}/post/{post_id}"
    print(f"\n[INFO] Loading {url}")
    page.goto(url, wait_until="load", timeout=60000)
    print("[DEBUG] Final URL:", page.url)

    # Wait for at least one card to appear
    page.wait_for_selector('[data-pressable-container="true"]', timeout=60000)

    card = get_card_for_post(page, user, post_id)
    if card is None:
        print(f"[WARN] No card found for {user}/{post_id}, returning 0s.")
        return {"like": 0, "comment": 0, "repost": 0}

    like = count_for_labels_in_card(card, ["Like", "Unlike"])
    comment = count_for_labels_in_card(card, ["Comment"]) 
    repost = count_for_labels_in_card(card, ["Repost"])

    print(f"[RESULT] {user}/{post_id} -> like={like}, comment={comment}, repost={repost}")
    return {"like": like, "comment": comment, "repost": repost}


#  URL Parsing Helpers (threads.com)
def parse_threads_url(url: str) -> Tuple[str, str]:
    """
    Supported formats:
        https://www.threads.net/@user/post/POSTID
        https://www.threads.com/@user/post/POSTID

    Returns (user, post_id). Raises ValueError on failure.
    """
    m = re.search(r'threads\.(?:net|com)/@([^/]+)/post/([^/?#]+)', url)
    if not m:
        raise ValueError(f"Cannot parse Threads URL: {url}")
    return m.group(1), m.group(2)
import os
import re
from collections import Counter
from typing import List, Dict, Iterable

# English tokenization
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
EN_STOP = set(
    """
a an the and or but if in on at for of to from with without into out by as is are was were be been
this that these those it its he she they them we you i my your our their not no yes do does did
i me my mine myself we our ours ourselves you your yours yourself yourselves he him his himself
she her hers herself it its itself they them their theirs themselves
to too very can could should would will just than then so such
""".split()
)

def tokenize_en(text: str) -> Iterable[str]:
    """Yield lowercased English tokens (letters/digits + apostrophe), excluding stopwords and 1-char tokens."""
    for t in WORD_RE.findall(text or ""):
        t = t.lower()
        if t and t not in EN_STOP and len(t) > 1:
            yield t


# CJK/Chinese tokenization
CJK_RE = re.compile(r"[\u3400-\u9FFF]")
CN_PUNCT_RE = re.compile(r"[\u3000-\u303F，。！？、；：“”‘’（）《》…—\-\s]+")

try:
    import jieba  # type: ignore
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False

def load_stopwords_file(path: str) -> set:
    """Load UTF-8 stopwords (one per line). Returns an empty set if the file is missing."""
    sw = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    sw.add(w)
    except FileNotFoundError:
        pass
    return sw

# Allow overriding the stopword path via environment variable
DEFAULT_STOPWORD_PATH = "/Users/fantastic-lin/Documents/part-time_job/threads-data/utils/hit_stopwords.txt"
STOPWORD_PATH = os.getenv("ZH_STOPWORDS_PATH", DEFAULT_STOPWORD_PATH)
ZH_STOP = load_stopwords_file(STOPWORD_PATH)

def _extract_cjk(text: str) -> str:
    """Keep only CJK chars and whitespace, remove Chinese punctuation, collapse spaces."""
    t = "".join(ch if CJK_RE.match(ch) else " " for ch in (text or ""))
    t = CN_PUNCT_RE.sub(" ", t)
    return " ".join(t.split())

def tokenize_zh(text: str) -> Iterable[str]:
    """
    Yield Chinese tokens.
    - Prefer jieba if installed.
    - Fallback: split on contiguous CJK blocks after stripping punctuation.
    - Filter by stopwords and length >= 2.
    """
    if _HAS_JIEBA:
        for tok in jieba.lcut(text or ""):
            tok = tok.strip()
            if not tok or tok in ZH_STOP:
                continue
            if CJK_RE.search(tok) and len(tok) >= 2:
                yield tok
    else:
        t = _extract_cjk(text)
        for tok in t.split():
            if tok and tok not in ZH_STOP and len(tok) >= 2:
                yield tok


# Language detection (simple heuristic)
def has_cjk(text: str) -> bool:
    """Return True if the text contains any CJK character."""
    return bool(CJK_RE.search(text or ""))

def detect_lang(rows: List[Dict], text_field: str = "text") -> str:
    """
    Rough language detector for a batch of rows.
    - 'zh' if >= 60% rows contain CJK
    - 'mixed' if 20%–60%
    - 'en' otherwise
    """
    if not rows:
        return "en"
    cjk_cnt = sum(1 for r in rows if has_cjk(str(r.get(text_field, ""))))
    ratio = cjk_cnt / max(len(rows), 1)
    if ratio >= 0.6:
        return "zh"
    if 0.2 <= ratio < 0.6:
        return "mixed"
    return "en"


# Main: simple term frequency
def term_freq_from_rows(
    rows: List[Dict],
    text_field: str = "text",
    lang: str = "auto",   # "auto" | "en" | "zh"
    topn: int = 50
) -> List[Dict]:
    """
    Compute term frequencies from a list of row dicts.
    - Tokenizes as English, Chinese, or both (if 'mixed' or lang='auto').
    - Returns top-N [{"term": str, "count": int}] sorted by count desc.
    """
    if not rows:
        return []

    chosen = detect_lang(rows, text_field=text_field) if lang == "auto" else lang

    c = Counter()
    if chosen == "en":
        for r in rows:
            for tok in tokenize_en(str(r.get(text_field, ""))):
                c.update([tok])
    elif chosen == "zh":
        for r in rows:
            for tok in tokenize_zh(str(r.get(text_field, ""))):
                c.update([tok])
    else:  # "mixed" — run both tokenizers and merge
        for r in rows:
            txt = str(r.get(text_field, ""))
            for tok in tokenize_zh(txt):
                c.update([tok])
            for tok in tokenize_en(txt):
                c.update([tok])

    return [{"term": w, "count": n} for w, n in c.most_common(topn)]

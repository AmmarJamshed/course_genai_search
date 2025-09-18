# app.py
# Coursemon — GENAI-style Search (No APIs, One-Box UI)
# ----------------------------------------------------
# • Single search bar, clean results list (title, snippet, URL)
# • No external LLMs/APIs: rule-based "GENAI" query rewrite
# • Expands user prompt into several smart queries:
#     - baseline
#     - skills synonyms + learning-intent terms (course|training|bootcamp|class|certificate)
#     - site-pack targeting edu/training domains (site:coursera.org OR …)
#     - location/price/mode/duration hints normalized
#     - phrase + intitle boosters
# • Searches Bing + DuckDuckGo + Yahoo (public HTML) → merge → dedupe → rerank
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Notes:
# - HTML selectors may change; this is best-effort scraping.
# - For production, move to a crawler index + partner feeds/APIs.

import re
import html
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode

import streamlit as st
import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_NAME = "Coursemon — GENAI Search (No API)"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

# -------------------- helpers --------------------

def sess() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    return s

def clean(x: Optional[str]) -> str:
    if not x: return ""
    return re.sub(r"\s+", " ", html.unescape(x)).strip()

def dedupe(results: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in results:
        u = (r.get("url") or "").split("#")[0]
        if not u or u in seen: 
            continue
        seen.add(u)
        out.append(r)
    return out

# -------------------- engines (HTML) --------------------

def search_bing(q: str, num: int, s: requests.Session) -> List[Dict]:
    """Scrape Bing HTML."""
    try:
        url = "https://www.bing.com/search?" + urlencode({"q": q, "count": min(num, 20)})
        r = s.get(url, timeout=10)
        if not r.ok: return []
        soup = BeautifulSoup(r.text, "lxml")
        out = []
        for li in soup.select("li.b_algo"):
            a = li.select_one("h2 a")
            if not a or not a.get("href"): 
                continue
            title = clean(a.get_text())
            link = a["href"]
            sn = li.select_one("p, div.b_caption p")
            snippet = clean(sn.get_text() if sn else "")
            out.append({"title": title, "snippet": snippet, "url": link, "engine": "bing"})
            if len(out) >= num: break
        return out
    except Exception:
        return []

def search_ddg(q: str, num: int, s: requests.Session) -> List[Dict]:
    """Scrape DuckDuckGo HTML."""
    try:
        r = s.get("https://html.duckduckgo.com/html/", params={"q": q}, timeout=10)
        if not r.ok: return []
        soup = BeautifulSoup(r.text, "lxml")
        out = []
        for res in soup.select(".result"):
            a = res.select_one("a.result__a, a.result__url")
            if not a or not a.get("href") or not a["href"].startswith("http"):
                continue
            title = clean(a.get_text())
            sn = res.select_one(".result__snippet")
            snippet = clean(sn.get_text() if sn else "")
            out.append({"title": title, "snippet": snippet, "url": a["href"], "engine": "ddg"})
            if len(out) >= num: break
        # fallback selector sweep
        if not out:
            for a in soup.select("a.result__a, a.result__url"):
                href = a.get("href")
                if href and href.startswith("http"):
                    out.append({"title": clean(a.get_text()), "snippet": "", "url": href, "engine": "ddg"})
                    if len(out) >= num: break
        return out
    except Exception:
        return []

def search_yahoo(q: str, num: int, s: requests.Session) -> List[Dict]:
    """Scrape Yahoo HTML."""
    try:
        url = "https://search.yahoo.com/search?" + urlencode({"p": q})
        r = s.get(url, timeout=10)
        if not r.ok: return []
        soup = BeautifulSoup(r.text, "lxml")
        out = []
        for d in soup.select("div#web h3.title a, div.dd.algo.algo-sr h3 a"):
            href = d.get("href")
            if not href or not href.startswith("http"):
                continue
            title = clean(d.get_text())
            cont = d.find_parent("div", class_=re.compile("algo"))
            sn = cont.select_one("div.compText, p") if cont else None
            snippet = clean(sn.get_text() if sn else "")
            out.append({"title": title, "snippet": snippet, "url": href, "engine": "yahoo"})
            if len(out) >= num: break
        return out
    except Exception:
        return []

# -------------------- "GENAI" query rewrite (rule-based) --------------------

SITE_PACK = [
    "site:coursera.org", "site:edx.org", "site:udacity.com", "site:udemy.com",
    "site:classcentral.com", "site:datacamp.com", "site:generalassemb.ly",
    "site:simplilearn.com", "site:brainstation.io", "site:lewagon.com",
    "site:linkedin.com/learning", "site:futurelearn.com",
    "site:.edu", "site:.ac.uk", "site:.edu.pk", "site:.ac.pk"
]

SKILL_SYNONYMS = {
    "data science": ["data science", "data analytics", "machine learning", "ai", "ml", "python data"],
    "python": ["python", "pandas", "numpy", "data wrangling"],
    "cybersecurity": ["cybersecurity", "information security", "soc", "siem"],
    "mba": ["mba", "business administration", "management", "executive mba"],
}

LEARNING_INTENT = ["course", "training", "bootcamp", "certificate", "class", "program", "workshop"]

LOCATIONS = ["karachi","lahore","islamabad","pakistan","uk","london","usa","new york","germany","berlin","uae","dubai","india","mumbai"]

def parse_hints(text: str) -> Dict:
    t = text.lower()
    hints = {"topic": None, "loc": None, "budget": None, "mode": None, "duration": None}
    # location
    for loc in LOCATIONS:
        if re.search(rf"\b{re.escape(loc)}\b", t):
            hints["loc"] = loc
            break
    # budget
    m = re.search(r"(?:(?:under|below|less than|<=)\s*)?([$€£]|rs\.?|pkr)?\s?(\d{2,6})", t)
    if m:
        # keep raw number; engines handle currency words
        hints["budget"] = m.group(2)
    # mode
    if re.search(r"\bonline|remote|live online\b", t): hints["mode"] = "online"
    elif re.search(r"\bonsite|in[- ]person|campus\b", t): hints["mode"] = "onsite"
    # duration
    d = re.search(r"\b(\d+\s*(hours?|days?|weeks?|months?))\b|evening|weekend|short|part[- ]time|full[- ]time", t)
    if d: hints["duration"] = d.group(0)
    # topic (remove obvious hints)
    topic = re.sub(r"under|below|less than|<=|\d{2,6}|online|onsite|in[- ]person|evening|weekend|short|part[- ]time|full[- ]time", " ", t)
    for loc in LOCATIONS: topic = re.sub(rf"\b{re.escape(loc)}\b", " ", topic)
    hints["topic"] = clean(topic)
    return hints

def topic_synonyms(topic: str) -> List[str]:
    out = [topic]
    for k, vs in SKILL_SYNONYMS.items():
        if k in topic:
            out.extend(vs)
    return list(dict.fromkeys(out))

def genai_expand(user_query: str) -> List[str]:
    """Produce several robust queries from a natural prompt."""
    hints = parse_hints(user_query)
    topic = hints["topic"] or user_query.lower()
    syns = topic_synonyms(topic)
    syn_str = " OR ".join(f"\"{s}\"" for s in syns)

    intent_str = " OR ".join(LEARNING_INTENT)

    # site pack
    site_str = " OR ".join(SITE_PACK)

    parts = []
    # baseline
    parts.append(f"({syn_str}) ({intent_str})")

    # with location/mode/duration
    extra = []
    if hints["loc"]: extra.append(hints["loc"])
    if hints["mode"]: extra.append(hints["mode"])
    if hints["duration"]: extra.append(hints["duration"])
    if hints["budget"]: extra.append(f"under {hints['budget']}")
    if extra:
        parts.append(f"({syn_str}) ({intent_str}) " + " ".join(extra))

    # with site pack
    parts.append(f"({syn_str}) ({intent_str}) ({site_str})")

    # phrase + intitle boosters
    key_phrase = " ".join(syns[:1] or [topic]).strip()
    parts.append(f"\"{key_phrase}\" intitle:({intent_str})")
    if hints["loc"]:
        parts.append(f"\"{key_phrase}\" {hints['loc']} intitle:({intent_str})")

    # price normalized with multiple currencies words
    if hints["budget"]:
        parts.append(f"({syn_str}) ({intent_str}) (under {hints['budget']} OR <= {hints['budget']} OR price {hints['budget']} OR PKR {hints['budget']} OR $ {hints['budget']})")

    # unique & non-empty
    uniq = []
    seen = set()
    for q in parts:
        q = clean(q)
        if q and q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq[:6]  # cap to keep it fast

# -------------------- ranking --------------------

def rerank(results: List[Dict], user_query: str) -> List[Tuple[Dict, float]]:
    texts = [f"{r.get('title','')} {r.get('snippet','')}" for r in results]
    if not texts: 
        return []
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
    X = vec.fit_transform(texts + [user_query])
    sims = cosine_similarity(X[:-1], X[-1]).ravel()
    ranked = sorted(zip(results, sims.tolist()), key=lambda x: x[1], reverse=True)
    return ranked

# -------------------- UI --------------------

st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide")
st.markdown(
    "<style>.card{padding:12px 16px;border:1px solid #eee;border-radius:12px;margin-bottom:12px}"
    ".t{font-weight:600;font-size:1.05rem}.u{color:#6b7280;font-size:.9rem}</style>",
    unsafe_allow_html=True
)
st.markdown("<h1>Coursemon</h1>", unsafe_allow_html=True)
st.caption("One-box GENAI-style search (no APIs). Type your need; I’ll expand it and search multiple engines.")

query = st.text_input("Search the web", value="evening data science bootcamp in Karachi under $300", label_visibility="collapsed")

st.markdown("---")

if query.strip():
    s = sess()
    expanded = genai_expand(query)

    # show what I'm actually searching (for transparency)
    with st.expander("🔎 How I rewrote your query"):
        for i, qv in enumerate(expanded, 1):
            st.code(f"{i}. {qv}")

    # collect results from multiple engines for each variant
    raw: List[Dict] = []
    for qv in expanded:
        raw += search_bing(qv, num=10, s=s)
        raw += search_ddg(qv, num=10, s=s)
        raw += search_yahoo(qv, num=10, s=s)

    merged = dedupe(raw)
    ranked = rerank(merged, query)

    if not ranked:
        st.warning("No results parsed. Try simplifying the text (e.g., 'python bootcamp Karachi').")
    else:
        st.write(f"**{len(ranked)} results**")
        for r, score in ranked[:50]:
            st.markdown(
                f'<div class="card">'
                f'<div class="t"><a href="{r["url"]}" target="_blank">{html.escape(r["title"] or r["url"])}</a></div>'
                f'<div class="u">{html.escape(r.get("url",""))}</div>'
                f'<div style="margin-top:6px;">{html.escape(r.get("snippet",""))}</div>'
                f'<div class="u" style="margin-top:6px;">score: {score:.2f} • {r.get("engine","")}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

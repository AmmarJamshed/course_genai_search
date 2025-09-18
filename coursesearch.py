# app.py
# Coursemon — GENAI Search (No API, One-Box)
# -------------------------------------------
# One search bar. Results list. No API keys.
# Engines: Bing, DuckDuckGo, Yahoo (public HTML scrape).
# Smart query rewrite focuses on courses/trainings anywhere on the web.

import re, html
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode

import streamlit as st
import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "Coursemon — GENAI Search (No API)"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# -------------------- HTTP session --------------------
def sess() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    return s

def clean(x: Optional[str]) -> str:
    if not x: return ""
    return re.sub(r"\s+", " ", html.unescape(x)).strip()

def dedupe(results: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in results:
        u = (r.get("url") or "").split("#")[0]
        if not u or u in seen: continue
        seen.add(u); out.append(r)
    return out

# -------------------- Engines (HTML scrape) --------------------
def search_bing(q: str, num: int, s: requests.Session) -> List[Dict]:
    try:
        url = "https://www.bing.com/search?" + urlencode({"q": q, "count": min(num, 20)})
        r = s.get(url, timeout=10)
        if not r.ok: return []
        soup = BeautifulSoup(r.text, "lxml")
        out = []
        for li in soup.select("li.b_algo"):
            a = li.select_one("h2 a")
            if not a or not a.get("href"): continue
            title = clean(a.get_text()); link = a["href"]
            sn = li.select_one("p, div.b_caption p")
            snippet = clean(sn.get_text() if sn else "")
            out.append({"title": title, "snippet": snippet, "url": link, "engine": "bing"})
            if len(out) >= num: break
        return out
    except Exception:
        return []

def search_ddg(q: str, num: int, s: requests.Session) -> List[Dict]:
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
    try:
        url = "https://search.yahoo.com/search?" + urlencode({"p": q})
        r = s.get(url, timeout=10)
        if not r.ok: return []
        soup = BeautifulSoup(r.text, "lxml")
        out = []
        for d in soup.select("div#web h3.title a, div.dd.algo.algo-sr h3 a"):
            href = d.get("href")
            if not href or not href.startswith("http"): continue
            title = clean(d.get_text())
            cont = d.find_parent("div", class_=re.compile("algo"))
            sn = cont.select_one("div.compText, p") if cont else None
            snippet = clean(sn.get_text() if sn else "")
            out.append({"title": title, "snippet": snippet, "url": href, "engine": "yahoo"})
            if len(out) >= num: break
        return out
    except Exception:
        return []

# -------------------- GENAI-style rewrite (rule-based) --------------------

# Learning signals we want in searches
LEARNING = [
    "course","training","bootcamp","certificate","diploma","short course",
    "MOOC","program","class","workshop","seminar","lecture",
    "curriculum","syllabus","catalog"
]

# Subject synonyms — broad coverage incl. humanities
SUBJECT_SYNONYMS = {
    "history": ["history","world history","ancient history","modern history","historical studies"],
    "philosophy": ["philosophy","ethics","logic","metaphysics","epistemology","philosophy of mind","aesthetics"],
    "humanities": ["humanities","liberal arts","classics","religious studies","anthropology","sociology","political philosophy"],
    "data science": ["data science","machine learning","ml","ai","data analytics","python data","statistics"],
    "python": ["python","pandas","numpy","data wrangling","programming in python"],
    "cybersecurity": ["cybersecurity","information security","infosec","soc","siem","network security"],
    "business": ["business","management","entrepreneurship","marketing","finance","accounting","operations"],
    "mba": ["mba","business administration","executive mba"],
    "math": ["mathematics","calculus","linear algebra","discrete math","probability","statistics"],
    "cs": ["computer science","algorithms","data structures","systems","software engineering"],
    "design": ["design","graphic design","ui ux","user experience","product design","visual design"],
    "english": ["english","literature","writing","composition","critical reading"],
    "economics": ["economics","microeconomics","macroeconomics","econometrics"],
    "law": ["law","legal studies","jurisprudence","international law","constitutional law"],
}

# Site packs
MOOC_PACK = [
    "site:coursera.org","site:edx.org","site:futurelearn.com","site:classcentral.com",
    "site:udacity.com","site:udemy.com","site:datacamp.com","site:skillshare.com",
    "site:linkedin.com/learning"
]
UNIV_PACK = [
    "site:.edu","site:.ac.uk","site:.edu.pk","site:.ac.pk","site:.ac.in","site:.edu.au",
    "site:openlearning.com","site:ocw.mit.edu","site:extension.harvard.edu","site:canvascatalog.com",
    "catalog","bulletin"
]
BOOTCAMP_PACK = [
    "site:generalassemb.ly","site:brainstation.io","site:lewagon.com","site:simplilearn.com",
    "site:thinkful.com","site:springboard.com"
]

LOCATIONS = ["karachi","lahore","islamabad","pakistan","uk","london","usa","new york","germany","berlin","uae","dubai","india","mumbai","canada","toronto","europe","asia"]

def parse_hints(text: str) -> Dict:
    t = text.lower()
    hints = {"topic": None, "loc": None, "budget": None, "mode": None, "duration": None}
    # location
    for loc in LOCATIONS:
        if re.search(rf"\b{re.escape(loc)}\b", t):
            hints["loc"] = loc; break
    # price
    m = re.search(r"(?:under|below|less than|<=)?\s*(?:rs\.?|pkr|\$|usd|eur|£)?\s*(\d{2,6})", t)
    if m: hints["budget"] = m.group(1)
    # mode
    if re.search(r"\bonline|remote|live online\b", t): hints["mode"] = "online"
    elif re.search(r"\b(on-?site|in[- ]person|campus)\b", t): hints["mode"] = "onsite"
    # duration
    d = re.search(r"\b(\d+\s*(hours?|days?|weeks?|months?))\b|evening|weekend|short|part[- ]time|full[- ]time", t)
    if d: hints["duration"] = d.group(0)
    # topic guess: strip common hint words
    topic = t
    topic = re.sub(r"(under|below|less than|<=|\d{2,6}|online|on-?site|in[- ]person|evening|weekend|short|part[- ]time|full[- ]time)", " ", topic)
    for loc in LOCATIONS: topic = re.sub(rf"\b{re.escape(loc)}\b", " ", topic)
    topic = clean(topic)
    hints["topic"] = topic if topic else text.lower()
    return hints

def subject_expand(topic: str) -> List[str]:
    topic = topic.strip()
    out = [topic]
    for k, vs in SUBJECT_SYNONYMS.items():
        if k in topic:
            out.extend(vs)
    # If two words joined by "and", split and expand each (e.g., "history and philosophy")
    if " and " in topic:
        for part in [p.strip() for p in topic.split(" and ") if p.strip()]:
            out.append(part)
            for k, vs in SUBJECT_SYNONYMS.items():
                if k in part:
                    out.extend(vs)
    return list(dict.fromkeys(out))

def genai_expand(user_query: str) -> List[str]:
    """Turn any topic prompt into robust course/training search queries."""
    hints = parse_hints(user_query)
    syns = subject_expand(hints["topic"])
    syn_str = " OR ".join(f"\"{s}\"" for s in syns[:6])  # cap to keep short
    learn = " OR ".join(LEARNING)

    site_mooc = " OR ".join(MOOC_PACK)
    site_univ = " OR ".join(UNIV_PACK)
    site_boot = " OR ".join(BOOTCAMP_PACK)

    # structural operators for course pages
    structural = "(intitle:course OR inurl:course OR inurl:courses OR inurl:program OR inurl:catalog OR inurl:syllabus)"

    variants = []

    # 1) Baseline: topic + learning signals
    variants.append(f"({syn_str}) ({learn})")

    # 2) Structural signals (forces course-like pages)
    variants.append(f"({syn_str}) {structural}")

    # 3) MOOC/marketplaces
    variants.append(f"({syn_str}) ({learn}) ({site_mooc})")

    # 4) University catalogs / official syllabi
    variants.append(f"({syn_str}) ({learn} OR syllabus OR curriculum) ({site_univ})")

    # 5) Bootcamps / vocational
    variants.append(f"({syn_str}) ({learn}) ({site_boot})")

    # 6) Syllabus PDFs (very academic)
    variants.append(f"({syn_str}) (syllabus OR curriculum) filetype:pdf")

    # 7) Add geo/mode/duration/price hints to one compact variant
    extras = []
    if hints["loc"]: extras.append(hints["loc"])
    if hints["mode"]: extras.append(hints["mode"])
    if hints["duration"]: extras.append(hints["duration"])
    if hints["budget"]: extras.append(f"under {hints['budget']}")
    if extras:
        variants.append(f"({syn_str}) ({learn}) {' '.join(extras)}")

    # unique & compact
    uniq, seen = [], set()
    for q in variants:
        q = clean(q)
        if q and q not in seen:
            seen.add(q); uniq.append(q)
    return uniq[:8]

# -------------------- Reranking --------------------
def rerank(results: List[Dict], user_query: str) -> List[Tuple[Dict, float]]:
    texts = [f"{r.get('title','')} {r.get('snippet','')}" for r in results]
    if not texts: return []
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
    unsafe_allow_html=True)
st.markdown("<h1>Coursemon</h1>", unsafe_allow_html=True)
st.caption("One-box search. I rewrite your query to target course/training pages across the web — no APIs.")

q = st.text_input("Search", value="History and philosophy", label_visibility="collapsed")
st.markdown("---")

if q.strip():
    s = sess()
    expanded = genai_expand(q)

    with st.expander("🔎 How I rewrote your query"):
        for i, qv in enumerate(expanded, 1):
            st.code(f"{i}. {qv}")

    raw: List[Dict] = []
    for qv in expanded:
        raw += search_bing(qv, num=10, s=s)
        raw += search_ddg(qv, num=10, s=s)
        raw += search_yahoo(qv, num=10, s=s)

    merged = dedupe(raw)
    ranked = rerank(merged, q)

    if not ranked:
        st.warning("No results parsed. Try a simpler phrasing (e.g., 'philosophy course syllabus').")
    else:
        st.write(f"**{len(ranked)} results**")
        for r, score in ranked[:50]:
            st.markdown(
                f'<div class="card">'
                f'<div class="t"><a href="{r["url"]}" target="_blank">{html.escape(r["title"] or r["url"])}</a></div>'
                f'<div class="u">{html.escape(r.get("url",""))}</div>'
                f'<div style="margin-top:6px;">{html.escape(r.get("snippet",""))}</div>'
                f'<div class="u" style="margin-top:6px;">score: {score:.2f} • {r.get("engine","")}</div>'
                f'</div>', unsafe_allow_html=True
            )

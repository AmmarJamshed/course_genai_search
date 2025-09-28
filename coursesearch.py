# app.py
# Coursemon — GENAI Search (Groq + Google + Location)
# ---------------------------------------------------

import re, html, json
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode

import streamlit as st
import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "Coursemon — GENAI Search (Groq + Google + Location)"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ==================== HTTP session ====================
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
        if not u or u in seen: 
            continue
        seen.add(u)
        out.append(r)
    return out

# ==================== Search engines ====================
def search_google(q: str, num: int, s: requests.Session) -> List[Dict]:
    """Scrape Google Search results (unofficial)."""
    try:
        url = "https://www.google.com/search?" + urlencode({"q": q, "num": min(num, 20)})
        r = s.get(url, timeout=10)
        if not r.ok:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        out = []
        for g in soup.select("div.g"):
            a = g.select_one("a")
            if not a or not a.get("href") or not a.get_text():
                continue
            title = clean(a.get_text())
            link = a["href"]
            sn = g.select_one("div.VwiC3b, span.aCOpRe, div.s3v9rd")
            snippet = clean(sn.get_text()) if sn else ""
            out.append({"title": title, "snippet": snippet, "url": link, "engine": "google"})
            if len(out) >= num:
                break
        return out
    except Exception as e:
        print("Google scraping failed:", e)
        return []

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

# ==================== Rule-based GENAI rewrite (fallback) ====================
def genai_expand(user_query: str) -> List[str]:
    base = user_query.strip()
    variants = [
        f"{base} course",
        f"{base} training",
        f"{base} program",
        f"{base} workshop",
        f"{base} syllabus",
    ]
    return list(dict.fromkeys(variants))[:8]

# ==================== Reranking (TF-IDF baseline) ====================
def rerank_tfidf(results: List[Dict], user_query: str) -> List[Tuple[Dict, float]]:
    texts = [f"{r.get('title','')} {r.get('snippet','')}" for r in results]
    if not texts: return []
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
    X = vec.fit_transform(texts + [user_query])
    sims = cosine_similarity(X[:-1], X[-1]).ravel()
    ranked = sorted(zip(results, sims.tolist()), key=lambda x: x[1], reverse=True)
    return ranked

# ==================== Groq Rewrite ====================
def groq_rewrite(user_query: str) -> Optional[List[str]]:
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return None
    try:
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system",
                 "content": ("Rewrite the query into max 8 variations targeting real course/training pages. "
                             "Return ONLY a JSON list of strings.")},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.3, "max_tokens": 400
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=30
        )
        if resp.status_code != 200: return None
        txt = resp.json()["choices"][0]["message"]["content"]
        queries = json.loads(txt)
        if isinstance(queries, list): return queries[:8]
        return None
    except Exception as e:
        print("Groq rewrite failed:", e)
        return None

# ==================== Groq Rerank ====================
def groq_rerank(results: List[Dict], user_query: str) -> Optional[List[Tuple[Dict, float]]]:
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return None
    try:
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system",
                 "content": ("You are a reranking engine for course/training search. "
                             "Return ONLY a JSON list of {index, score} objects.")},
                {"role": "user",
                 "content": json.dumps({
                     "query": user_query,
                     "results": [
                         {"index": i, "title": r.get("title"), "snippet": r.get("snippet"), "url": r.get("url")}
                         for i, r in enumerate(results)
                     ]
                 })}
            ],
            "temperature": 0, "max_tokens": 800
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=30
        )
        if resp.status_code != 200: return None
        txt = resp.json()["choices"][0]["message"]["content"]
        order = json.loads(txt)
        ranked = []
        for o in order:
            idx, score = o["index"], float(o.get("score", 1.0))
            if 0 <= idx < len(results):
                ranked.append((results[idx], score))
        return ranked
    except Exception as e:
        print("Groq rerank failed:", e)
        return None

# ==================== UI ====================
st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide")
st.markdown(
    "<style>.card{padding:12px 16px;border:1px solid #eee;border-radius:12px;margin-bottom:12px}"
    ".t{font-weight:600;font-size:1.05rem}.u{color:#6b7280;font-size:.9rem}</style>",
    unsafe_allow_html=True)
st.markdown("<h1>Coursemon</h1>", unsafe_allow_html=True)
st.caption("One-box search. Google + Groq enhanced rewriting & reranking. Now with location filter.")

q = st.text_input("Search", value="History and philosophy", label_visibility="collapsed")
location = st.text_input("Location (city or country)", value="", placeholder="e.g., Karachi, Pakistan, London, USA")
st.markdown("---")

if q.strip():
    s = sess()

    # If location entered, append to query
    q_with_loc = f"{q} {location}".strip() if location else q

    # Rewrite step
    expanded = groq_rewrite(q_with_loc)
    rewrite_engine = "groq"
    if not expanded:
        expanded = genai_expand(q_with_loc)
        rewrite_engine = "rules"

    with st.expander(f"🔎 How I rewrote your query (via {rewrite_engine})"):
        for i, qv in enumerate(expanded, 1):
            st.code(f"{i}. {qv}")

    # Collect results
    raw: List[Dict] = []
    for qv in expanded:
        raw += search_google(qv, num=10, s=s)
        raw += search_bing(qv, num=10, s=s)
        raw += search_ddg(qv, num=10, s=s)
        raw += search_yahoo(qv, num=10, s=s)

    merged = dedupe(raw)

    # Rerank step
    ranked = groq_rerank(merged, q_with_loc)
    rerank_engine = "groq"
    if not ranked:
        ranked = rerank_tfidf(merged, q_with_loc)
        rerank_engine = "tfidf"

    if not ranked:
        st.warning("No results parsed. Try a simpler phrasing.")
    else:
        st.write(f"**{len(ranked)} results** (rewrite via {rewrite_engine}, rerank via {rerank_engine})")
        for r, score in ranked[:50]:
            st.markdown(
                f'<div class="card">'
                f'<div class="t"><a href="{r["url"]}" target="_blank">{html.escape(r["title"] or r["url"])}</a></div>'
                f'<div class="u">{html.escape(r.get("url",""))}</div>'
                f'<div style="margin-top:6px;">{html.escape(r.get("snippet",""))}</div>'
                f'<div class="u" style="margin-top:6px;">score: {score:.2f} • {r.get("engine","")}</div>'
                f'</div>', unsafe_allow_html=True
            )

# app.py
# Coursemon — Minimal Web Search (No API Keys)
# --------------------------------------------
# • Single search bar; shows a simple list of results (title, snippet, URL)
# • Scrapes public HTML from Bing, DuckDuckGo, and Yahoo (best-effort parsing)
# • No dataset, no API keys, no JavaScript engine
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Notes:
# - This is best-effort HTML parsing; selectors may change over time.
# - For production reliability and scale, switch to official APIs or a crawler index.

import re
import html
import time
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlencode

import streamlit as st
import requests
from bs4 import BeautifulSoup

APP_NAME = "Coursemon — Search (No-API)"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

# =============== Utilities ===============

def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    return s

def clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", html.unescape(text)).strip()

def dedupe(results: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in results:
        u = r.get("url") or ""
        key = u.split("#")[0]
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out

# =============== Engines (HTML) ===============

def search_bing(query: str, num: int = 10, s: Optional[requests.Session] = None) -> List[Dict]:
    """Scrape Bing HTML results."""
    s = s or session()
    # Bing supports 'count' up to 50; 'first' is start index.
    params = {"q": query, "count": min(num, 20)}
    url = "https://www.bing.com/search?" + urlencode(params)
    try:
        r = s.get(url, timeout=10)
        if not r.ok:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        results = []
        for li in soup.select("li.b_algo"):
            a = li.select_one("h2 a")
            if not a or not a.get("href"):
                continue
            title = clean(a.get_text())
            link = a["href"]
            # snippet can be in p or div snippet container
            sn = li.select_one("p, div.b_caption p")
            snippet = clean(sn.get_text() if sn else "")
            results.append({"title": title, "snippet": snippet, "url": link, "engine": "bing"})
            if len(results) >= num:
                break
        return results
    except Exception:
        return []

def search_ddg(query: str, num: int = 10, s: Optional[requests.Session] = None) -> List[Dict]:
    """Scrape DuckDuckGo HTML results."""
    s = s or session()
    # Use HTML endpoint that returns simple markup
    url = "https://html.duckduckgo.com/html/"
    try:
        r = s.get(url, params={"q": query}, timeout=10)
        if not r.ok:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        results = []
        # Try multiple selectors for resilience
        for a in soup.select("a.result__a, a.result__url, a.result__a.js-result-title-link"):
            href = a.get("href")
            if not href or not href.startswith("http"):
                continue
            title = clean(a.get_text())
            # the snippet is nearby, often in .result__snippet or within the same result container
            parent = a.find_parent(class_=re.compile("result"))
            sn = parent.select_one(".result__snippet") if parent else None
            snippet = clean(sn.get_text() if sn else "")
            results.append({"title": title, "snippet": snippet, "url": href, "engine": "ddg"})
            if len(results) >= num:
                break
        return results
    except Exception:
        return []

def search_yahoo(query: str, num: int = 10, s: Optional[requests.Session] = None) -> List[Dict]:
    """Scrape Yahoo HTML results."""
    s = s or session()
    url = "https://search.yahoo.com/search?" + urlencode({"p": query})
    try:
        r = s.get(url, timeout=10)
        if not r.ok:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        results = []
        for d in soup.select("div#web h3.title a, div.dd.algo.algo-sr h3 a"):
            href = d.get("href")
            if not href or not href.startswith("http"):
                continue
            title = clean(d.get_text())
            # snippet often in following sibling
            sn = None
            cont = d.find_parent("div", class_=re.compile("algo"))
            if cont:
                sn = cont.select_one("div.compText, p")
            snippet = clean(sn.get_text() if sn else "")
            results.append({"title": title, "snippet": snippet, "url": href, "engine": "yahoo"})
            if len(results) >= num:
                break
        return results
    except Exception:
        return []

# =============== Streamlit UI ===============

st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide")
st.markdown(
    "<style> .result-card {padding: 12px 16px; border: 1px solid #eee; border-radius: 12px; margin-bottom: 12px;} "
    ".result-title{font-size:1.05rem; font-weight:600;} "
    ".muted{color:#6b7280; font-size:0.9rem;} </style>",
    unsafe_allow_html=True
)

st.markdown("<h1 style='margin-bottom:0.2rem;'>Coursemon</h1>", unsafe_allow_html=True)
st.caption("Minimal web search, no API keys. Type and press Enter.")

q = st.text_input("Search the web", value="evening data science bootcamp in Karachi under $300", label_visibility="collapsed")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    n_results = st.slider("Results per engine", 5, 20, 10)
with col2:
    use_bing = st.checkbox("Bing", value=True)
with col3:
    use_ddg = st.checkbox("DuckDuckGo", value=True)
use_yah = st.checkbox("Yahoo", value=False)

st.markdown("---")

if q.strip():
    s = session()
    all_results: List[Dict] = []
    with st.spinner("Searching..."):
        if use_bing:
            all_results += search_bing(q, num=n_results, s=s)
        if use_ddg:
            all_results += search_ddg(q, num=n_results, s=s)
        if use_yah:
            all_results += search_yahoo(q, num=n_results, s=s)

    # Merge & dedupe by URL
    results = dedupe(all_results)

    if not results:
        st.warning("No results parsed. Try toggling engines, simplifying the query, or increasing results per engine.")
    else:
        st.write(f"**{len(results)} results**")
        for r in results:
            st.markdown(f"""
<div class="result-card">
  <div class="result-title"><a href="{r['url']}" target="_blank">{html.escape(r['title'] or r['url'])}</a></div>
  <div class="muted">{html.escape(r.get('url',''))}</div>
  <div style="margin-top:6px;">{html.escape(r.get('snippet',''))}</div>
  <div class="muted" style="margin-top:6px;">source: {r.get('engine','')}</div>
</div>
""", unsafe_allow_html=True)

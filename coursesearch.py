# app.py — Coursemon Global Course & Training Search (Light Edition)
# -----------------------------------------------------------------
# ✅ Works entirely without machine learning
# ✅ Async scraping from Coursera, Udemy, EdX, Simplilearn, DataCamp, Educative
# ✅ Optional SerpAPI results (if key in secrets)
# ✅ Keyword-based filtering + TF-IDF reranking
# ✅ 100% Streamlit Cloud compatible
# -----------------------------------------------------------------

import re, html, asyncio
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode

import streamlit as st
import aiohttp
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "Coursemon — Global Course & Training Search 🌍"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ==================== Helpers ====================
def clean(x: Optional[str]) -> str:
    if not x:
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(x))).strip()

def dedupe(results: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in results:
        u = (r.get("url") or "").split("#")[0]
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(r)
    return out

async def fetch_html(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.get(url, headers={"User-Agent": UA}, timeout=15) as resp:
            if resp.status != 200:
                return ""
            return await resp.text()
    except Exception:
        return ""

# ==================== Scrapers ====================

async def search_coursera(q: str, session: aiohttp.ClientSession) -> List[Dict]:
    url = f"https://www.coursera.org/search?query={q.replace(' ', '%20')}"
    html_text = await fetch_html(session, url)
    if not html_text: return []
    soup = BeautifulSoup(html_text, "lxml")
    results = []
    for item in soup.select("li[data-testid='search-result']"):
        title = clean(item.select_one("h2"))
        desc = clean(item.select_one("p"))
        link_tag = item.select_one("a[href]")
        link = f"https://www.coursera.org{link_tag['href']}" if link_tag else ""
        if title and link:
            results.append({"title": title, "snippet": desc, "url": link, "engine": "coursera"})
    return results

async def search_udemy(q: str, session: aiohttp.ClientSession) -> List[Dict]:
    url = f"https://www.udemy.com/courses/search/?q={q.replace(' ', '%20')}"
    html_text = await fetch_html(session, url)
    if not html_text: return []
    soup = BeautifulSoup(html_text, "lxml")
    results = []
    for a in soup.select("a.udlite-custom-focus-visible"):
        title = clean(a.get_text())
        link = "https://www.udemy.com" + a.get("href", "")
        results.append({"title": title, "snippet": "", "url": link, "engine": "udemy"})
    return results

async def search_edx(q: str, session: aiohttp.ClientSession) -> List[Dict]:
    url = f"https://www.edx.org/search?q={q.replace(' ', '+')}"
    html_text = await fetch_html(session, url)
    if not html_text: return []
    soup = BeautifulSoup(html_text, "lxml")
    results = []
    for card in soup.select("div.discovery-card"):
        title = clean(card.select_one("h3"))
        desc = clean(card.select_one("div.discovery-card__content__description"))
        a = card.select_one("a.discovery-card-link")
        link = a["href"] if a and a.get("href", "").startswith("http") else f"https://www.edx.org{a['href']}" if a else ""
        if title and link:
            results.append({"title": title, "snippet": desc, "url": link, "engine": "edx"})
    return results

async def search_simplilearn(q: str, session: aiohttp.ClientSession) -> List[Dict]:
    url = f"https://www.simplilearn.com/search?q={q.replace(' ', '+')}"
    html_text = await fetch_html(session, url)
    if not html_text: return []
    soup = BeautifulSoup(html_text, "lxml")
    results = []
    for card in soup.select("div.search-result-card"):
        title = clean(card.select_one("h3"))
        desc = clean(card.select_one("p"))
        a = card.select_one("a[href]")
        link = f"https://www.simplilearn.com{a['href']}" if a else ""
        if title and link:
            results.append({"title": title, "snippet": desc, "url": link, "engine": "simplilearn"})
    return results

async def search_datacamp(q: str, session: aiohttp.ClientSession) -> List[Dict]:
    url = f"https://www.datacamp.com/search?q={q.replace(' ', '+')}"
    html_text = await fetch_html(session, url)
    if not html_text: return []
    soup = BeautifulSoup(html_text, "lxml")
    results = []
    for card in soup.select("a.course-block"):
        title = clean(card.select_one("h4") or card.get_text())
        link = "https://www.datacamp.com" + card.get("href", "")
        results.append({"title": title, "snippet": "", "url": link, "engine": "datacamp"})
    return results

async def search_educative(q: str, session: aiohttp.ClientSession) -> List[Dict]:
    url = f"https://www.educative.io/search?query={q.replace(' ', '%20')}"
    html_text = await fetch_html(session, url)
    if not html_text: return []
    soup = BeautifulSoup(html_text, "lxml")
    results = []
    for card in soup.select("a.course-card"):
        title = clean(card.get_text())
        link = "https://www.educative.io" + card.get("href", "")
        results.append({"title": title, "snippet": "", "url": link, "engine": "educative"})
    return results

async def search_serpapi(q: str, session: aiohttp.ClientSession) -> List[Dict]:
    api_key = st.secrets.get("SERP_API_KEY")
    if not api_key:
        return []
    params = {"engine": "google", "q": q, "num": 10, "api_key": api_key}
    try:
        async with session.get("https://serpapi.com/search", params=params, timeout=10) as r:
            if r.status != 200:
                return []
            data = await r.json()
            return [
                {"title": clean(item.get("title")),
                 "snippet": clean(item.get("snippet")),
                 "url": item.get("link"),
                 "engine": "serpapi"}
                for item in data.get("organic_results", [])
            ]
    except Exception:
        return []

# ==================== Async Aggregator ====================
@st.cache_data(show_spinner="Fetching global courses...", ttl=1800)
def cached_results(queries: List[str]) -> List[Dict]:
    return asyncio.run(gather_all(queries))

async def gather_all(queries: List[str]) -> List[Dict]:
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for qv in queries:
            tasks += [
                search_coursera(qv, session),
                search_udemy(qv, session),
                search_edx(qv, session),
                search_simplilearn(qv, session),
                search_datacamp(qv, session),
                search_educative(qv, session),
                search_serpapi(qv, session)
            ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in all_results:
            if isinstance(res, list):
                results.extend(res)
    return results

# ==================== Filtering & Ranking ====================
def keyword_filter(results: List[Dict]) -> List[Dict]:
    include_terms = ["course", "training", "program", "class", "bootcamp", "certificate", "learning"]
    exclude_terms = ["news", "blog", "job", "vacancy", "forum", "review"]
    out = []
    for r in results:
        text = f"{r.get('title','')} {r.get('snippet','')} {r.get('url','')}".lower()
        if not any(t in text for t in include_terms): continue
        if any(t in text for t in exclude_terms): continue
        out.append(r)
    return out

def rerank_tfidf(results: List[Dict], user_query: str) -> List[Tuple[Dict, float]]:
    texts = [f"{r.get('title','')} {r.get('snippet','')}" for r in results]
    if not texts: return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    X = vec.fit_transform(texts + [user_query])
    sims = cosine_similarity(X[:-1], X[-1]).ravel()
    ranked = sorted(zip(results, sims.tolist()), key=lambda x: x[1], reverse=True)
    return ranked

def expand_query(q: str) -> List[str]:
    base = q.strip()
    variants = [
        f"{base} course", f"{base} training", f"{base} certification",
        f"{base} bootcamp", f"{base} workshop", f"{base} class"
    ]
    return list(dict.fromkeys(variants))

# ==================== UI ====================
st.set_page_config(page_title=APP_NAME, page_icon="🌍", layout="wide")
st.title("🌍 Coursemon Global Course & Training Search")
st.caption("Find global courses and trainings from Coursera, Udemy, EdX, Simplilearn, DataCamp, Educative, and Google — async, fast, and API-free.")

query = st.text_input("Search for any topic", value="Data Science", label_visibility="collapsed")
st.markdown("---")

if query.strip():
    expanded = expand_query(query)
    with st.expander("🔍 Expanded Queries"):
        for i, qv in enumerate(expanded, 1):
            st.code(f"{i}. {qv}")

    raw = cached_results(expanded)
    merged = dedupe(raw)
    filtered = keyword_filter(merged)
    ranked = rerank_tfidf(filtered, query)

    if not ranked:
        st.warning("No relevant courses or trainings found.")
    else:
        st.success(f"✅ Found {len(ranked)} relevant results.")
        for r, score in ranked[:60]:
            st.markdown(
                f'<div style="border:1px solid #ddd;padding:12px;border-radius:10px;margin-bottom:8px">'
                f'<b><a href="{r["url"]}" target="_blank">{html.escape(r["title"] or r["url"])}</a></b><br>'
                f'<small>{r.get("engine","")} • Score: {score:.2f}</small><br>'
                f'{html.escape(r.get("snippet",""))}</div>', unsafe_allow_html=True
            )

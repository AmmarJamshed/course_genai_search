# app.py — Fixed SerpAPI async integration
# -----------------------------------------

import re, html, json, asyncio
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode

import streamlit as st
import aiohttp
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "Coursemon — GENAI Search"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ==================== Helpers ====================
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

# ==================== Async Fetch ====================
async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return ""
            return await resp.text()
    except Exception:
        return ""

# ✅ Async SerpAPI integration (no threads, pure async)
async def search_serpapi_async(q: str, num: int, session: aiohttp.ClientSession) -> List[Dict]:
    api_key = st.secrets.get("SERP_API_KEY")
    if not api_key:
        return []
    try:
        params = {
            "engine": "google",
            "q": q,
            "num": num,
            "api_key": api_key
        }
        async with session.get("https://serpapi.com/search", params=params, timeout=10) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            out = []
            for item in data.get("organic_results", []):
                out.append({
                    "title": clean(item.get("title")),
                    "snippet": clean(item.get("snippet")),
                    "url": item.get("link"),
                    "engine": "serpapi"
                })
            return out
    except Exception:
        return []

# ✅ Google (HTML backup)
async def search_google_async(q: str, num: int, session: aiohttp.ClientSession) -> List[Dict]:
    url = "https://www.google.com/search?" + urlencode({"q": q, "num": min(num, 20)})
    html_text = await fetch(session, url)
    if not html_text: return []
    soup = BeautifulSoup(html_text, "lxml")
    out = []
    for g in soup.select("div.tF2Cxc, div.g"):
        a = g.select_one("a")
        if not a or not a.get("href"): 
            continue
        title = clean(a.get_text())
        link = a["href"]
        sn = g.select_one("div.VwiC3b, span.aCOpRe, div.s3v9rd, div.IsZvec")
        snippet = clean(sn.get_text()) if sn else ""
        if link.startswith("/"): 
            continue
        out.append({"title": title, "snippet": snippet, "url": link, "engine": "google"})
        if len(out) >= num: break
    return out

async def search_bing_async(q: str, num: int, session: aiohttp.ClientSession) -> List[Dict]:
    url = "https://www.bing.com/search?" + urlencode({"q": q, "count": min(num, 20)})
    html_text = await fetch(session, url)
    if not html_text: return []
    soup = BeautifulSoup(html_text, "lxml")
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

async def search_ddg_async(q: str, num: int, session: aiohttp.ClientSession) -> List[Dict]:
    html_text = await fetch(session, "https://html.duckduckgo.com/html/?" + urlencode({"q": q}))
    if not html_text: return []
    soup = BeautifulSoup(html_text, "lxml")
    out = []
    for res in soup.select(".result"):
        a = res.select_one("a.result__a, a.result__url")
        if not a or not a.get("href") or not a["href"].startswith("http"): continue
        title = clean(a.get_text())
        sn = res.select_one(".result__snippet")
        snippet = clean(sn.get_text() if sn else "")
        out.append({"title": title, "snippet": snippet, "url": a["href"], "engine": "ddg"})
        if len(out) >= num: break
    return out

async def search_yahoo_async(q: str, num: int, session: aiohttp.ClientSession) -> List[Dict]:
    url = "https://search.yahoo.com/search?" + urlencode({"p": q})
    html_text = await fetch(session, url)
    if not html_text: return []
    soup = BeautifulSoup(html_text, "lxml")
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

# ==================== Async Orchestrator ====================
@st.cache_data(show_spinner="Fetching results...", ttl=3600)
def cached_results(expanded: List[str]) -> List[Dict]:
    return asyncio.run(gather_results(expanded))

async def gather_results(expanded: List[str]) -> List[Dict]:
    results = []
    async with aiohttp.ClientSession(headers={"User-Agent": UA}) as session:
        tasks = []
        for qv in expanded:
            tasks += [
                search_serpapi_async(qv, 10, session),
                search_google_async(qv, 10, session),
                search_bing_async(qv, 10, session),
                search_ddg_async(qv, 10, session),
                search_yahoo_async(qv, 10, session),
            ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in all_results:
            if isinstance(res, list):
                results.extend(res)
    return results

# ==================== Expand / Rerank ====================
def genai_expand(user_query: str) -> List[str]:
    base = user_query.strip()
    variants = [f"{base} course", f"{base} training", f"{base} program", f"{base} workshop", f"{base} syllabus"]
    return list(dict.fromkeys(variants))[:5]

def rerank_tfidf(results: List[Dict], user_query: str) -> List[Tuple[Dict, float]]:
    texts = [f"{r.get('title','')} {r.get('snippet','')}" for r in results]
    if not texts: return []
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
    X = vec.fit_transform(texts + [user_query])
    sims = cosine_similarity(X[:-1], X[-1]).ravel()
    ranked = sorted(zip(results, sims.tolist()), key=lambda x: x[1], reverse=True)
    return ranked

# ==================== UI ====================
st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide")
st.markdown(
    "<style>.card{padding:12px 16px;border:1px solid #eee;border-radius:12px;margin-bottom:12px}"
    ".t{font-weight:600;font-size:1.05rem}.u{color:#6b7280;font-size:.9rem}</style>",
    unsafe_allow_html=True)
st.markdown("<h1>Coursemon</h1>", unsafe_allow_html=True)
st.caption("One-box search with SerpAPI + Google + Bing + DDG + Yahoo, Async + filters + cache.")

q = st.text_input("Search", value="History and philosophy", label_visibility="collapsed")
course_type = st.selectbox("Filter by course type", ["All", "Online Course", "Onsite Training", "Online Live Training"])
location = st.text_input("Location (city or country)", value="", placeholder="e.g., Karachi, Pakistan, London, USA")
st.markdown("---")

if q.strip():
    q_with_loc = f"{q} {location}".strip() if location else q
    if course_type != "All":
        q_with_loc = f"{q_with_loc} {course_type}"
    expanded = genai_expand(q_with_loc)
    with st.expander("🔎 Query variants"):
        for i, qv in enumerate(expanded, 1):
            st.code(f"{i}. {qv}")

    raw = cached_results(expanded)
    merged = dedupe(raw)
    ranked = rerank_tfidf(merged, q_with_loc)

    if not ranked:
        st.warning("No results found.")
    else:
        st.write(f"**{len(ranked)} results** (includes SerpAPI-enhanced Google results)")
        for r, score in ranked[:50]:
            st.markdown(
                f'<div class="card">'
                f'<div class="t"><a href="{r["url"]}" target="_blank">{html.escape(r["title"] or r["url"])}</a></div>'
                f'<div class="u">{html.escape(r.get("url",""))}</div>'
                f'<div style="margin-top:6px;">{html.escape(r.get("snippet",""))}</div>'
                f'<div class="u" style="margin-top:6px;">score: {score:.2f} • {r.get("engine","")}</div>'
                f'</div>', unsafe_allow_html=True
            )

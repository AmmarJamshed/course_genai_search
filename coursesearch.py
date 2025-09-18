# app.py
# Coursemon — GenAI Web Search (Scraper MVP) — Patched
# ----------------------------------------------------
# Live web search (no fixed dataset):
# - DuckDuckGo HTML search (no API key) -> candidate links
# - Polite fetching with robots.txt check (lenient fallback) + caching (via requests-cache, set outside)
# - Site-specific parsers (Coursera, edX, Udemy, Udacity) + generic heuristic parser
# - TF-IDF ranking of [title + description + provider] vs. your query
#
# How to run:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Notes:
# - Keep result/page limits reasonable. For production, use Scrapy/Playwright + site permissions/APIs.

import re
import os
import time
import html
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st
import requests
import requests_cache
from bs4 import BeautifulSoup
from urllib import robotparser
import tldextract

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "Coursemon — GenAI Web Search (Scraper MVP)"
USER_AGENT = (
    "Mozilla/5.0 (compatible; CoursemonScraper/1.1; +https://coursemon.net/bot) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
)

# Cache HTTP for 30 minutes to be polite & fast
requests_cache.install_cache("coursemon_cache", expire_after=1800)

@dataclass
class Course:
    title: str
    provider: str
    url: str
    description: str = ""
    location: str = ""
    mode: str = ""          # Online / Onsite / Hybrid / Unknown
    price: str = ""         # keep raw string
    duration: str = ""
    start_date: str = ""
    extras: Dict = None

# -----------------------
# Helpers
# -----------------------
def polite_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s

def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join([p for p in [ext.domain, ext.suffix] if p]).lower()

def clean_text(x: Optional[str]) -> str:
    if not x:
        return ""
    return re.sub(r"\s+", " ", html.unescape(x)).strip()

def allowed_by_robots(url: str, session: requests.Session, cache: Dict) -> bool:
    """Robots check — if robots.txt can't be fetched or parsed, ALLOW by default (lenient for MVP)."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        if robots_url in cache:
            rp = cache[robots_url]
        else:
            resp = session.get(robots_url, timeout=6)
            rp = robotparser.RobotFileParser()
            rp.set_url(robots_url)
            rp.parse(resp.text.splitlines() if resp.ok else [])
            cache[robots_url] = rp
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        # Lenient fallback for MVP
        return True

# -----------------------
# DuckDuckGo HTML search (patched)
# -----------------------
def ddg_search(query: str, max_results: int = 12, session: Optional[requests.Session] = None) -> List[str]:
    """
    Use DuckDuckGo HTML endpoint and parse result links.
    This is best-effort HTML scraping and may change; we handle gracefully.
    """
    session = session or polite_session()
    base = "https://html.duckduckgo.com/html/"  # more stable HTML endpoint
    params = {"q": query}
    try:
        resp = session.get(base, params=params, timeout=12)
        if not resp.ok:
            return []
        soup = BeautifulSoup(resp.text, "lxml")

        # Try multiple selectors to be resilient to class changes
        links = []
        for a in soup.select("a.result__a, a.result__url, a.result__a.js-result-title-link"):
            href = a.get("href")
            if not href:
                continue
            if href.startswith("http"):
                links.append(href)
            if len(links) >= max_results:
                break

        # Another fallback: look for <a> tags inside result containers
        if not links:
            for r in soup.select(".result, .result__body"):
                a = r.find("a", href=True)
                if a and a["href"].startswith("http"):
                    links.append(a["href"])
                    if len(links) >= max_results:
                        break
        return links
    except Exception:
        return []

# -----------------------
# Site-specific parsers
# -----------------------
def parse_coursera(soup: BeautifulSoup, url: str) -> Dict:
    out = {}
    t = soup.find("h1") or soup.find("h2")
    out["title"] = clean_text(t.get_text()) if t else ""
    prov = soup.select_one("[data-test='partner-link']") or soup.find(string=re.compile("Coursera", re.I))
    out["provider"] = "Coursera" if prov is None else f"Coursera — {clean_text(getattr(prov, 'get_text', lambda: prov)())[:120]}"
    desc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
    out["description"] = clean_text(desc.get("content") if desc else "")
    out["mode"] = "Online"
    return out

def parse_edx(soup: BeautifulSoup, url: str) -> Dict:
    out = {}
    title = soup.find("h1") or soup.find("h2")
    out["title"] = clean_text(title.get_text()) if title else ""
    prov = soup.find("a", href=re.compile(r"/school/|/xseries/|/micromasters/"))
    out["provider"] = "edX" if prov is None else f"edX — {clean_text(prov.get_text())}"
    desc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
    out["description"] = clean_text(desc.get("content") if desc else "")
    out["mode"] = "Online"
    return out

def parse_udemy(soup: BeautifulSoup, url: str) -> Dict:
    out = {}
    title = soup.find("h1") or soup.find("h2")
    out["title"] = clean_text(title.get_text()) if title else ""
    out["provider"] = "Udemy"
    desc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
    out["description"] = clean_text(desc.get("content") if desc else "")
    out["mode"] = "Online"
    return out

def parse_udacity(soup: BeautifulSoup, url: str) -> Dict:
    out = {}
    title = soup.find("h1") or soup.find("h2")
    out["title"] = clean_text(title.get_text()) if title else ""
    out["provider"] = "Udacity"
    desc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
    out["description"] = clean_text(desc.get("content") if desc else "")
    out["mode"] = "Online"
    return out

def generic_parser(soup: BeautifulSoup, url: str) -> Dict:
    out = {}
    ogt = soup.find("meta", {"property": "og:title"})
    if ogt and ogt.get("content"):
        out["title"] = clean_text(ogt.get("content"))
    else:
        h1 = soup.find("h1") or soup.find("h2")
        out["title"] = clean_text(h1.get_text()) if h1 else ""

    ogd = soup.find("meta", {"property": "og:description"}) or soup.find("meta", {"name": "description"})
    out["description"] = clean_text(ogd.get("content")) if ogd and ogd.get("content") else ""

    domain = extract_domain(url)
    provider = domain
    prov_hint = soup.find(string=re.compile(r"(University|Institute|College|School|Academy|Training)", re.I))
    if prov_hint and hasattr(prov_hint, "parent"):
        provider = clean_text(provider + " — " + prov_hint.parent.get_text()[:120])
    out["provider"] = provider

    text = soup.get_text(" ", strip=True)
    price_match = re.search(r"(?i)(USD|PKR|Rs\.?|£|\$|€)\s?\d[\d,\.]*", text)
    duration_match = re.search(r"(?i)(\d+\s*(hours?|days?|weeks?|months?|year|years))", text)
    location_match = re.search(r"(?i)\b(Karachi|Lahore|Islamabad|London|Berlin|Dubai|New York|Pakistan|UK|USA|Spain|Germany)\b", text)

    out["price"] = price_match.group(0) if price_match else ""
    out["duration"] = duration_match.group(0) if duration_match else ""
    out["location"] = location_match.group(0) if location_match else ""

    if re.search(r"(?i)\bonline\b|\bremote\b|\blive online\b", text):
        out["mode"] = "Online"
    elif re.search(r"(?i)\b(on-site|onsite|campus|in person|in-person)\b", text):
        out["mode"] = "Onsite"
    else:
        out["mode"] = ""
    return out

PARSERS = {
    "coursera.org": parse_coursera,
    "edx.org": parse_edx,
    "udemy.com": parse_udemy,
    "udacity.com": parse_udacity,
}

def fetch_and_parse(url: str, session: requests.Session, robots_cache: Dict) -> Optional[Course]:
    try:
        if not allowed_by_robots(url, session, robots_cache):
            return None

        resp = session.get(url, timeout=12)
        if not resp.ok or not resp.text:
            return None
        soup = BeautifulSoup(resp.text, "lxml")

        parser = None
        for key, fn in PARSERS.items():
            if key in url:
                parser = fn
                break
        data = parser(soup, url) if parser else generic_parser(soup, url)

        title = data.get("title") or (soup.title.get_text(strip=True) if soup.title else "")
        provider = data.get("provider") or extract_domain(url)
        description = data.get("description") or ""

        course = Course(
            title=clean_text(title)[:180],
            provider=clean_text(provider)[:180],
            url=url,
            description=clean_text(description)[:500],
            location=clean_text(data.get("location", ""))[:120],
            mode=clean_text(data.get("mode", ""))[:40],
            price=clean_text(data.get("price", ""))[:60],
            duration=clean_text(data.get("duration", ""))[:80],
            start_date=clean_text(data.get("start_date", ""))[:60],
            extras={k: v for k, v in data.items() if k not in {"title","provider","description","location","mode","price","duration","start_date"}}
        )
        if not course.title:
            return None
        return course
    except Exception:
        return None

# -----------------------
# Ranking
# -----------------------
def rank_courses(courses: List[Course], query: str) -> List[Tuple[Course, float]]:
    texts = [" ".join([c.title or "", c.description or "", c.provider or ""]) for c in courses]
    if not texts:
        return []
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
    X = vec.fit_transform(texts + [query])
    sims = cosine_similarity(X[:-1], X[-1]).ravel()
    ranked = sorted(zip(courses, sims.tolist()), key=lambda x: x[1], reverse=True)
    return ranked

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")
st.title(APP_NAME)
st.caption("Type what you need. I’ll search the open web, crawl allowed pages, parse likely course/training info, and rank results — live.")

with st.sidebar:
    default_q = "short evening data science bootcamp in Karachi under $300"
    q = st.text_area("🔎 Your query", value=default_q, height=90)
    st.markdown("### Crawl Controls")
    max_results = st.slider("Max search results", 5, 30, 20)
    max_fetch = st.slider("Max pages to fetch", 3, 30, 12)
    per_domain_limit = st.slider("Per-domain cap", 1, 10, 3)
    domains_filter = st.text_input("Restrict to domains (comma-separated)", value="", help="e.g. coursera.org,edx.org,nust.edu.pk")
    st.markdown("---")
    st.caption("Respects robots.txt where available. For production, use partnerships/APIs.")

if not q.strip():
    st.stop()

session = polite_session()
robots_cache: Dict[str, robotparser.RobotFileParser] = {}

# 1) Search the web
links = ddg_search(q, max_results=max_results, session=session)

# Filter by domain & per-domain cap
allowed_domains = [d.strip().lower() for d in domains_filter.split(",") if d.strip()] if domains_filter.strip() else []
filtered_links = []
per_domain_counts = {}
for url in links:
    dom = extract_domain(url)
    if allowed_domains and not any(ad in url for ad in allowed_domains):
        continue
    if per_domain_counts.get(dom, 0) >= per_domain_limit:
        continue
    per_domain_counts[dom] = per_domain_counts.get(dom, 0) + 1
    filtered_links.append(url)
    if len(filtered_links) >= max_fetch:
        break

with st.expander("🔧 Debug: discovered links"):
    st.write({"raw_links": links, "filtered_links": filtered_links})

st.write("## Web Results")
if not filtered_links:
    st.warning("No fetchable links from the search (try widening per-domain cap, removing domain filters, or tweaking the query).")
    st.stop()

# 2) Fetch + parse
courses: List[Course] = []
progress = st.progress(0)
for i, url in enumerate(filtered_links):
    progress.progress((i+1) / max(1, len(filtered_links)))
    c = fetch_and_parse(url, session=session, robots_cache=robots_cache)
    if c:
        courses.append(c)
    time.sleep(0.5 + random.random() * 0.5)  # polite delay
progress.empty()

if not courses:
    st.info("I couldn't extract structured info from the fetched pages. Try adjusting the query or domains.")
    st.stop()

# 3) Rank & show
ranked = rank_courses(courses, q)
st.write(f"### Top Matches ({len(ranked)})")
for course, score in ranked:
    with st.container(border=True):
        st.markdown(f"#### [{course.title}]({course.url})")
        st.write(f"**Provider:** {course.provider or '—'}  |  **Location:** {course.location or '—'}  |  **Mode:** {course.mode or '—'}")
        st.write(f"**Price:** {course.price or '—'}  |  **Duration:** {course.duration or '—'}  |  **Start:** {course.start_date or '—'}")
        if course.description:
            st.write(course.description)
        st.caption(f"Relevance score: {score:.2f}")

st.markdown("---")
with st.expander("Roadmap / Production Hardening"):
    st.markdown(
        """
- Upgrade to Scrapy/Playwright for JS-heavy sites and robust crawling queues.
- Add official APIs/partner data feeds for reliability and ToS alignment.
- Swap TF-IDF for embedding search (BGE-M3/Instructor-XL) + vector DB (FAISS/Weaviate/Pinecone).
- Add faceted filters (price, level, accreditation, start date, delivery).
- Institution dashboards + sponsored listings (data monetization, not commissions).
"""
    )

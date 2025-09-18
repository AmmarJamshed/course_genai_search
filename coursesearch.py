# app.py
# Coursemon — GenAI Web Search MVP (Live Scraper)
# ------------------------------------------------
# What this does
# - Takes a natural-language prompt (e.g., "evening data science bootcamp in Karachi under $300")
# - Performs a web search (DuckDuckGo HTML - no API key) to find likely course/training pages
# - Polite crawling: robots.txt check, caching, timeouts, user-agent
# - Parses pages with site-specific extractors (Coursera, edX, Udemy, Udacity) + a generic heuristic parser
# - Scores & ranks results vs your prompt, and displays clean result cards with extracted fields
#
# NOTES & ETHICS
# - Respect each site's robots.txt and Terms of Service.
# - Use reasonable limits (max pages, delays) and add your institute partnerships where possible.
# - For heavy-scale ops, migrate to a proper crawler (Scrapy/Playwright) and obtain permissions.
#
# How to run
#   1) pip install -r requirements.txt
#   2) streamlit run app.py

import re
import os
import time
import html
import math
import random
import urllib.parse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import streamlit as st
import requests
import requests_cache
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from urllib import robotparser
import tldextract

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# Config & global toggles
# -----------------------
APP_NAME = "Coursemon — GenAI Web Search (Scraper MVP)"
USER_AGENT = (
    "Mozilla/5.0 (compatible; CoursemonScraper/1.0; +https://coursemon.net/bot) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
)

# Cache HTTP to be polite & fast (30 minutes)
requests_cache.install_cache("coursemon_cache", expire_after=1800)


# -----------------------
# Data classes
# -----------------------
@dataclass
class Course:
    title: str
    provider: str
    url: str
    description: str = ""
    location: str = ""
    mode: str = ""          # Online / Onsite / Hybrid / Unknown
    price: str = ""         # keep raw string; pricing extraction is messy
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


def allowed_by_robots(url: str, session: requests.Session, cache: Dict) -> bool:
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
        # If robots fetch fails, be conservative:
        return False


def clean_text(x: Optional[str]) -> str:
    if not x:
        return ""
    return re.sub(r"\s+", " ", html.unescape(x)).strip()


def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
    return domain.lower()


def score_similarity(texts: List[str], query: str) -> float:
    texts = [t if t else "" for t in texts]
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
    X = vec.fit_transform(texts + [query])
    qv = X[-1]
    sims = cosine_similarity(X[:-1], qv)
    return float(np.max(sims)) if len(texts) else 0.0


# -----------------------
# DuckDuckGo HTML search
# -----------------------
def ddg_search(query: str, max_results: int = 12, session: Optional[requests.Session] = None) -> List[str]:
    """Use DuckDuckGo HTML to get result links (no API key).
    NOTE: This is best-effort HTML parsing and may break; handle gracefully.
    """
    session = session or polite_session()
    base = "https://duckduckgo.com/html/"
    params = {"q": query, "kl": "wt-wt"}  # world target
    try:
        resp = session.get(base, params=params, timeout=10)
        if not resp.ok:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        links = []
        for a in soup.select("a.result__a"):
            href = a.get("href")
            if not href:
                continue
            # DuckDuckGo gives direct links usually; sanitize
            links.append(href)
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
    # Title
    t = soup.find("h1")
    if not t:
        t = soup.find("h2")
    out["title"] = clean_text(t.get_text()) if t else ""
    # Provider
    prov = soup.select_one("[data-test='partner-link'], a[href*='/coursera.org/professional-certificates/']")
    out["provider"] = "Coursera" if prov is None else f"Coursera — {clean_text(prov.get_text())}"
    # Description
    desc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
    out["description"] = clean_text(desc.get("content") if desc else "")
    out["mode"] = "Online"
    return out


def parse_edx(soup: BeautifulSoup, url: str) -> Dict:
    out = {}
    title = soup.find("h1") or soup.find("h2")
    out["title"] = clean_text(title.get_text()) if title else ""
    # provider
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
    prov = soup.find("a", href=re.compile(r"/user/|/organization/"))
    out["provider"] = "Udemy" if prov is None else f"Udemy — {clean_text(prov.get_text())}"
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
    # Try og:title, h1, title
    ogt = soup.find("meta", {"property": "og:title"})
    if ogt and ogt.get("content"):
        out["title"] = clean_text(ogt.get("content"))
    else:
        h1 = soup.find("h1") or soup.find("h2")
        out["title"] = clean_text(h1.get_text()) if h1 else ""

    # Description
    ogd = soup.find("meta", {"property": "og:description"}) or soup.find("meta", {"name": "description"})
    out["description"] = clean_text(ogd.get("content")) if ogd and ogd.get("content") else ""

    # Provider inference from domain and page
    domain = extract_domain(url)
    provider = domain
    # often an element like "By XYZ University" or "Provider"
    prov_hint = soup.find(string=re.compile(r"(University|Institute|College|School|Academy|Training)", re.I))
    if prov_hint and hasattr(prov_hint, "parent"):
        provider = clean_text(provider + " — " + prov_hint.parent.get_text()[:120])
    out["provider"] = provider

    # Heuristic grab for price/duration/location
    text = soup.get_text(" ", strip=True)
    price_match = re.search(r"(?i)(USD|PKR|Rs\.?|£|\$|€)\s?\d[\d,\.]*", text)
    duration_match = re.search(r"(?i)(\d+\s*(hours?|days?|weeks?|months?|year|years))", text)
    location_match = re.search(r"(?i)\b(Karachi|Lahore|Islamabad|London|Berlin|Dubai|New York|Pakistan|UK|USA|Spain|Germany)\b", text)

    out["price"] = price_match.group(0) if price_match else ""
    out["duration"] = duration_match.group(0) if duration_match else ""
    out["location"] = location_match.group(0) if location_match else ""

    # Mode inference
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


def fetch_and_parse(url: str, session: requests.Session, robots_cache: Dict, js_render: bool = False) -> Optional[Course]:
    try:
        if not allowed_by_robots(url, session, robots_cache):
            return None

        # Fetch
        resp = session.get(url, timeout=12)
        if not resp.ok or not resp.text:
            return None
        soup = BeautifulSoup(resp.text, "lxml")

        domain = extract_domain(url)
        parser = None
        for key, fn in PARSERS.items():
            if key in url:
                parser = fn
                break

        if parser:
            data = parser(soup, url)
        else:
            data = generic_parser(soup, url)

        # Fallbacks
        title = data.get("title") or soup.title.get_text(strip=True) if soup.title else ""
        provider = data.get("provider") or domain
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
        # Filter obviously blank results
        if not course.title:
            return None
        return course

    except Exception:
        return None


# -----------------------
# Ranking
# -----------------------
def rank_courses(courses: List[Course], query: str) -> List[Tuple[Course, float]]:
    # Build simple scoring over [title, description, provider]
    texts = []
    for c in courses:
        texts.append(" ".join([c.title or "", c.description or "", c.provider or ""]))
    if not texts:
        return []
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
    X = vec.fit_transform(texts + [query])
    sims = cosine_similarity(X[:-1], X[-1])
    scores = [float(s) for s in sims]
    ranked = sorted(zip(courses, scores), key=lambda x: x[1], reverse=True)
    return ranked


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")

st.title(APP_NAME)
st.caption("Type what you need. I’ll search the open web, crawl allowed pages, parse likely course/training info, and rank results — live.")

with st.sidebar:
    default_q = "short evening data science bootcamp in Karachi under $300"
    q = st.text_area("🔎 Your query", value=default_q, height=90, help="Ask for any course/training/school (location, price, mode, duration, etc.)")
    st.markdown("### Crawl Controls")
    max_results = st.slider("Max search results", 5, 30, 12, help="From DuckDuckGo HTML")
    max_fetch = st.slider("Max pages to fetch", 3, 30, 10, help="Limit to avoid hammering sites")
    per_domain_limit = st.slider("Per-domain cap", 1, 10, 3)
    js_render = st.checkbox("Enable JS rendering (requests_html)", value=False, help="Experimental; slower")
    domains_filter = st.text_input("Restrict to domains (comma-separated)", value="", help="e.g. coursera.org,edx.org,nust.edu.pk")
    st.markdown("---")
    st.caption("Be respectful of robots.txt and site Terms. For production, use partnerships/APIs.")

if not q.strip():
    st.stop()

session = polite_session()
robots_cache: Dict[str, robotparser.RobotFileParser] = {}

# 1) Search the web
st.write("## Web Results")
links = ddg_search(q, max_results=max_results, session=session)

# Domain filter
allowed_domains = []
if domains_filter.strip():
    allowed_domains = [d.strip().lower() for d in domains_filter.split(",") if d.strip()]

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

if not filtered_links:
    st.warning("No fetchable links from the search (check query or domain filters).")
    st.stop()

# 2) Fetch + parse pages
courses: List[Course] = []
progress = st.progress(0)
for i, url in enumerate(filtered_links):
    progress.progress((i+1) / max(1, len(filtered_links)))
    course = fetch_and_parse(url, session=session, robots_cache=robots_cache, js_render=js_render)
    if course:
        courses.append(course)
    time.sleep(0.6 + random.random() * 0.6)  # small polite delay

progress.empty()

if not courses:
    st.info("I couldn't extract structured info from the fetched pages. Try adjusting the query or domain filters.")
    st.stop()

# 3) Rank results vs query
ranked = rank_courses(courses, q)

# 4) Show results
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
- Add **Scrapy or Playwright** for robust crawling and JS-heavy sites; implement site-specific parsers per partner.
- Integrate **official APIs** (where available) and signed **partner data feeds** for reliability and ToS alignment.
- Swap TF-IDF for **embedding search** (BGE-M3/Instructor-XL) + a vector DB (FAISS/Weaviate/Pinecone).
- Add **faceted filters** (price, duration, level, mode, accreditation, start date).
- Build **Institution Dashboards** (search demand, geo heatmaps) and **Sponsored Listings** (no commission model).
"""
    )

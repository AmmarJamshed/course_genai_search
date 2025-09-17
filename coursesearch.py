#!/usr/bin/env python
# coding: utf-8

# Streamlit meta-search that actually scrapes the web for courses/trainings.
# - Discovers links via DuckDuckGo HTML results (no API key)
# - Scrapes each result URL: OpenGraph + JSON-LD (schema.org Course/Product)
# - Heuristically filters to likely "course/training" pages
# - Ranks by keyword coverage; renders rich cards
#
# For higher accuracy, add site-specific adapters or use provider APIs.

import re
import time
import json
import math
import html
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import streamlit as st

# -------------------------------
# Page config & simple CSS
# -------------------------------
st.set_page_config(page_title="Coursemon Web Scraper Search", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.app-title {font-size: 2.0rem; font-weight: 800; margin-bottom: 0.25rem;}
.app-sub {opacity: 0.85; margin-bottom: 1rem;}
.platform-card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 12px;
  background: white;
  box-shadow: 0 3px 10px rgba(0,0,0,0.04);
}
.pill {
  display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #eee;
  margin-right:6px; margin-bottom:6px; font-size:12px; background:#f8fafc;
}
a.card-title { text-decoration: none; }
a.card-title:hover { text-decoration: underline; }
.dim{opacity:0.8}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Helpers & heuristics
# -------------------------------
STOPWORDS = set("""
a an and the to for of on in with or vs from learn course tutorial training class
i me my we our us you your they their them it its is are was were be been being
about into over under by at as how what which who where when why this that these those
weekend weekday evening morning hands-on hands free cheap certificate
""".split())

LEVEL_WORDS = {
    "beginner": {"beginner", "intro", "introduction", "foundations", "starter", "basic", "no prior"},
    "intermediate": {"intermediate", "mid", "some experience"},
    "advanced": {"advanced", "expert", "pro", "senior", "specialist"}
}

MODALITY_WORDS = {
    "online_course": {"online", "self-paced", "recorded", "asynchronous", "course"},
    "online_training": {"live", "instructor-led", "cohort", "bootcamp", "workshop", "training"},
    "onsite_training": {"onsite", "on-site", "in-person", "at office", "physical", "classroom"},
}

COURSE_HINT_WORDS = {
    "course","courses","training","trainings","bootcamp","workshop","class",
    "certificate","certification","syllabus","curriculum","instructor","enroll","enrol","register"
}

# Some common course providers (helps domain inference)
KNOWN_PROVIDERS = {
    "coursera.org":"Coursera",
    "udemy.com":"Udemy",
    "simplivlearning.com":"Simpliv",
    "educative.io":"Educative",
    "edx.org":"edX",
    "udacity.com":"Udacity",
    "datacamp.com":"DataCamp",
    "skillshare.com":"Skillshare",
    "linkedin.com":"LinkedIn Learning",
    "pluralsight.com":"Pluralsight",
    "khanacademy.org":"Khan Academy",
    "futurelearn.com":"FutureLearn",
    "alison.com":"Alison"
}

@dataclass
class ParsedIntent:
    tokens: List[str]
    level: Optional[str]
    modality: Optional[str]
    needs_certificate: bool
    short_format: bool
    free_preferred: bool

def tokenize(q: str, top_k: int = 12) -> List[str]:
    q = q.lower()
    toks = re.findall(r"[a-z0-9\-\+/#]+", q)
    out, seen = [], set()
    for t in toks:
        if t in STOPWORDS or t.isdigit() or len(t) <= 2:
            continue
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= top_k: break
    if not out:
        out = ["data","science","python","ai"]
    return out

def parse_intent(user_text: str,
                 level_pref: str,
                 modality_radio: str,
                 must_cert: bool,
                 short_crash: bool,
                 free_only: bool) -> ParsedIntent:
    tokens = tokenize(user_text)
    # infer level from text if Any not chosen
    level = None
    if level_pref != "Any":
        level = level_pref.lower()
    else:
        tl = user_text.lower()
        for lvl, words in LEVEL_WORDS.items():
            if any(w in tl for w in words):
                level = lvl; break

    # modality from radio
    m = {
        "Online Courses":"online_course",
        "Online Trainings (Simpliv)":"online_training",
        "Onsite Trainings (coming soon)":"onsite_training",
        "Mixed / Any":"any"
    }[modality_radio]
    modality = None if m == "any" else m

    return ParsedIntent(tokens, level, modality, must_cert, short_crash, free_only)

def build_queries(pi: ParsedIntent, user_text: str) -> List[str]:
    base = " ".join(pi.tokens)
    variants = [user_text.strip(), base]
    if pi.level: variants.append(base + f" {pi.level}")
    if pi.modality == "online_training":
        variants.append(base + " live instructor-led workshop")
    elif pi.modality == "online_course":
        variants.append(base + " online self-paced")
    elif pi.modality == "onsite_training":
        variants.append(base + " onsite in-person")
    seen, out = set(), []
    for v in variants:
        v2 = v.lower().strip()
        if v2 and v2 not in seen:
            seen.add(v2); out.append(v.strip())
    return out[:4]

# -------------------------------
# DuckDuckGo discovery
# -------------------------------
HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"}

def ddg_search(query: str, max_results: int = 12) -> List[str]:
    """
    Simple HTML search on DuckDuckGo (no API key).
    """
    url = "https://html.duckduckgo.com/html/"
    try:
        r = requests.post(url, data={"q": query}, headers=HEADERS, timeout=15)
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.select("a.result__a"):
        href = a.get("href")
        if not href: continue
        # Resolve redirect-style links
        out.append(href)
        if len(out) >= max_results: break
    return out

# -------------------------------
# Page scraping & metadata extraction
# -------------------------------
def fetch(url: str, timeout=20) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and "text/html" in (r.headers.get("Content-Type","")):
            return r.text
    except requests.RequestException:
        return None
    return None

def parse_jsonld(soup: BeautifulSoup) -> List[dict]:
    data = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            j = json.loads(tag.string or "")
            if isinstance(j, list): data += j
            elif isinstance(j, dict): data.append(j)
        except Exception:
            continue
    return data

def og(soup: BeautifulSoup, prop: str) -> Optional[str]:
    tag = soup.find("meta", property=f"og:{prop}")
    if tag and tag.get("content"): return tag["content"].strip()
    return None

def text_fallback(soup: BeautifulSoup) -> str:
    # use first paragraph as fallback description
    p = soup.find("p")
    if p: return p.get_text(" ", strip=True)[:300]
    return ""

def looks_like_course(meta: Dict[str, Any], url: str, page_text: str) -> bool:
    """
    Heuristic gate: decide whether a URL is likely a course/training page.
    """
    title = (meta.get("title") or "").lower()
    desc = (meta.get("description") or "").lower()
    u = url.lower()

    # Strong signals: schema.org Course/Product with educational context
    if meta.get("schema_type") in {"Course","CreativeWork","Product"}:
        if any(w in (title+desc) for w in COURSE_HINT_WORDS):
            return True

    # URL path hints
    if any(p in u for p in ["/course", "/courses", "/training", "/bootcamp", "/workshop", "/learn/"]):
        return True

    # Content hints
    if sum(1 for w in COURSE_HINT_WORDS if w in (title+desc)) >= 2:
        return True

    # Last resort: page text (cheap)
    txt = page_text.lower()
    if sum(1 for w in COURSE_HINT_WORDS if w in txt) >= 3:
        return True

    return False

def extract_metadata(url: str) -> Optional[Dict[str, Any]]:
    html_text = fetch(url)
    if not html_text: return None

    soup = BeautifulSoup(html_text, "html.parser")

    # OpenGraph basics
    title = og(soup, "title") or (soup.title.get_text(strip=True) if soup.title else None)
    desc = og(soup, "description")
    site_name = og(soup, "site_name")

    # JSON-LD details
    schemas = parse_jsonld(soup)
    schema_type = None
    price = None
    currency = None
    rating = None
    provider = None
    duration = None
    cert = None

    for j in schemas:
        t = (j.get("@type") if isinstance(j, dict) else None)
        if isinstance(t, list):
            t = next((x for x in t if isinstance(x, str)), None)
        if not t: continue

        if not schema_type and isinstance(t, str):
            schema_type = t

        # Course-ish fields
        prov = j.get("provider") or j.get("providerName") or j.get("brand")
        if isinstance(prov, dict):
            provider = prov.get("name") or provider
        elif isinstance(prov, str):
            provider = prov

        # Offer/price
        offers = j.get("offers")
        if isinstance(offers, dict):
            price = offers.get("price") or price
            currency = offers.get("priceCurrency") or currency
        elif isinstance(offers, list):
            for o in offers:
                if isinstance(o, dict):
                    price = o.get("price") or price
                    currency = o.get("priceCurrency") or currency

        # AggregateRating
        ar = j.get("aggregateRating")
        if isinstance(ar, dict):
            try:
                rating = float(ar.get("ratingValue"))
            except Exception:
                pass

        # Duration
        duration = duration or j.get("timeRequired") or j.get("duration")

        # Certificate hint
        cert = cert or j.get("educationalCredentialAwarded") or j.get("award")

        # Description/title fallback
        if not desc:
            d = j.get("description")
            if isinstance(d, str): desc = d
        if not title:
            t2 = j.get("name") or j.get("headline")
            if isinstance(t2, str): title = t2

    # Fallback description
    if not desc:
        desc = text_fallback(soup)

    # Provider from domain (fallback)
    if not provider:
        host = urlparse(url).netloc
        provider = KNOWN_PROVIDERS.get(host.replace("www.",""), host)

    # Minimal meta
    meta = {
        "title": title or "",
        "description": desc or "",
        "url": url,
        "provider": provider,
        "schema_type": schema_type,
        "price": price,
        "currency": currency,
        "rating": rating,
        "duration": duration,
        "certificate_flag": bool(cert),
    }

    # Gate by heuristics
    if not looks_like_course(meta, url, soup.get_text(" ", strip=True)[:4000]):
        return None

    return meta

# -------------------------------
# Ranking / scoring
# -------------------------------
def score_item(meta: Dict[str, Any], tokens: List[str], pi: ParsedIntent) -> float:
    title = (meta.get("title") or "").lower()
    desc  = (meta.get("description") or "").lower()

    # token coverage
    t_hits = sum(1 for t in tokens if t in title)
    d_hits = sum(1 for t in tokens if t in desc)

    # small boosts for visible metadata
    boost = 0.0
    if pi.level and pi.level in (title+desc): boost += 0.2
    if pi.modality and any(w in (title+desc) for w in MODALITY_WORDS.get(pi.modality, [])): boost += 0.2
    if pi.needs_certificate and ("certificate" in (title+desc) or meta.get("certificate_flag")): boost += 0.2
    if meta.get("rating"):
        r = float(meta["rating"])
        boost += min(0.3, (r-3.5)*0.1)  # tiny increase if rating is high

    base = 0.6*min(1.0, (t_hits*0.6 + d_hits*0.2)) + 0.4*(1.0 if t_hits>0 else 0.0)
    return base + boost

def fmt_price(meta):
    p = meta.get("price")
    if p is None or p=="":
        return "—"
    c = (meta.get("currency") or "").upper() or "USD"
    try:
        p_float = float(p)
        if p_float == 0:
            return "Free"
        return f"{c} {p_float:,.0f}"
    except Exception:
        return f"{c} {p}"

# -------------------------------
# UI
# -------------------------------
st.markdown('<div class="app-title">🎓 Coursemon Meta-Search (Web Scraper)</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Type what you want. I’ll discover matching trainings/courses across the web and render rich cards.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Filters")
    training_mode = st.radio(
        "Mode",
        ["Online Courses","Online Trainings (Simpliv)","Onsite Trainings (coming soon)","Mixed / Any"],
        index=0
    )
    st.caption("Optional tags")
    level_pref = st.selectbox("Level", ["Any","Beginner","Intermediate","Advanced"], index=0)
    must_have_cert = st.checkbox("Must offer certificate", value=False)
    short_format = st.checkbox("Short format / crash course", value=False)
    free_only = st.checkbox("Prefer free options", value=False)

user_text = st.text_area(
    "Describe what you’re looking for",
    value="data science for beginners certificate",
    height=90,
    placeholder="e.g., Weekend live bootcamp on Python for finance, beginner, certificate"
)
go = st.button("Find Recommendations 🚀")

if go and user_text.strip():
    pi = parse_intent(user_text, level_pref, training_mode, must_have_cert, short_format, free_only)
    queries = build_queries(pi, user_text)

    # Summary chips
    chips = []
    if pi.level: chips.append(("Level", pi.level.title()))
    if pi.modality: chips.append(("Modality", pi.modality.replace("_"," ").title()))
    if must_have_cert: chips.append(("Certificate", "Required"))
    if short_format: chips.append(("Duration", "Short / Crash"))
    if free_only: chips.append(("Price", "Prefer Free"))

    st.markdown("### Your Search Summary")
    if chips:
        for k,v in chips:
            st.markdown(f'<span class="pill"><b>{k}:</b> {v}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill">Free-form search</span>', unsafe_allow_html=True)

    st.markdown("### Collecting results from the web…")

    all_urls = []
    for q in queries:
        with st.status(f"Searching: *{q}*", expanded=False) as status:
            urls = ddg_search(q + " site:coursera.org OR site:udemy.com OR site:simplivlearning.com "
                                 "OR site:educative.io OR site:edx.org OR site:udacity.com "
                                 "OR site:datacamp.com OR site:skillshare.com OR site:futurelearn.com OR site:alison.com "
                                 "OR course training bootcamp workshop")
            status.update(label=f"Found {len(urls)} links for query: {q}")
            all_urls.extend(urls)
            time.sleep(0.5)

    # Deduplicate
    normed = {}
    for u in all_urls:
        try:
            nu = requests.utils.requote_uri(u.strip())
        except Exception:
            nu = u.strip()
        if nu not in normed:
            normed[nu] = True
    all_urls = list(normed.keys())

    st.write(f"Discovered **{len(all_urls)}** candidate links. Scraping for details…")

    results: List[Dict[str,Any]] = []
    progress = st.progress(0)
    for idx, url in enumerate(all_urls, start=1):
        meta = extract_metadata(url)
        if meta:
            # Hard filters (best-effort)
            if pi.level and pi.level not in (meta["title"].lower()+meta["description"].lower()):
                # allow if provider is well-known (loosen a bit)
                pass
            if must_have_cert and not (meta.get("certificate_flag") or "certificate" in (meta["title"]+meta["description"]).lower()):
                # still allow if provider is well-known cert issuer
                if meta.get("provider","").lower() not in ["coursera","edx","udacity","alison","datacamp","udemy","simpliv","educative"]:
                    progress.progress(min(1.0, idx/len(all_urls))); continue
            if free_only:
                # heuristic: price "Free" OR words in description
                price_txt = fmt_price(meta)
                if price_txt != "Free" and "free" not in (meta["title"]+meta["description"]).lower():
                    progress.progress(min(1.0, idx/len(all_urls))); continue

            meta["__score"] = score_item(meta, pi.tokens, pi)
            results.append(meta)
        progress.progress(min(1.0, idx/len(all_urls)))

    if not results:
        st.warning("No course pages matched. Try relaxing filters or widening the description.")
    else:
        # Rank + cut
        results.sort(key=lambda x: x["__score"], reverse=True)
        topn = results[:40]

        st.markdown("### Results")
        st.caption("Ranked by keyword coverage + metadata signals. Click the title to open the course page.")
        for item in topn:
            with st.container():
                st.markdown('<div class="platform-card">', unsafe_allow_html=True)
                st.markdown(
                    f'#### <a class="card-title" href="{item["url"]}" target="_blank">{html.escape(item["title"] or "(Untitled)")}</a>',
                    unsafe_allow_html=True
                )

                # badges
                chips = []
                prov = item.get("provider")
                if prov: chips.append(f'<span class="pill">{html.escape(prov)}</span>')
                if pi.level: chips.append(f'<span class="pill">{pi.level.title()}</span>')
                if pi.modality: chips.append(f'<span class="pill">{pi.modality.replace("_"," ").title()}</span>')
                if item.get("certificate_flag"): chips.append('<span class="pill">Certificate</span>')
                price_txt = fmt_price(item)
                if price_txt: chips.append(f'<span class="pill">{price_txt}</span>')
                if item.get("rating"):
                    chips.append(f'<span class="pill">⭐ {item["rating"]:.1f}</span>')
                st.markdown(" ".join(chips), unsafe_allow_html=True)

                with st.expander("Overview"):
                    st.write(item.get("description") or "No description available.")
                    if item.get("duration"):
                        st.caption(f"Duration: {item['duration']}")

                st.markdown('</div>', unsafe_allow_html=True)

elif go:
    st.warning("Please type what you’re looking for.")

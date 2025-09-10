#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
# GenAI-free meta search for courses & trainings (Coursera, Udemy, Simpliv, Educative)
# No external AI libs needed. Uses a lightweight keyword extractor and builds deep links.

import re
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st

# -------------------------------
# Page config & styling
# -------------------------------
st.set_page_config(
    page_title="Course & Training Meta-Search",
    page_icon="🎓",
    layout="wide",
)

def inject_style():
    st.markdown(
        """
        <style>
        .app-title {font-size: 2.0rem; font-weight: 800; margin-bottom: 0.25rem;}
        .app-sub {opacity: 0.85; margin-bottom: 1rem;}
        .platform-card {
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 10px;
            background: white;
            box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        }
        .pill {
            display:inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid #eee;
            margin-right: 6px;
            margin-bottom: 6px;
            font-size: 12px;
            background: #f8fafc;
        }
        .dim { opacity: 0.7; }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_style()

# -------------------------------
# Simple intent container (no LLM)
# -------------------------------
@dataclass
class ParsedIntent:
    topic: str
    subtopics: List[str]
    skills: List[str]
    level: Optional[str]      # beginner/intermediate/advanced/any
    modality: Optional[str]   # online_course/online_training/onsite_training/any
    extras: Dict[str, str]
    keywords: List[str]

# -------------------------------
# Lightweight keyword extraction
# -------------------------------
STOPWORDS = set("""
a an the to for of on in with and or vs from learn course tutorial training class
i me my we our us you your they their them it its is are was were be been being
about into over under by at as how what which who where when why
this that those these want need looking seeking search find best top
weekend weekday evening morning hands-on hands free cheap certificate
""".split())

LEVEL_WORDS = {
    "beginner": {"beginner", "intro", "introduction", "foundations", "starter", "basic"},
    "intermediate": {"intermediate", "mid", "some experience"},
    "advanced": {"advanced", "expert", "pro", "senior", "specialist"}
}

MODALITY_WORDS = {
    "online_course": {"online", "self-paced", "recorded", "asynchronous", "course"},
    "online_training": {"live", "instructor-led", "cohort", "bootcamp", "workshop", "training"},
    "onsite_training": {"onsite", "on-site", "in-person", "at office", "physical", "classroom"},
}

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_level(text: str) -> Optional[str]:
    tl = text.lower()
    for lvl, words in LEVEL_WORDS.items():
        if any(w in tl for w in words):
            return lvl
    return None

def extract_modality(text: str) -> Optional[str]:
    tl = text.lower()
    for mod, words in MODALITY_WORDS.items():
        if any(w in tl for w in words):
            return mod
    return None

def extract_keywords(text: str, top_k: int = 8) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9\-\+/#]+", text.lower())
    picked, seen = [], set()
    for t in tokens:
        if t in STOPWORDS: 
            continue
        if t.isdigit():
            continue
        if len(t) <= 2:
            continue
        if t not in seen:
            seen.add(t)
            picked.append(t)
        if len(picked) >= top_k:
            break
    # sensible defaults if user is vague
    if not picked:
        picked = ["data", "analytics", "python", "ai"]
    return picked

def guess_topic_and_skills(keywords: List[str]) -> (str, List[str]):
    if not keywords:
        return "", []
    # Heuristic: first 1–2 words as topic head, rest as skills
    topic = " ".join(keywords[:2]).title()
    skills = keywords[2:6]
    return topic, skills

def parse_query_noai(user_text: str) -> ParsedIntent:
    user_text = normalize_text(user_text)
    kws = extract_keywords(user_text, top_k=10)
    topic, skills = guess_topic_and_skills(kws)
    level = extract_level(user_text)
    modality = extract_modality(user_text)
    extras = {}

    # additional hints
    if "certificate" in user_text.lower():
        extras["certificate"] = "required"
    if any(w in user_text.lower() for w in ["short", "crash", "weekend"]):
        extras["duration"] = "short"
    if any(w in user_text.lower() for w in ["free", "no cost", "open"]):
        extras["price"] = "free_preferred"

    return ParsedIntent(
        topic=topic,
        subtopics=[],
        skills=skills,
        level=level,
        modality=modality,
        extras=extras,
        keywords=kws
    )

# -------------------------------
# Deep-link builders
# -------------------------------
def qjoin(parts: List[str]) -> str:
    clean = [p for p in parts if p]
    return " ".join(clean).strip()

def search_url(platform: str, query: str) -> str:
    q = urllib.parse.quote_plus(query)
    if platform == "Coursera":
        return f"https://www.coursera.org/search?query={q}"
    if platform == "Udemy":
        return f"https://www.udemy.com/courses/search/?q={q}"
    if platform == "Simpliv":
        return f"https://www.simplivlearning.com/search?query={q}"
    if platform == "Educative":
        return f"https://www.educative.io/search?query={q}"
    return "#"

def make_query_variants(pi: ParsedIntent, raw_text: str) -> List[str]:
    base_kw = pi.keywords or extract_keywords(raw_text, top_k=8)
    variants = []
    if pi.topic:
        variants.append(qjoin([pi.topic] + pi.skills))
    variants.append(qjoin(base_kw))
    if pi.level:
        variants.append(qjoin(base_kw + [pi.level]))
    variants.append(raw_text.strip())

    out, seen = [], set()
    for v in variants:
        v2 = v.strip().lower()
        if v2 and v2 not in seen:
            seen.add(v2)
            out.append(v)
    return out[:4]

# -------------------------------
# UI
# -------------------------------
st.markdown('<div class="app-title">🎓 Course & Training Meta-Search (No OpenAI)</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Type anything (e.g., “Hands-on weekend bootcamp on Python for finance with real datasets and a certificate”).</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Filters")
    training_mode = st.radio(
        "Mode",
        ["Online Courses", "Online Trainings (Simpliv)", "Onsite Trainings (coming soon)", "Mixed / Any"],
        index=0
    )
    platforms = st.multiselect(
        "Platforms",
        ["Coursera", "Udemy", "Simpliv", "Educative"],
        default=["Coursera", "Udemy", "Simpliv", "Educative"]
    )
    st.markdown("---")
    st.caption("Optional tags")
    level_pref = st.selectbox("Level", ["Any","Beginner","Intermediate","Advanced"], index=0)
    must_have_cert = st.checkbox("Must offer certificate", value=False)
    short_format = st.checkbox("Short format / crash course", value=False)
    free_only = st.checkbox("Prefer free options", value=False)

user_text = st.text_area(
    "Describe what you’re looking for",
    height=100,
    placeholder="e.g., Weekend crash course on Generative AI for product managers, case studies, no coding, certificate."
)
go = st.button("Find Recommendations 🚀")

mode_to_modality = {
    "Online Courses": "online_course",
    "Online Trainings (Simpliv)": "online_training",
    "Onsite Trainings (coming soon)": "onsite_training",
    "Mixed / Any": "any"
}

if go and user_text.strip():
    parsed = parse_query_noai(user_text)

    # Apply sidebar prefs
    extras = dict(parsed.extras)
    if level_pref != "Any":
        extras["level"] = level_pref.lower()
    if must_have_cert:
        extras["certificate"] = "required"
    if short_format:
        extras["duration"] = "short"
    if free_only:
        extras["price"] = "free_preferred"

    modality = parsed.modality or mode_to_modality.get(training_mode, "any")

    chips = []
    if parsed.topic:
        chips.append(("Topic", parsed.topic))
    if parsed.skills:
        chips.append(("Skills", ", ".join(parsed.skills[:5])))
    if parsed.level:
        chips.append(("Level (auto)", parsed.level))
    if modality and modality != "any":
        chips.append(("Modality", modality.replace("_"," ").title()))
    for k, v in extras.items():
        chips.append((k.title(), v))

    st.markdown("### Your Search Summary")
    if chips:
        for label, val in chips:
            st.markdown(f'<span class="pill"><b>{label}:</b> {val}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill">Free-form search</span>', unsafe_allow_html=True)

    # Add level/modality hints to keywords
    kw = parsed.keywords[:]
    if "level" in extras and extras["level"] not in kw:
        kw.append(extras["level"])
    if modality and modality != "any":
        # modality hint words to try nudging platform search
        if modality == "online_training":
            kw.extend(["live", "instructor-led", "workshop"])
        elif modality == "online_course":
            kw.extend(["online", "self-paced"])
        elif modality == "onsite_training":
            kw.extend(["onsite", "in-person"])

    variants = make_query_variants(ParsedIntent(
        topic=parsed.topic,
        subtopics=[],
        skills=parsed.skills,
        level=parsed.level,
        modality=modality,
        extras=extras,
        keywords=kw
    ), user_text)

    effective_platforms = platforms.copy()
    if training_mode == "Online Trainings (Simpliv)":
        effective_platforms = [p for p in platforms if p == "Simpliv"]
    elif training_mode == "Onsite Trainings (coming soon)":
        st.warning("Onsite trainings directory is coming soon. Showing online alternatives for now.")
        effective_platforms = [p for p in platforms if p in ["Coursera","Udemy","Simpliv","Educative"]]

    st.markdown("### Recommendations & Deep Links")
    st.caption("Click to open results on each platform. Use platform filters for price, duration, language, etc.")

    col1, col2 = st.columns([1,1])
    for i, qv in enumerate(variants, start=1):
        with (col1 if i % 2 else col2):
            st.markdown(
                f'<div class="platform-card"><div class="dim">Query variant {i}</div><h4>{qv}</h4>',
                unsafe_allow_html=True
            )
            links = []
            for pf in effective_platforms:
                links.append(f"- [{pf}]({search_url(pf, qv)})")
            st.markdown("\n".join(links))
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Tips")
    st.markdown(
        """
- Add specific **tools** (e.g., “YOLOv8, PyTorch, OpenCV”) or **domain** (“for agriculture”) to sharpen results.
- Switch **Mode** to focus on Simpliv live trainings or keep it mixed for broader discovery.
- Use platform filters to refine by **price**, **duration**, **language**, **ratings**, etc.
        """
    )
elif go:
    st.warning("Please type what you’re looking for (one sentence is fine).")

st.markdown("---")
st.caption("No external AI used. This app generates clean search links based on your text.")


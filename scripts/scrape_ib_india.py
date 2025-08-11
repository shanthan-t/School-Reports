#!/usr/bin/env python3
# scripts/scrape_ib_india.py

"""
Scrape IB 'Find an IB World School' for India, geocode, and write data/schools.csv.

Workflow:
1) Try to discover school profile links via the site's sitemap (stable, fast).
2) If that fails, fall back to driving the public search UI with Playwright.
3) Parse each profile page to extract name, address, programmes, website.
4) Geocode addresses (OpenCage if OPENCAGE_API_KEY set, else Nominatim).
5) Save:
   - data/schools_raw.json  (parsed, ungeocoded details)
   - data/schools.csv       (final for your app)
   - data/cache/geocode_cache.csv  (to avoid re-querying geocoders)
"""

from __future__ import annotations

import os
import re
import csv
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# Geocoding
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

try:
    from opencage.geocoder import OpenCageGeocode  # optional
    HAS_OPENCAGE = True
except Exception:
    HAS_OPENCAGE = False

# Playwright (fallback path)
from playwright.sync_api import sync_playwright

# ----------------- Constants & paths -----------------
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
RAW_JSON = DATA_DIR / "schools_raw.json"
CSV_PATH = DATA_DIR / "schools.csv"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROGRAMMES = ["PYP", "MYP", "DP", "CP"]

# Toggle while debugging the Playwright flow
PW_HEADLESS = False      # set to True after first successful run
PW_SLOW_MO_MS = 200      # slows actions so you can see what's happening


# ----------------- Data model -----------------
@dataclass
class School:
    name: str
    city: Optional[str]
    state: Optional[str]
    country: str
    address: Optional[str]
    programmes: List[str]
    website: Optional[str]
    profile_url: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    region: Optional[str] = None

    def to_row(self) -> Dict[str, Optional[str]]:
        return {
            "name": self.name,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "address": self.address,
            "programmes": ",".join(self.programmes),
            "website": self.website,
            "profile_url": self.profile_url,
            "lat": self.lat,
            "lng": self.lng,
            "region": self.region,
        }


# ----------------- Helpers -----------------
def extract_city_state(address_text: str) -> (Optional[str], Optional[str]):
    """Very simple heuristic for Indian addresses like '..., City, State PIN, India'."""
    if not address_text:
        return None, None
    parts = [p.strip() for p in address_text.split(",") if p.strip()]
    # drop trailing "India"
    if parts and parts[-1].lower().startswith("india"):
        parts = parts[:-1]
    # last token may be state (sometimes includes PIN -> strip digits)
    city = parts[-2] if len(parts) >= 2 else (parts[-1] if parts else None)
    state = parts[-1] if parts else None
    if state and any(ch.isdigit() for ch in state):
        state = parts[-2] if len(parts) >= 2 else None
    return city, state


def parse_programmes(text: str) -> List[str]:
    text_u = text.upper()
    found = [p for p in PROGRAMMES if p in text_u]
    return sorted(set(found))


def load_geocode_cache() -> Dict[str, Dict[str, float]]:
    path = CACHE_DIR / "geocode_cache.csv"
    cache: Dict[str, Dict[str, float]] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cache[row["query"]] = {"lat": float(row["lat"]), "lng": float(row["lng"])}
    return cache


def save_geocode_cache(cache: Dict[str, Dict[str, float]]) -> None:
    path = CACHE_DIR / "geocode_cache.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query", "lat", "lng"])
        w.writeheader()
        for q, v in cache.items():
            w.writerow({"query": q, "lat": v["lat"], "lng": v["lng"]})


def geocode_all(schools: List[School]) -> List[School]:
    cache = load_geocode_cache()

    oc_key = os.getenv("OPENCAGE_API_KEY")
    use_opencage = bool(oc_key) and HAS_OPENCAGE

    if use_opencage:
        oc = OpenCageGeocode(oc_key)  # type: ignore
    else:
        geo = Nominatim(user_agent="ib-schools-india")
        geocode = RateLimiter(geo.geocode, min_delay_seconds=1)

    for s in tqdm(schools, desc="Geocoding", unit="school"):
        query = s.address or f"{s.name}, {s.city or ''}, {s.state or ''}, India"
        query = " ".join(query.split())

        if query in cache:
            coords = cache[query]
            s.lat, s.lng = coords["lat"], coords["lng"]
            continue

        try:
            if use_opencage:
                res = oc.geocode(query, countrycode="in", limit=1, no_annotations=1)
                if res:
                    s.lat = float(res[0]["geometry"]["lat"])
                    s.lng = float(res[0]["geometry"]["lng"])
            else:
                loc = geocode(query)  # type: ignore
                if loc:
                    s.lat = float(loc.latitude)
                    s.lng = float(loc.longitude)

            if s.lat is not None and s.lng is not None:
                cache[query] = {"lat": s.lat, "lng": s.lng}
        except Exception:
            # swallow and move on; we'll leave lat/lng empty
            continue

    save_geocode_cache(cache)
    return schools


def add_region(state: Optional[str]) -> Optional[str]:
    north = {"Jammu and Kashmir","Himachal Pradesh","Punjab","Haryana","Delhi","Uttarakhand","Uttar Pradesh","Rajasthan","Chandigarh"}
    west = {"Maharashtra","Gujarat","Goa","Dadra and Nagar Haveli and Daman and Diu"}
    south = {"Karnataka","Kerala","Tamil Nadu","Telangana","Andhra Pradesh","Puducherry"}
    east = {"West Bengal","Odisha","Bihar","Jharkhand"}
    northeast = {"Assam","Meghalaya","Manipur","Mizoram","Nagaland","Tripura","Arunachal Pradesh","Sikkim"}

    st = (state or "").strip()
    if st in south: return "South"
    if st in west: return "West"
    if st in north: return "North"
    if st in east: return "East"
    if st in northeast: return "North-East"
    return None


# ----------------- Source 1: sitemap -----------------
def fetch_profile_links_from_sitemap() -> List[str]:
    """
    Try common sitemap endpoints; collect any URLs that look like school profiles.
    """
    candidates = [
        "https://www.ibo.org/sitemap.xml",
        "https://www.ibo.org/sitemap_index.xml",
    ]
    seen_xml = set()
    profile_links: set[str] = set()

    def fetch_xml(url: str) -> Optional[BeautifulSoup]:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                return None
            return BeautifulSoup(r.content, "xml")
        except Exception:
            return None

    queue = candidates[:]
    while queue:
        url = queue.pop(0)
        if url in seen_xml:
            continue
        seen_xml.add(url)
        doc = fetch_xml(url)
        if not doc:
            continue

        # Add nested sitemaps
        for loc in doc.find_all("loc"):
            link = (loc.text or "").strip()
            if not link:
                continue
            if link.endswith(".xml"):
                queue.append(link)
            if "/programmes/find-an-ib-school/school/" in link or "/school/" in link:
                profile_links.add(link)

    return sorted(profile_links)


def parse_profile_page(url: str) -> Optional[School]:
    try:
        html = requests.get(url, timeout=25).text
        soup = BeautifulSoup(html, "html.parser")

        # Name
        name_el = soup.select_one("h1, .c-hero__heading, .school-name")
        name = name_el.get_text(" ", strip=True) if name_el else "Unknown"

        # Address
        addr_el = soup.select_one("[itemprop='address'], .c-contact__address, .address, .school-address")
        address = addr_el.get_text(" ", strip=True) if addr_el else ""

        # Filter: we only want India
        if "india" not in (address or "").lower() and " india" not in soup.get_text(" ", strip=True).lower():
            return None

        # Programmes from text
        all_text = soup.get_text(" ", strip=True).upper()
        programmes = [p for p in PROGRAMMES if p in all_text]

        # External website (first non-IBO HTTP link)
        website = None
        for a in soup.find_all("a", href=True):
            u = a["href"]
            if u.startswith("http") and "ibo.org" not in u:
                website = u
                break

        city, state = extract_city_state(address)
        return School(
            name=name,
            city=city,
            state=state,
            country="India",
            address=address,
            programmes=programmes,
            website=website,
            profile_url=url,
        )
    except Exception:
        return None


# ----------------- Source 2: Playwright (UI) -----------------
def scrape_india_via_ui() -> List[School]:
    """Drive the public search UI with Playwright; resilient to cookie banners."""
    schools: List[School] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=PW_HEADLESS, slow_mo=PW_SLOW_MO_MS)
        page = browser.new_page()
        page.set_default_timeout(20000)

        page.goto("https://www.ibo.org/programmes/find-an-ib-school/")
        page.wait_for_load_state("domcontentloaded")

        # --- Cookie/consent banners (try a few patterns) ---
        tried = False
        for _ in range(3):
            tried = True
            # OneTrust standard id
            try:
                page.locator("#onetrust-accept-btn-handler").click(timeout=3000)
            except Exception:
                pass
            # Generic buttons
            try:
                page.get_by_role("button", name=re.compile(r"(Accept|I agree|Allow all)", re.I)).first.click(timeout=3000)
            except Exception:
                pass
            # Iframe-based banners (common)
            try:
                fl = page.frame_locator("iframe[id^='sp_message_iframe'], iframe[title*='privacy'], iframe[id^='ot-']")
                fl.get_by_text(re.compile(r"Accept|Agree|Allow", re.I)).click(timeout=3000)
            except Exception:
                pass

        # --- Select Country = India ---
        picked = False
        try:
            page.get_by_label(re.compile("Country|Country or territory", re.I)).click()
            page.keyboard.type("India")
            page.keyboard.press("Enter")
            picked = True
        except Exception:
            pass
        if not picked:
            try:
                dd = page.locator("select[name*='country'], [role='combobox']")
                if dd.count() > 0:
                    dd.first.click()
                    page.keyboard.type("India")
                    page.keyboard.press("Enter")
                    picked = True
            except Exception:
                pass

        # --- Search / Apply button if needed ---
        try:
            page.get_by_role("button", name=re.compile("Search|Apply|Find", re.I)).first.click(timeout=6000)
        except Exception:
            pass

        # Wait & scroll to load cards; then harvest profile links
        page.wait_for_load_state("networkidle")
        prev = -1
        for _ in range(35):
            links_now = page.locator("a[href*='/programmes/find-an-ib-school/school/'], a[href*='/school/']")
            count = links_now.count()
            if count == prev:
                break
            prev = count
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(700)

        anchors = page.locator("a[href*='/programmes/find-an-ib-school/school/'], a[href*='/school/']").all()
        profile_links = set()
        for a in anchors:
            href = a.get_attribute("href")
            if href and "/school/" in href:
                if href.startswith("/"):
                    href = "https://www.ibo.org" + href
                profile_links.add(href)

        print(f"Found {len(profile_links)} school profile links via UI")

        # Visit each profile and parse
        for href in sorted(profile_links):
            s = parse_profile_page(href)
            if s is not None:
                schools.append(s)

        # Optional debug artifacts
        if not schools:
            try:
                page.screenshot(path="data/debug_find_ib.png", full_page=True)
                Path("data/page_debug.html").write_text(page.content(), encoding="utf-8")
                print("Saved data/debug_find_ib.png and data/page_debug.html for debugging.")
            except Exception:
                pass

        browser.close()

    return schools


# ----------------- Main -----------------
def main():
    print("Trying sitemap first…")
    links = fetch_profile_links_from_sitemap()
    schools: List[School] = []

    if links:
        print(f"Found {len(links)} profile links in sitemap. Parsing pages…")
        for href in tqdm(links, desc="Parsing profiles", unit="page"):
            s = parse_profile_page(href)
            if s is not None:
                schools.append(s)
        # keep only India schools (already filtered, but double-guard)
        schools = [s for s in schools if (s.country == "India")]
    else:
        print("Sitemap path yielded no usable links. Falling back to browser automation…")
        schools = scrape_india_via_ui()

    if not schools:
        print("No schools parsed. Aborting.")
        return

    # Save raw (pre-geocode) for audit/repro
    RAW_JSON.write_text(json.dumps([asdict(s) for s in schools], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved raw to {RAW_JSON}")

    # Geocode
    print("Geocoding…")
    geocode_all(schools)

    # Finalize dataframe
    rows = []
    for s in schools:
        s.region = add_region(s.state)
        rows.append(s.to_row())
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["name", "city"]).reset_index(drop=True)

    # Reorder columns for your app
    cols = ["name","lat","lng","city","state","region","programmes","website","address","country","profile_url"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols]

    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Wrote {CSV_PATH} with {len(df)} rows")


if __name__ == "__main__":
    main()

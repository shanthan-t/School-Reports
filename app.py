import base64
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt

# Optional geocoding for user-entered address
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ---------- Page setup ----------
st.set_page_config(page_title="IB Schools in India", layout="wide")
st.markdown("""
<style>
.main > div { padding-top: 1rem; }
.header { font-size: 1.6rem; font-weight: 700; margin-bottom: .5rem; }
.subtle { color: #667085; }
</style>
""", unsafe_allow_html=True)

# ---------- Seed data (only used as fallback on first run) ----------
schools = [
    {"name":"Oakridge International School", "lat":12.8874546, "lng":77.7525313, "image":"images/oakridge.jpg", "city":"Bengaluru", "state":"Karnataka", "region":"South"},
    {"name":"The CHIREC International School", "lat":17.466031647693534, "lng":78.35534812700402, "image":"images/chirec.jpg", "city":"Hyderabad", "state":"Telangana", "region":"South"},
    {"name":"The Aga Khan Academy", "lat":17.246731921803573, "lng":78.48058103079558, "image":"images/aga_khan.jpg", "city":"Hyderabad", "state":"Telangana", "region":"South"},
    {"name":"Indus International School", "lat":17.450084607132062, "lng":78.17842338655431, "image":"images/indus.jpg", "city":"Hyderabad", "state":"Telangana", "region":"South"},
    {"name":"GD Goenka School", "lat":17.36170946076783, "lng":78.56383857308079, "image":"images/goenka.jpg", "city":"Hyderabad", "state":"Telangana", "region":"South"},
    {"name":"NIHOC The International School", "lat":17.396194814998562, "lng":78.62564163110986, "image":"images/nihoc.jpg", "city":"Hyderabad", "state":"Telangana", "region":"South"},
    {"name":"Sancta Maria School", "lat":17.607169462355436, "lng":78.40693523875808, "image":"images/sancta_maria.jpg", "city":"Hyderabad", "state":"Telangana", "region":"South"},
    {"name":"International School of Hyderabad", "lat":17.542697883238592, "lng":78.28683287949816, "image":"images/ish.jpg", "city":"Hyderabad", "state":"Telangana", "region":"South"},
    {"name":"Manthan International School", "lat":17.460292395334818, "lng":78.29846158995463, "image":"images/manthan.jpg", "city":"Hyderabad", "state":"Telangana", "region":"South"},
    {"name":"Rishi Valley School", "lat":13.63638709376012, "lng":78.45389756450555, "image":"images/rishi_valley.jpg", "city":"Chittoor", "state":"Andhra Pradesh", "region":"South"},
    {"name":"Ambitus World School", "lat":16.542942593949597, "lng":80.6723706311272, "image":"images/ambitus.jpg", "city":"Vijayawada", "state":"Andhra Pradesh", "region":"South"},
    {"name":"The Vizag International School", "lat":18.064579966081098, "lng":83.33420476163535, "image":"images/vizag.jpg", "city":"Visakhapatnam", "state":"Andhra Pradesh", "region":"South"},
    {"name":"Candor International School", "lat":17.32121987674343, "lng":78.62425640188032, "image":"images/candor.jpg", "city":"Hyderabad", "state":"Telangana", "region":"South"},
    {"name":"The Green School", "lat":12.94563760981085, "lng":77.78634957552616, "image":"images/green.jpg", "city":"Bengaluru", "state":"Karnataka", "region":"South"},
    {"name":"Inventure Academy", "lat":12.895413764458958, "lng":77.74419758962246, "image":"images/inventure.jpg", "city":"Bengaluru", "state":"Karnataka", "region":"South"},
    {"name":"Pathways World School", "lat":28.312711895906258, "lng":77.01278390505134, "image":"images/pathways.jpg", "city":"Gurugram", "state":"Haryana", "region":"North"},
    {"name":"Ecole Mondiale World School", "lat":19.113143387849235, "lng":72.83429774545095, "image":"images/ecole_mondiale.jpg", "city":"Mumbai", "state":"Maharashtra", "region":"West"},
    {"name":"Tridha", "lat":19.122906737081873, "lng":72.85904558978828, "image":"images/tridha.jpg", "city":"Mumbai", "state":"Maharashtra", "region":"West"},
    {"name":"Woodstock School", "lat":30.453965220447383, "lng":78.10085140517408, "image":"images/woodstock.jpg", "city":"Mussoorie", "state":"Uttarakhand", "region":"North"},
    {"name":"Alpha International School", "lat":12.924296953124166, "lng":80.15830999869962, "image":"images/alpha.jpg", "city":"Chennai", "state":"Tamil Nadu", "region":"South"},
    {"name":"Shiv Nadar School", "lat":13.005156533030055, "lng":80.26326775927699, "image":"images/shiv_nadar.jpg", "city":"Chennai", "state":"Tamil Nadu", "region":"South"},
    {"name":"Ørestad High School", "lat":55.63183195977894, "lng":12.58110280982054, "image":"images/orestad.jpg", "city":"Copenhagen", "state":"Outside India", "region":"International"},
]
SEED_DF = pd.DataFrame(schools)

# ---------- Utilities ----------
def img_to_data_uri(path: str) -> str:
    """Embed local images so Folium can render them."""
    p = Path(path)
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    mime = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64}"

def add_region_from_state(state: str | None) -> str | None:
    north = {"Jammu and Kashmir","Himachal Pradesh","Punjab","Haryana","Delhi","Uttarakhand","Uttar Pradesh","Rajasthan","Chandigarh"}
    west = {"Maharashtra","Gujarat","Goa","Dadra and Nagar Haveli and Daman and Diu"}
    south = {"Karnataka","Kerala","Tamil Nadu","Telangana","Andhra Pradesh","Puducherry"}
    east = {"West Bengal","Odisha","Bihar","Jharkhand"}
    northeast = {"Assam","Meghalaya","Manipur","Mizoram","Nagaland","Tripura","Arunachal Pradesh","Sikkim"}
    stt = (state or "").strip()
    if stt in south: return "South"
    if stt in west: return "West"
    if stt in north: return "North"
    if stt in east: return "East"
    if stt in northeast: return "North-East"
    return None

# ---------- Data loading (CSV with fallback) ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = Path("data/schools.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # normalize columns
        for c in ["name","city","state","region","programmes","website","address","country","profile_url"]:
            if c not in df.columns:
                df[c] = pd.NA
        # numeric coords
        df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
        df["lng"] = pd.to_numeric(df.get("lng"), errors="coerce")
        # fill region if missing
        if "region" in df.columns:
            df["region"] = df["region"].where(df["region"].notna(), df["state"].map(add_region_from_state))
        else:
            df["region"] = df["state"].map(add_region_from_state)
        return df
    # fallback to seed
    return SEED_DF.copy()

df = load_data()

# ---------- Scoring utilities ----------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points in km."""
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 6371.0088
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def score_row(row, origin, prog_weights, region_boost, dist_weight_per_km):
    """
    origin: (lat, lng) or None
    prog_weights: dict like {"PYP":10, "MYP":15, "DP":25, "CP":10}
    region_boost: set of regions that get a boost
    dist_weight_per_km: positive number; score -= dist * weight
    """
    score = 0.0
    # programme contribution
    progs = []
    if isinstance(row.get("programmes"), str):
        progs = [p.strip().upper() for p in row["programmes"].split(",") if p.strip()]
    for p in progs:
        score += prog_weights.get(p, 0)

    # region boost
    if isinstance(row.get("region"), str) and row["region"] in region_boost:
        score += 10.0

    # distance penalty
    if origin is not None and pd.notna(row.get("lat")) and pd.notna(row.get("lng")):
        d = haversine_km(origin[0], origin[1], row["lat"], row["lng"])
        if not np.isnan(d):
            score -= d * dist_weight_per_km

    return score

def contribution_breakdown(row, origin, prog_weights, region_boost, dist_weight_per_km):
    """Return a dict of additive contributions that sum to the displayed score."""
    # programme contribution
    progs = []
    if isinstance(row.get("programmes"), str):
        progs = [p.strip().upper() for p in row["programmes"].split(",") if p.strip()]
    prog_contrib = float(sum(prog_weights.get(p, 0) for p in progs))

    # region boost
    region_contrib = 10.0 if isinstance(row.get("region"), str) and row["region"] in region_boost else 0.0

    # distance penalty
    if origin is not None and pd.notna(row.get("lat")) and pd.notna(row.get("lng")):
        d = haversine_km(origin[0], origin[1], row["lat"], row["lng"])
        dist_contrib = -(d * dist_weight_per_km) if not np.isnan(d) else 0.0
    else:
        dist_contrib = 0.0

    total = prog_contrib + region_contrib + dist_contrib
    return {"Programmes": prog_contrib, "Region boost": region_contrib, "Distance penalty": dist_contrib, "Total": total}

# ---------- Sidebar (filters) ----------
st.sidebar.title("Filters")
q = st.sidebar.text_input("Search by name", placeholder="e.g., Oakridge, CHIREC")

regions = sorted([r for r in df["region"].dropna().unique().tolist()])
states = sorted([s for s in df["state"].dropna().unique().tolist()])
region_sel = st.sidebar.multiselect("Region", regions, default=regions or [])
state_sel = st.sidebar.multiselect("State/UT", states, default=states or [])

# Filter dataframe
mask = pd.Series(True, index=df.index)
if region_sel:
    mask &= df["region"].isin(region_sel)
if state_sel:
    mask &= df["state"].isin(state_sel)
if q:
    mask &= df["name"].str.contains(q, case=False, regex=False)
df_filt = df[mask].copy()

# ---------- Preferences (ranking) ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Preferences")

st.sidebar.caption("Programme weights (higher = more important)")
w_pyp = st.sidebar.slider("PYP weight", 0, 40, 10)
w_myp = st.sidebar.slider("MYP weight", 0, 40, 15)
w_dp  = st.sidebar.slider("DP weight",  0, 40, 25)
w_cp  = st.sidebar.slider("CP weight",  0, 40, 10)
prog_weights = {"PYP": w_pyp, "MYP": w_myp, "DP": w_dp, "CP": w_cp}

all_regions = sorted([r for r in df["region"].dropna().unique().tolist()])
region_boost = set(st.sidebar.multiselect("Boost these regions", all_regions, default=[]))

st.sidebar.caption("Distance preference (optional)")
dist_weight_per_km = st.sidebar.slider("Distance penalty per km", 0.0, 2.0, 0.2, 0.05)
max_km = st.sidebar.slider("Max distance (km) for table", 5, 200, 50, 5)

origin = st.session_state.get("origin")  # (lat, lng) if user clicked on map earlier

with st.sidebar.expander("Set your location (geocode)"):
    addr = st.text_input("Type an address or area (e.g., HSR Layout, Bengaluru)")
    if st.button("Geocode"):
        try:
            geocoder = Nominatim(user_agent="ib-schools-app")
            geocode_fn = RateLimiter(geocoder.geocode, min_delay_seconds=1)
            loc = geocode_fn(addr)
            if loc:
                origin = (loc.latitude, loc.longitude)
                st.session_state["origin"] = origin
                st.success(f"Location set at ~({origin[0]:.4f}, {origin[1]:.4f})")
            else:
                st.warning("Could not geocode that address.")
        except Exception as e:
            st.error(f"Geocoding failed: {e}")

# ---------- Header ----------
st.markdown('<div class="header">IB Schools Map</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Hover for names • Click a pin for photo & details.</div>', unsafe_allow_html=True)
st.write("")

if origin is None:
    st.info("Tip: Click anywhere on the map to set your location (we’ll compute distance and re-rank).")
else:
    st.success(f"Origin set at ~({origin[0]:.4f}, {origin[1]:.4f}). Click elsewhere on the map to move it.")

# ---------- Compute distance + score + color ----------
if origin is not None:
    df_filt["distance_km"] = df_filt.apply(
        lambda r: haversine_km(origin[0], origin[1], r["lat"], r["lng"]) 
        if pd.notna(r.get("lat")) and pd.notna(r.get("lng")) else np.nan, axis=1
    )
else:
    df_filt["distance_km"] = np.nan

df_filt["score"] = df_filt.apply(
    lambda r: score_row(r, origin, prog_weights, region_boost, dist_weight_per_km), axis=1
)

bins = [-1e9, 20, 40, 60, 1e9]
labels = ["red", "orange", "blue", "green"]
df_filt["color"] = pd.cut(df_filt["score"], bins=bins, labels=labels, include_lowest=True)

# ---------- Layout: left (table) + right (map) ----------
col_left, col_map = st.columns([2, 5])

with col_left:
    st.subheader("Top matches")
    df_rank = df_filt.copy()
    if origin is not None:
        df_rank = df_rank.sort_values(["score", "distance_km"], ascending=[False, True])
        df_rank = df_rank[df_rank["distance_km"].fillna(1e9) <= max_km]
    else:
        df_rank = df_rank.sort_values("score", ascending=False)

    show_cols = ["name", "city", "state", "programmes", "score", "distance_km"]
    for c in show_cols:
        if c not in df_rank.columns:
            df_rank[c] = pd.NA

    st.dataframe(
        df_rank[show_cols].head(15).style.format({"score": "{:.1f}", "distance_km": "{:.1f}"}),
        use_container_width=True
    )
    st.caption("Scores = programme weights (+), region boost (+10), minus distance penalty (km × weight).")

def _extract_click_latlng(ret: dict) -> tuple[float,float] | None:
    """Try several keys used by different streamlit-folium versions to get map click lat/lng."""
    if not isinstance(ret, dict):
        return None
    candidates = [
        "last_object_clicked", "last_clicked", "last_active_drawing",
        "last_marker_clicked", "map_clicked"
    ]
    for k in candidates:
        v = ret.get(k)
        if isinstance(v, dict):
            if "lat" in v and "lng" in v:
                try:
                    return float(v["lat"]), float(v["lng"])
                except Exception:
                    pass
    return None

with col_map:
    center = [origin[0], origin[1]] if origin else [15.9129, 79.7400]
    m = folium.Map(location=center, zoom_start=6, control_scale=True, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)

    # draw origin
    if origin:
        folium.CircleMarker(
            location=[origin[0], origin[1]],
            radius=7, fill=True, popup="Your location", tooltip="Your location"
        ).add_to(m)

    for _, row in df_filt.iterrows():
        if pd.isna(row.get("lat")) or pd.isna(row.get("lng")):
            continue  # skip rows without coords

        data_uri = ""
        if "image" in row and isinstance(row["image"], str):
            data_uri = img_to_data_uri(row["image"])
        img_html = f'<img src="{data_uri}" alt="{row["name"]}" style="width:240px;height:auto;border-radius:8px;">' if data_uri else "(image missing)"

        prog = row.get("programmes", "")
        website = row.get("website", "")
        link_html = f'<div style="margin-top:6px;"><a href="{website}" target="_blank">Website</a></div>' if isinstance(website, str) and website.startswith("http") else ""
        dist = row.get("distance_km")
        dist_html = f'<div style="color:#374151;margin-bottom:6px;">Distance: {dist:.1f} km</div>' if isinstance(dist, (int, float, np.floating)) and not np.isnan(dist) else ""

        score_val = row.get("score")
        score_html = f'<div style="margin-top:6px;color:#475569;">Score: {score_val:.1f}</div>' if isinstance(score_val, (int, float, np.floating)) else ""

        popup_html = f"""
        <div style="font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif; width:260px;">
          <div style="font-weight:700;font-size:16px;margin-bottom:6px;">{row["name"]}</div>
          <div style="margin-bottom:6px;color:#374151;">{row.get("city","")}, {row.get("state","")}</div>
          <div style="margin-bottom:6px;color:#444;">Programmes: {prog or "—"}</div>
          {dist_html}
          {img_html}
          {link_html}
          {score_html}
        </div>
        """

        color = str(row.get("color")) if pd.notna(row.get("color")) else "blue"
        folium.Marker(
            location=[row["lat"], row["lng"]],
            tooltip=row["name"],
            popup=folium.Popup(popup_html, max_width=280),
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(cluster)

    # Render map and capture clicks to set origin
    ret = st_folium(m, width=None, height=650)
    try:
        click_ll = _extract_click_latlng(ret)
        if click_ll:
            st.session_state["origin"] = (float(click_ll[0]), float(click_ll[1]))
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    except Exception:
        pass

    # Legend
    st.markdown(
        '<div class="subtle">Legend — Score: '
        '<span style="color:#dc2626;">●</span> low '
        '<span style="color:#f59e0b;">●</span> medium '
        '<span style="color:#3b82f6;">●</span> high '
        '<span style="color:#10b981;">●</span> top</div>',
        unsafe_allow_html=True
    )

# ---------- Compare & Explain ----------
st.markdown("---")
st.subheader("Compare & explain")

# Build ranked list to pick from (same logic as the left table)
df_rank_select = df_filt.copy()
if origin is not None:
    df_rank_select = df_rank_select.sort_values(["score", "distance_km"], ascending=[False, True])
else:
    df_rank_select = df_rank_select.sort_values("score", ascending=False)

# Build nice option labels: "Name — City, State"
def _opt_label(row):
    parts = [str(row.get("name") or "—")]
    loc = ", ".join([x for x in [row.get("city"), row.get("state")] if isinstance(x, str) and x])
    if loc:
        parts.append(f"— {loc}")
    return " ".join(parts)

df_rank_select["__opt__"] = df_rank_select.apply(_opt_label, axis=1)
options = df_rank_select["__opt__"].tolist()

default_opts = options[:3]  # default to top-3
chosen = st.multiselect("Pick up to 3 schools to compare:", options, default=default_opts, max_selections=3)

if chosen:
    df_cmp = df_rank_select[df_rank_select["__opt__"].isin(chosen)].copy()
    # Show quick metrics row
    st.write("")
    mcols = st.columns(len(df_cmp))
    for (idx, row), mc in zip(df_cmp.iterrows(), mcols):
        with mc:
            st.markdown(f"**{row['name']}**")
            st.caption(f"{row.get('city','')}, {row.get('state','')}")
            st.metric("Score", f"{row.get('score', float('nan')):.1f}")
            if "distance_km" in row and isinstance(row["distance_km"], (int, float, np.floating)) and not np.isnan(row["distance_km"]):
                st.metric("Distance (km)", f"{row['distance_km']:.1f}")
            st.caption(f"Programmes: {row.get('programmes','—') or '—'}")

    # Side-by-side contribution bar charts
    st.write("")
    ccols = st.columns(len(df_cmp))
    for (idx, row), cc in zip(df_cmp.iterrows(), ccols):
        parts = contribution_breakdown(row, st.session_state.get("origin"), prog_weights, region_boost, dist_weight_per_km)
        # Prepare chart data
        labels_c = ["Programmes", "Region boost", "Distance penalty"]
        vals = [parts["Programmes"], parts["Region boost"], parts["Distance penalty"]]
        fig, ax = plt.subplots()
        ax.bar(labels_c, vals)
        ax.set_title(f"Why this score: {row['name']}")
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("Contribution")
        st.pyplot(fig, clear_figure=True)

    # Comparison table + download
    st.write("")
    show_cols_cmp = ["name","city","state","programmes","score","distance_km","website","profile_url"]
    for c in show_cols_cmp:
        if c not in df_cmp.columns:
            df_cmp[c] = pd.NA
    st.dataframe(
        df_cmp[show_cols_cmp].reset_index(drop=True).style.format({"score":"{:.1f}","distance_km":"{:.1f}"}),
        use_container_width=True
    )

    csv_bytes = df_rank_select[["name","city","state","programmes","score","distance_km","website","profile_url"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download full ranked list (CSV)", data=csv_bytes, file_name="ib_schools_ranked.csv", mime="text/csv")
else:
    st.caption("Select up to three schools above to see side-by-side metrics and an explanation of their scores.")

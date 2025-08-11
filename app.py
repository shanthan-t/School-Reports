import base64
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster

# ---------- Page setup ----------
st.set_page_config(page_title="IB Schools in India", layout="wide")
st.markdown("""
<style>
.main > div { padding-top: 1rem; }
.header { font-size: 1.6rem; font-weight: 700; margin-bottom: .5rem; }
.subtle { color: #667085; }
</style>
""", unsafe_allow_html=True)

# ---------- Data (image names unchanged) ----------
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
df = pd.DataFrame(schools)

def img_to_data_uri(path: str) -> str:
    """Embed local images so Folium can render them."""
    p = Path(path)
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    mime = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64}"

# ---------- Sidebar (filters) ----------
st.sidebar.title("Filters")
q = st.sidebar.text_input("Search by name", placeholder="e.g., Oakridge, CHIREC")
regions = sorted(df["region"].unique())
states = sorted(df["state"].unique())
region_sel = st.sidebar.multiselect("Region", regions, default=regions)
state_sel = st.sidebar.multiselect("State/UT", states, default=states)

# Filter dataframe
mask = df["region"].isin(region_sel) & df["state"].isin(state_sel)
if q:
    mask &= df["name"].str.contains(q, case=False, regex=False)
df_filt = df[mask].copy()

# ---------- Header ----------
st.markdown('<div class="header">IB Schools Map</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Hover for names • Click a pin for photo & details.</div>', unsafe_allow_html=True)
st.write("")

# ---------- Layout: keep map to the right ----------
_, col_map = st.columns([1, 6])  # small spacer + map column

with col_map:
    # Map
    m = folium.Map(location=[15.9129, 79.7400], zoom_start=6, control_scale=True, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)
    icon_url = "https://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"

    for _, row in df_filt.iterrows():
        data_uri = img_to_data_uri(row["image"])
        img_html = f'<img src="{data_uri}" alt="{row["name"]}" style="width:240px;height:auto;border-radius:8px;">' if data_uri else "(image missing)"
        popup_html = f"""
        <div style="font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif; width:260px;">
          <div style="font-weight:700;font-size:16px;margin-bottom:6px;">{row["name"]}</div>
          <div style="margin-bottom:8px;color:#374151;">{row["city"]}, {row["state"]}</div>
          {img_html}
        </div>
        """
        icon = folium.features.CustomIcon(icon_url, icon_size=(32, 32), icon_anchor=(16, 32))
        folium.Marker(
            location=[row["lat"], row["lng"]],
            tooltip=row["name"],
            popup=folium.Popup(popup_html, max_width=280),
            icon=icon
        ).add_to(cluster)

    st_folium(m, width=None, height=650)

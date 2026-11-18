import os
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

API_URL = os.getenv("API_URL", "http://127.0.0.1:8002")

st.set_page_config(
    page_title="PathSnap",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ PathSnap")
st.subheader("Intelligent Route Prediction System")
st.write("Explore road networks and predict routes using machine learning.")

st.header("System Status")

backend_ready = False

try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    model_response = requests.get(
        f"{API_URL}/api/model/status", timeout=5
    )

    if health.status_code == 200:
        st.success("Backend connected successfully")
        backend_ready = True
    else:
        st.error("Backend is not healthy")

    if model_response.status_code == 200:
        model = model_response.json()
        if model["model_loaded"]:
            st.success(f"ML Model loaded: {model['model_name']}")
        else:
            st.warning("ML model is not loaded")
    else:
        st.error("Could not retrieve model status")

except requests.exceptions.RequestException as error:
    st.error("Backend is not reachable.")
    st.caption(f"Technical details: {error}")

st.divider()
st.header("📍 Route Input")

with st.form("route_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Starting Location")
        start_lat = st.number_input(
            "Start Latitude", value=12.9716, format="%.6f"
        )
        start_lon = st.number_input(
            "Start Longitude", value=77.5946, format="%.6f"
        )

    with col2:
        st.subheader("Destination")
        end_lat = st.number_input(
            "Destination Latitude", value=12.9352, format="%.6f"
        )
        end_lon = st.number_input(
            "Destination Longitude", value=77.6245, format="%.6f"
        )

    submitted = st.form_submit_button(
        "Preview Locations", use_container_width=True
    )

if submitted:
    if not (-90 <= start_lat <= 90 and -90 <= end_lat <= 90):
        st.error("Latitude must be between -90 and 90.")
    elif not (-180 <= start_lon <= 180 and -180 <= end_lon <= 180):
        st.error("Longitude must be between -180 and 180.")
    elif (start_lat, start_lon) == (end_lat, end_lon):
        st.error("Start and destination must be different.")
    else:
        st.session_state["route_points"] = {
            "start": [start_lat, start_lon],
            "destination": [end_lat, end_lon]
        }

if "route_points" in st.session_state:
    points = st.session_state["route_points"]
    start = points["start"]
    destination = points["destination"]

    st.divider()
    st.subheader("Location Preview")

    route_map = folium.Map(
        location=[
            (start[0] + destination[0]) / 2,
            (start[1] + destination[1]) / 2
        ],
        zoom_start=13
    )

    folium.Marker(
        start, popup="Starting Location",
        tooltip="Start", icon=folium.Icon(color="green")
    ).add_to(route_map)

    folium.Marker(
        destination, popup="Destination",
        tooltip="Destination", icon=folium.Icon(color="red")
    ).add_to(route_map)

    route_map.fit_bounds([start, destination])

    st_folium(
        route_map, width=None, height=450,
        key="pathsnap_map"
    )

    if backend_ready:
        st.info(
            "Location preview is working. Actual ML route prediction "
            "will be enabled after candidate generation and HMM "
            "decoder integration."
        )

st.divider()
st.caption("PathSnap | Member 4 – Frontend and Backend Integration")

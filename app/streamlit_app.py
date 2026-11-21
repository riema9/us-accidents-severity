"""Streamlit version of the accident severity dashboard (deployable on Streamlit Community Cloud).

Run locally from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.models.predictor import Predictor

CLASS_NAMES = ["Severity 1", "Severity 2", "Severity 3", "Severity 4"]
MEANING = ["Shortest traffic delay", "Short traffic delay", "Longer traffic delay", "Longest traffic delay"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]

st.set_page_config(page_title="US Accident Severity Predictor", layout="wide")


@st.cache_resource(show_spinner="Loading model and data (first start takes a minute)...")
def load_predictor():
    return Predictor()


P = load_predictor()


def find_weather(word, fallback):
    if word in P.weather_options:
        return word
    return next((w for w in P.weather_options if word.lower() in w.lower()), fallback)


DEFAULT_WEATHER = find_weather("Clear", P.weather_options[0])
DAY_SUN = "Day" if "Day" in P.sun_options else P.sun_options[0]

st.title("US Accident Severity Predictor")
st.caption("XGBoost model with tuned class weights. Severity measures traffic delay, not injuries.")

left, right = st.columns([3, 2])

with left:
    st.subheader("Weather")
    temperature = st.slider("Temperature (F)", -20, 110, 65)
    humidity = st.slider("Humidity (%)", 0, 100, 55)
    visibility = st.slider("Visibility (mi)", 0, 10, 10)
    wind = st.slider("Wind speed (mph)", 0, 60, 7)
    weather = st.selectbox("Condition", P.weather_options,
                           index=P.weather_options.index(DEFAULT_WEATHER))

    st.subheader("Time")
    hour = st.slider("Hour of day", 0, 23, 17)
    day = st.selectbox("Day", range(7), format_func=lambda i: DAYS[i], index=4)
    month = st.selectbox("Month", range(1, 13), format_func=lambda i: MONTHS[i - 1], index=10)
    sunrise_sunset = st.radio("Daylight", P.sun_options,
                              index=P.sun_options.index(DAY_SUN), horizontal=True)

    st.subheader("Road")
    junction = st.checkbox("Junction")
    traffic_signal = st.checkbox("Traffic signal")
    crossing = st.checkbox("Crossing")

    st.subheader("Location")
    lat = st.number_input("Latitude", value=34.05, format="%.3f")
    lng = st.number_input("Longitude", value=-118.24, format="%.3f")

with right:
    st.subheader("Prediction")
    if st.button("Predict severity", type="primary", use_container_width=True):
        r = P.predict({
            "temperature": temperature, "humidity": humidity, "visibility": visibility, "wind": wind,
            "precipitation": None, "weather": weather, "sunrise_sunset": sunrise_sunset,
            "hour": hour, "day_of_week": day, "month": month,
            "junction": junction, "traffic_signal": traffic_signal, "crossing": crossing,
            "lat": lat, "lng": lng,
        })
        d = r["decision"]
        st.metric("Predicted severity", f"{d + 1}: {CLASS_NAMES[d]}")
        st.write(MEANING[d])
        st.caption(f"Decision weight {r['share'] * 100:.0f}% after tuning")
        st.markdown("**What the model believes (raw probabilities)**")
        for i, p in enumerate(r["proba"]):
            st.write(f"{CLASS_NAMES[i]}: {p * 100:.1f}%")
            st.progress(float(p))
        st.info("The badge applies class weights tuned on validation data, so rare, severe crashes are not "
                "ignored. It can differ from the longest bar.")
    else:
        st.write("Set the conditions and press **Predict severity**.")

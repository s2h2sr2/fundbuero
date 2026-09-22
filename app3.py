import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from datetime import date
import io

# ─────────────────────────────────────────
# Seitenkonfiguration
# ─────────────────────────────────────────

st.set_page_config(page_title="Das Fundbüro", page_icon="🔍", layout="wide")

# ─────────────────────────────────────────
# Modell laden (einmalig, gecacht)
# ─────────────────────────────────────────

MODEL_PATH = "models/dein_model.h5"
KATEGORIEN = ["Hoodie", "Hose", "Flasche", "Schuhe"]
IMG_SIZE = (224, 224)

from tensorflow.keras.layers import DepthwiseConv2D

class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modell nicht gefunden unter: {MODEL_PATH}")
        st.stop()
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D}
    )
    return model

model = load_model()

# ─────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────

def klassifiziere_bild(image: Image.Image) -> tuple:
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    vorhersage = model.predict(img_array)
    index = int(np.argmax(vorhersage))
    konfidenz = float(np.max(vorhersage)) * 100
    return KATEGORIEN[index], konfidenz

def bild_zu_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────

if "gegenstaende" not in st.session_state:
    st.session_state.gegenstaende = []

if "seite" not in st.session_state:
    st.session_state.seite = "home"

# ─────────────────────────────────────────
# HEADER mit Navigation
# ─────────────────────────────────────────

st.markdown("""
<div style='background-color:#6c3483; padding: 1.5rem 2rem;
     border-radius: 12px; margin-bottom: 1.5rem;
     display:flex; align-items:center; justify-content:space-between;'>
    <h1 style='color:white; font-size:3rem; margin:0;'>
        Das <span style='color:#f4d03f;'>Fund</span><span style='color:white;'>büro</span> 🔍
    </h1>
</div>
""", unsafe_allow_html=True)

nav1, nav2, nav3 = st.columns([1, 1, 8])
with nav1:
    if st.button("Start", use_container_width=True):
        st.session_state.seite = "home"
with nav2:
    if st.button("Suchen", use_container_width=True):
        st.session_state.seite = "suchen"

st.markdown("---")

# ─────────────────────────────────────────
# SEITE: HOME
# ─────────────────────────────────────────

if st.session_state.seite == "home":

    st.markdown("## Was ist das Fundbüro?")
    st.markdown("")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("""
        <div style='background-color:#6c3483; border-radius:12px;
             padding:2rem; color:white;'>
            <h3>Hast du was gefunden?</h3>
            <p>Hier kannst du alles, was du findest, hochladen,
            damit Leute ihr Eigentum wiederfinden können.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Gegenstand eintragen", use_container_width=True):
            st.session_state.seite = "eintragen"

    with col_r:
        st.markdown("""
        <div style='background-color:#6c3483; border-radius:12px;
             padding:2rem; color:white;'>
            <h3>Hast du was verloren?</h3>
            <p>Hiermit kannst du deinen verlorenen Gegenstand suchen.
            Mit hilfreichen Filtern geht es ganz fix.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Jetzt suchen", use_container_width=True):
            st.session_state.seite = "suchen"

# ─────────────────────────────────────────
# SEITE: EINTRAGEN
# ─────────────────────────────────────────

elif st.session_state.seite == "eintragen":

    st.markdown("## Gegenstand eintragen")

    uploaded_file = st.file_uploader(
        "Bild hochladen (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladenes Bild", width=300)

        with st.spinner("KI analysiert das Bild..."):
            kategorie, konfidenz = klassifiziere_bild(image)

        st.success(f"Erkannte Kategorie: **{kategorie}** ({konfidenz:.1f}% sicher)")

        farbe     = st.text_input("Farbe des Gegenstands", placeholder="z.B. Blau")
        groesse   = st.selectbox("Größe", ["–", "XS", "S", "M", "L", "XL", "XXL", "Keine Angabe"])
        material  = st.text_input("Material (optional)", placeholder="z.B. Baumwolle")
        funddatum = st.date_input("Funddatum", value=date.today())

        if st.button("Gegenstand eintragen", use_container_width=True):
            eintrag = {
                "bild":      bild_zu_bytes(image),
                "kategorie": kategorie,
                "farbe":     farbe,
                "groesse":   groesse,
                "material":  material,
                "datum":     str(funddatum),
            }
            st.session_state.gegenstaende.append(eintrag)
            st.success("Gegenstand wurde eingetragen!")

# ─────────────────────────────────────────
# SEITE: SUCHEN
# ─────────────────────────────────────────

elif st.session_state.seite == "suchen":

    st.markdown("## Gegenstand suchen")

    st.markdown("### Filter")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        filter_farbe = st.text_input("Farbe", placeholder="z.B. Rot")
    with f2:
        filter_groesse = st.selectbox("Größe", ["Alle", "XS", "S", "M", "L", "XL", "XXL", "Keine Angabe"])
    with f3:
        filter_datum = st.date_input("Datum", value=None)
    with f4:
        filter_material = st.text_input("Material", placeholder="z.B. Leder")

    filter_kategorie = st.selectbox("Kategorie", ["Alle"] + KATEGORIEN)

    st.markdown("---")

    # ── Ergebnisse filtern ──
    ergebnisse = st.session_state.gegenstaende

    if filter_kategorie != "Alle":
        ergebnisse = [e for e in ergebnisse if e["kategorie"] == filter_kategorie]
    if filter_farbe.strip():
        ergebnisse = [e for e in ergebnisse if filter_farbe.strip().lower() in e["farbe"].lower()]
    if filter_groesse != "Alle":
        ergebnisse = [e for e in ergebnisse if e["groesse"] == filter_groesse]
    if filter_datum is not None:
        ergebnisse = [e for e in ergebnisse if str(filter_datum) in e["datum"]]
    if filter_material.strip():
        ergebnisse = [e for e in ergebnisse if filter_material.strip().lower() in e["material"].lower()]

    st.markdown("### Ergebnisse")

    if not ergebnisse:
        st.info("Keine Gegenstände gefunden. Passe deine Filter an!")
    else:
        cols = st.columns(3)
        for i, eintrag in enumerate(ergebnisse):
            with cols[i % 3]:
                with st.container(border=True):
                    st.image(eintrag["bild"], use_container_width=True)
                    st.markdown(f"**{eintrag['kategorie']}**")
                    st.markdown(f"Farbe: {eintrag['farbe'] or '–'}")
                    st.markdown(f"Größe: {eintrag['groesse']}")
                    st.markdown(f"Material: {eintrag['material'] or '–'}")
                    st.markdown(f"Datum: {eintrag['datum']}")

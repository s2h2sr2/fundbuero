import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from datetime import date

# ─────────────────────────────────────────
# Seitenkonfiguration
# ─────────────────────────────────────────
st.set_page_config(page_title="Das Fundbüro", page_icon="🔍", layout="wide")

# ─────────────────────────────────────────
# Modell laden (einmalig, gecacht)
# ─────────────────────────────────────────
MODEL_PATH = "model/dein_model.h5"
KATEGORIEN = ["Hoodie", "Schuhe", "Hose", "Flasche"]
IMG_SIZE = (224, 224)  # anpassen, falls dein Modell andere Größe erwartet

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modell nicht gefunden unter: {MODEL_PATH}")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ─────────────────────────────────────────
# Hilfsfunktion: Bild klassifizieren
# ─────────────────────────────────────────
def klassifiziere_bild(image: Image.Image) -> str:
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    vorhersage = model.predict(img_array)
    index = int(np.argmax(vorhersage))
    return KATEGORIEN[index]

# ─────────────────────────────────────────
# Session State für gefundene Gegenstände
# ─────────────────────────────────────────
if "gegenstaende" not in st.session_state:
    st.session_state.gegenstaende = []

# ─────────────────────────────────────────
# HEADER / LOGO
# ─────────────────────────────────────────
st.markdown("""
<div style='background-color:#1a1a2e; padding: 2rem 2rem 1rem 2rem; border-radius: 12px; margin-bottom: 1rem;'>
    <h1 style='color:white; font-size: 3rem; margin:0;'>Das <span style='color:#e94560;'>Fund</span><span style='color:white;'>büro</span> 🔍</h1>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# ZWEI SPALTEN: Links = Finden, Rechts = Suchen
# ─────────────────────────────────────────
col_links, col_rechts = st.columns(2, gap="large")

# ══════════════════════════════════════════
# LINKE SPALTE: Gegenstand hochladen
# ══════════════════════════════════════════
with col_links:
    st.markdown("## 📦 Hast du was gefunden?")
    st.markdown("""
    > Hier kannst du alles, was du findest, hochladen,  
    > damit Leute ihr Eigentum wiederfinden können.
    """)

    uploaded_file = st.file_uploader(
        "Bild hochladen (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

        with st.spinner("🤖 KI analysiert das Bild..."):
            kategorie = klassifiziere_bild(image)

        st.success(f"✅ Erkannte Kategorie: **{kategorie}**")

        farbe = st.text_input("Farbe des Gegenstands", placeholder="z.B. Blau")
        groesse = st.selectbox("Größe", ["–", "XS", "S", "M", "L", "XL", "XXL", "Keine Angabe"])
        material = st.text_input("Material (optional)", placeholder="z.B. Baumwolle")
        funddatum = st.date_input("Funddatum", value=date.today())

        if st.button("💾 Gegenstand eintragen", use_container_width=True):
            eintrag = {
                "bild": image,
                "kategorie": kategorie,
                "farbe": farbe,
                "groesse": groesse,
                "material": material,
                "datum": str(funddatum),
            }
            st.session_state.gegenstaende.append(eintrag)
            st.success("🎉 Gegenstand wurde eingetragen!")

# ══════════════════════════════════════════
# RECHTE SPALTE: Suchen & Filtern
# ══════════════════════════════════════════
with col_rechts:
    st.markdown("## 🔍 Hast du was verloren?")
    st.markdown("""
    > Hiermit kannst du deinen verlorenen Gegenstand suchen.  
    > Mit hilfreichen Filtern geht es ganz fix.
    """)

    st.markdown("### 🎛️ Filter")

    filter_kategorie = st.selectbox(
        "Kategorie", ["Alle"] + KATEGORIEN
    )
    filter_farbe = st.text_input("Farbe", placeholder="z.B. Rot")
    filter_groesse = st.selectbox(
        "Größe", ["Alle", "XS", "S", "M", "L", "XL", "XXL", "Keine Angabe"]
    )
    filter_material = st.text_input("Material", placeholder="z.B. Leder")

    st.markdown("---")
    st.markdown("### 🗂️ Ergebnisse")

    # Filtern
    ergebnisse = st.session_state.gegenstaende

    if filter_kategorie != "Alle":
        ergebnisse = [e for e in ergebnisse if e["kategorie"] == filter_kategorie]

    if filter_farbe.strip():
        ergebnisse = [
            e for e in ergebnisse
            if filter_farbe.strip().lower() in e["farbe"].lower()
        ]

    if filter_groesse != "Alle":
        ergebnisse = [e for e in ergebnisse if e["groesse"] == filter_groesse]

    if filter_material.strip():
        ergebnisse = [
            e for e in ergebnisse
            if filter_material.strip().lower() in e["material"].lower()
        ]

    if not ergebnisse:
        st.info("ℹ️ Keine Gegenstände gefunden. Passe deine Filter an!")
    else:
        for eintrag in ergebnisse:
            with st.container(border=True):
                img_col, info_col = st.columns([1, 2])
                with img_col:
                    st.image(eintrag["bild"], use_container_width=True)
                with info_col:
                    st.markdown(f"**Kategorie:** {eintrag['kategorie']}")
                    st.markdown(f"**Farbe:** {eintrag['farbe'] or '–'}")
                    st.markdown(f"**Größe:** {eintrag['groesse']}")
                    st.markdown(f"**Material:** {eintrag['material'] or '–'}")
                    st.markdown(f"**Datum:** {eintrag['datum']}")

import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Appliance Troubleshooter RAG", layout="centered")

st.title("🔧 Appliance Troubleshooting Assistant (Local RAG)")

st.markdown(
    "Pose une question sur un appareil électroménager (lave-linge, frigo, four...) "
    "et le système RAG utilisera la base de connaissances locale pour répondre."
)

question = st.text_input("Ta question :", placeholder="My washing machine is leaking water...")

if st.button("Envoyer") and question.strip():
    with st.spinner("Je réfléchis..."):
        try:
            resp = requests.post(BACKEND_URL, json={"question": question})
        except Exception as e:
            st.error(f"Impossible de contacter l'API backend : {e}")
        else:
            if resp.ok:
                data = resp.json()
                st.subheader("💬 Réponse")
                st.write(data.get("answer", ""))

                sources = data.get("sources", [])
                if sources:
                    st.subheader("📚 Contexte utilisé")
                    for s in sources:
                        st.write("- ", s)
                else:
                    st.info("Aucune source spécifique n’a été utilisée ou trouvée.")
            else:
                st.error(f"Erreur API {resp.status_code} : {resp.text}")
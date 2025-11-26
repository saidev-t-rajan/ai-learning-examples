import streamlit as st

st.set_page_config(page_title="AI Metrics", layout="wide")

page = st.sidebar.radio("Navigation", ["Dashboard", "Chat"])

if page == "Dashboard":
    from app.web import dashboard

    dashboard.render()
else:
    st.info("Chat UI coming soon. Use CLI: python -m app.main")

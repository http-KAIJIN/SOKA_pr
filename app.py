import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(
    page_title="SOKA – Analyse Financière",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 SOKA – Analyse Financière")

ticker = st.text_input("Entrez le symbole de l’action (ex : AAPL, TSLA)", "AAPL")

if ticker:
    data = yf.download(ticker, period="6mo")
    if not data.empty:
        st.line_chart(data["Close"])
    else:
        st.error("Aucune donnée trouvée pour ce symbole.")

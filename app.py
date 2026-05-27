"""
🚀 SOKA - Analyse Financière 
Web app Python/Streamlit | yfinance | Technical Analysis | Backtesting
Auteur: [BENLAIDI OUSSAMA] | CMC AI Student
"""

import re
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TRADING_DAYS = 252
_TICKER_RE = re.compile(r'^[A-Z0-9.\-\^=]{1,10}$')

# =============================================================================
# 📊 CONFIGURATION PAGE
# =============================================================================
st.set_page_config(
    page_title="SOKA - Analyse Financière",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 SOKA")
st.caption("Analyse Financière & Backtesting")

# =============================================================================
# 🔧 SIDEBAR - INPUTS
# =============================================================================
st.sidebar.header("⚙️ Configuration")

# Asset selection
ticker = st.sidebar.text_input("💹 Ticker", value="AAPL", help="ex: AAPL, MSFT, TSLA")

# Date range
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("📅 Début", value=datetime(2024, 1, 1))
end_date = col2.date_input("📅 Fin", value=datetime.now())

# Frequency
frequency = st.sidebar.selectbox(
    "⏰ Fréquence", 
    ["1d", "1wk", "1mo"], 
    index=0,
    help="Daily/Weekly/Monthly data"
)

# Analysis options
st.sidebar.subheader("🔍 Analyse")
compute_indicators = st.sidebar.checkbox("📊 Indicateurs techniques", True)
compute_backtest = st.sidebar.checkbox("⚔️ Backtest SMA", True)
show_stats = st.sidebar.checkbox("📈 Statistiques", True)

st.sidebar.subheader("📐 Paramètres")
risk_free_rate = st.sidebar.number_input(
    "Taux sans risque (% annuel)",
    min_value=0.0, max_value=20.0, value=0.0, step=0.25,
    help="Taux sans risque pour le Sharpe Ratio (ex: 4.5 pour un bon du Trésor américain à 4.5%)"
) / 100

if st.sidebar.button("🚀 ANALYSER", type="primary"):
    st.session_state.analysis_triggered = True

# =============================================================================
# 📥 DATA LOADING
# =============================================================================
@st.cache_data
def load_data(ticker, start_date, end_date, interval="1d"):
    """Télécharge données OHLC depuis Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            return pd.DataFrame()
        return data.dropna()
    except Exception:
        return pd.DataFrame()

# =============================================================================
# 🔢 CALCULS RENDMENTS
# =============================================================================
def compute_returns(data):
    """Rendements arithmétiques et logarithmiques"""
    data['Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    return data.dropna()

# =============================================================================
# 📊 INDICATEURS TECHNIQUES
# =============================================================================
def add_sma(data, short_window=20, long_window=50):
    """Moyennes mobiles simples"""
    data["SMA_short"] = data["Close"].rolling(window=short_window).mean()
    data["SMA_long"] = data["Close"].rolling(window=long_window).mean()
    return data

def add_rsi(data, period=14):
    """Relative Strength Index"""
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data

def add_bollinger_bands(data, window=20, num_std=2):
    """Bandes de Bollinger (sur Close seulement)"""
    close = data['Close']

    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()

    data["BB_Middle"] = middle.astype(float)
    data["BB_Upper"] = (middle + num_std * std).astype(float)
    data["BB_Lower"] = (middle - num_std * std).astype(float)

    return data


def add_macd(data, fast=12, slow=26, signal=9):
    """MACD"""
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    data['MACD'] = ema_fast - ema_slow
    data['MACD_Signal'] = data['MACD'].ewm(span=signal).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    return data

# =============================================================================
# ⚔️ BACKTESTING
# =============================================================================
def backtest_sma_strategy(data, short_window=20, long_window=50):
    """Stratégie croisement SMA + signaux Buy/Sell (simple)"""

    # Position (1 = long, 0 = hors marché)
    data["Position"] = 0
    data.loc[data["SMA_short"] > data["SMA_long"], "Position"] = 1

    # Detect transitions via diff: +1 = Buy entry, -1 = Sell exit
    position_change = data["Position"].diff()
    data["Buy_Signal"] = data["Close"].where(position_change == 1)
    data["Sell_Signal"] = data["Close"].where(position_change == -1)

    # Backtest performance
    data["Strategy_Return"] = data["Position"].shift(1) * data["Return"]
    data["Strategy_Cumulative"] = (1 + data["Strategy_Return"]).cumprod()
    data["Buy_Hold"] = (1 + data["Return"]).cumprod()

    return data


# =============================================================================
# 📈 STATISTIQUES
# =============================================================================
def compute_detailed_stats(returns, risk_free_rate=0.0):
    """Statistiques complètes annualisées"""
    if len(returns) == 0:
        return {}

    ann_return = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0

    return {
        'Rendement Journalier (%)': returns.mean() * 100,
        'Volatilité Journalière (%)': returns.std() * 100,
        'Rendement Annualisé (%)': ann_return * 100,
        'Volatilité Annualisée (%)': ann_vol * 100,
        'Taux Sans Risque (%)': risk_free_rate * 100,
        'Sharpe Ratio': sharpe,
        'Min (%)': returns.min() * 100,
        'Max (%)': returns.max() * 100,
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    }

# =============================================================================
# 📊 VISUALISATIONS
# =============================================================================
def plot_price_chart(data, ticker):
    """Graphique prix + indicateurs"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Prix & SMA', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # ================== PRIX ==================
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Prix'
    ), row=1, col=1)
    
    # ================== SMA ==================
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SMA_short'],
        name='SMA 20', line=dict(color='orange')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=data['SMA_long'],
        name='SMA 50', line=dict(color='blue')
    ), row=1, col=1)
    
    # ================== BOLLINGER ==================
    fig.add_trace(go.Scatter(
        x=data.index, y=data['BB_Upper'],
        name='BB Haut', line=dict(color='gray', dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=data['BB_Lower'],
        name='BB Bas', line=dict(color='gray', dash='dash')
    ), row=1, col=1)

    # ================== BUY / SELL ==================
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Buy_Signal'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=14),
        name='Buy'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Sell_Signal'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=14),
        name='Sell'
    ), row=1, col=1)
    
    # ================== RSI ==================
    fig.add_trace(go.Scatter(
        x=data.index, y=data['RSI'],
        name='RSI', line=dict(color='purple')
    ), row=2, col=1)

    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # ================== MACD ==================
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MACD'],
        name='MACD', line=dict(color='blue')
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=data['MACD_Signal'],
        name='Signal', line=dict(color='red')
    ), row=3, col=1)
    
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        title_text=f"Prix et Indicateurs Techniques - {ticker}"
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_backtest_results(data):
    """Comparaison Buy&Hold vs Stratégie"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Equity curves
    ax1.plot(data.index, data['Strategy_Cumulative'], 
             label='Stratégie SMA', linewidth=2, color='green')
    ax1.plot(data.index, data['Buy_Hold'], 
             label='Buy & Hold', linewidth=2, color='blue')
    ax1.set_title('Performance Cumulative')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    strategy_dd = (data['Strategy_Cumulative'] / data['Strategy_Cumulative'].cummax() - 1) * 100
    bh_dd = (data['Buy_Hold'] / data['Buy_Hold'].cummax() - 1) * 100
    
    ax2.fill_between(data.index, strategy_dd, 0, alpha=0.3, color='green', label='Stratégie')
    ax2.fill_between(data.index, bh_dd, 0, alpha=0.3, color='blue', label='Buy & Hold')
    ax2.set_title('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# =============================================================================
# 🎯 MAIN APPLICATION
# =============================================================================
if 'analysis_triggered' in st.session_state and st.session_state.analysis_triggered:

    ticker_clean = ticker.strip().upper()
    if not _TICKER_RE.match(ticker_clean):
        st.error(
            f"❌ Ticker invalide : « {ticker} ». "
            "Utilisez uniquement des lettres, chiffres ou les symboles `.^-=` (max 10 caractères)."
        )
        st.stop()

    # Load data
    with st.spinner('Chargement des données...'):
        raw_data = load_data(ticker_clean, start_date, end_date, frequency)

    if raw_data.empty:
        st.error(f"❌ Aucune donnée pour {ticker_clean}. Vérifiez le ticker et la période.")
    elif not raw_data.empty:
        st.success(f"✅ {len(raw_data)} lignes chargées")
        st.session_state.data = raw_data.copy()
        data = raw_data.copy()
        
        # Calculs
        data = compute_returns(data)
        
        if compute_indicators:
            data = add_sma(data)
            data = add_rsi(data)
            data = add_bollinger_bands(data)
            data = add_macd(data)
        
        if compute_backtest:
            data = backtest_sma_strategy(data)
        
        # =============================================================================
        # 📋 DASHBOARD PRINCIPAL
        # =============================================================================
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        returns = data['Return'].dropna()

        with col1:
            if compute_backtest and 'Strategy_Cumulative' in data.columns:
                st.metric("Rendement Total (%)",
                         f"{(data['Strategy_Cumulative'].iloc[-1]-1)*100:.2f}%")
            else:
                bh = (1 + returns).cumprod().iloc[-1] - 1
                st.metric("Rendement Total (%)", f"{bh*100:.2f}%")
        with col2:
            st.metric("Vol Annualisée (%)",
                     f"{returns.std()*np.sqrt(TRADING_DAYS)*100:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{compute_detailed_stats(returns, risk_free_rate)['Sharpe Ratio']:.2f}")
        with col4:
            st.metric("Période", f"{len(data)} jours")

        # Graphiques
        tab1, tab2, tab3 = st.tabs(["📈 Prix & Indicateurs", "⚔️ Backtest", "📊 Statistiques"])

        with tab1:
            if compute_indicators:
                plot_price_chart(data, ticker_clean)
            else:
                st.info("Activez les indicateurs techniques dans la barre latérale pour afficher ce graphique.")

        with tab2:
            if compute_backtest and 'Strategy_Cumulative' in data.columns:
                st.subheader("Performance Stratégie vs Buy & Hold")
                plot_backtest_results(data)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Stratégie Final (%)",
                             f"{(data['Strategy_Cumulative'].iloc[-1]-1)*100:.1f}%")
                with col2:
                    st.metric("Buy & Hold Final (%)",
                             f"{(data['Buy_Hold'].iloc[-1]-1)*100:.1f}%")
            else:
                st.info("Activez le Backtest SMA dans la barre latérale pour afficher ce graphique.")

        with tab3:
            if show_stats:
                stats = compute_detailed_stats(returns, risk_free_rate)
                st.subheader("📊 Statistiques Détaillées")
                st.json(stats)

                # Distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(returns*100, bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(returns.mean()*100, color='red', linestyle='--', label='Moyenne')
                ax.set_xlabel('Rendement Journalier (%)')
                ax.set_title('Distribution des Rendements')
                ax.legend()
                st.pyplot(fig)

        # Données brutes
        with st.expander("📋 Données Brutes (100 dernières lignes)"):
            available_cols = [c for c in ['Close', 'SMA_short', 'SMA_long', 'RSI',
                                          'MACD', 'Position', 'Strategy_Return']
                              if c in data.columns]
            st.dataframe(data[available_cols].tail(100))
        
        # Export
        csv = data.to_csv()
        st.download_button(
            label="💾 Télécharger CSV",
            data=csv,
            file_name=f'SOKA_{ticker_clean}_{start_date}_to_{end_date}.csv',
            mime='text/csv'
        )

# Footer
st.markdown("---")
st.caption("🎓 Projet pédagogique CMC AI | Python/Streamlit/yfinance")

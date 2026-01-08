"""
🚀 SOKA - Analyse Financière 
Web app Python/Streamlit | yfinance | Technical Analysis | Backtesting
Auteur: [OUSSAMA BENLAIDI] | CMC AI Student
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SOKA - Analyse Financière",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<div style='text-align: center;'>
    <h1 style='color: #1f77b4; font-size: 3em;'>🚀 SOKA</h1>
    <p style='font-size: 1.2em; color: #666;'>Analyse Financière & Backtesting</p>
</div>
""", unsafe_allow_html=True)

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

if st.sidebar.button("🚀 ANALYSER", type="primary"):
    st.session_state.analysis_triggered = True


@st.cache_data
def load_data(ticker, start_date, end_date, interval="1d"):
    """Télécharge données OHLC depuis Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            st.error(f"❌ Aucune donnée pour {ticker}")
            return pd.DataFrame()
        data.dropna(inplace=True)
        st.success(f"✅ {len(data)} lignes chargées")
        return data
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        return pd.DataFrame()


def compute_returns(data):
    """Rendements arithmétiques et logarithmiques"""
    data['Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)
    return data

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


def backtest_sma_strategy(data, short_window=20, long_window=50):
    """Stratégie croisement SMA"""
    data["Position"] = 0
    data.loc[data["SMA_short"] > data["SMA_long"], "Position"] = 1
    data.loc[data["SMA_short"] < data["SMA_long"], "Position"] = -1
    
    data["Strategy_Return"] = data["Position"].shift(1) * data["Return"]
    data["Strategy_Cumulative"] = (1 + data["Strategy_Return"]).cumprod()
    data["Buy_Hold"] = (1 + data["Return"]).cumprod()
    
    return data


def compute_detailed_stats(returns):
    """Statistiques complètes annualisées"""
    if len(returns) == 0:
        return {}
    
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    return {
        'Rendement Journalier (%)': returns.mean() * 100,
        'Volatilité Journalière (%)': returns.std() * 100,
        'Rendement Annualisé (%)': ann_return * 100,
        'Volatilité Annualisée (%)': ann_vol * 100,
        'Sharpe Ratio': sharpe,
        'Min (%)': returns.min() * 100,
        'Max (%)': returns.max() * 100,
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    }


def plot_price_chart(data):
    """Graphique prix + indicateurs"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Prix & SMA', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Prix + SMA + Bollinger
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name='Prix'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_short'], 
                           name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_long'], 
                           name='SMA 50', line=dict(color='blue')), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], 
                           name='BB Haut', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], 
                           name='BB Bas', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], 
                           name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                           name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], 
                           name='Signal', line=dict(color='red')), row=3, col=1)
    
    fig.update_layout(height=800, title_text=f"Prix et Indicateurs Techniques - {ticker}")
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

if 'analysis_triggered' in st.session_state and st.session_state.analysis_triggered:
    
    # Load data
    with st.spinner('Chargement des données...'):
        raw_data = load_data(ticker, start_date, end_date, frequency)
    
    if not raw_data.empty:
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
        
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        returns = data['Return'].dropna()
        
        with col1:
            st.metric("Rendement Total (%)", 
                     f"{(data['Strategy_Cumulative'].iloc[-1]-1)*100:.2f}%")
        with col2:
            st.metric("Vol Annualisée (%)", 
                     f"{returns.std()*np.sqrt(252)*100:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{compute_detailed_stats(returns)['Sharpe Ratio']:.2f}")
        with col4:
            st.metric("Période", f"{len(data)} jours")
        
        # Graphiques
        tab1, tab2, tab3 = st.tabs(["📈 Prix & Indicateurs", "⚔️ Backtest", "📊 Statistiques"])
        
        with tab1:
            plot_price_chart(data)
        
        with tab2:
            st.subheader("Performance Stratégie vs Buy & Hold")
            plot_backtest_results(data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stratégie Final (%)", 
                         f"{(data['Strategy_Cumulative'].iloc[-1]-1)*100:.1f}%")
            with col2:
                st.metric("Buy & Hold Final (%)", 
                         f"{(data['Buy_Hold'].iloc[-1]-1)*100:.1f}%")
        
        with tab3:
            if show_stats:
                stats = compute_detailed_stats(returns)
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
            st.dataframe(data[['Close', 'SMA_short', 'SMA_long', 'RSI', 
                             'MACD', 'Position', 'Strategy_Return']].tail(100))
        
        # Export
        csv = data.to_csv()
        st.download_button(
            label="💾 Télécharger CSV",
            data=csv,
            file_name=f'SOKA_{ticker}_{start_date}_to_{end_date}.csv',
            mime='text/csv'
        )
        
    else:
        st.error("❌ Impossible de charger les données")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 Projet pédagogique CMC AI | Python/Streamlit/yfinance</p>
</div>
""", unsafe_allow_html=True)


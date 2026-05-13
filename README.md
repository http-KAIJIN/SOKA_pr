# 🚀 SOKA — Analyse Financière & Backtesting (Streamlit)

**SOKA** est une application web développée en **Python + Streamlit** pour l’analyse financière, l’affichage d’indicateurs techniques et le **backtesting** d’une stratégie simple basée sur le croisement de moyennes mobiles (SMA).

> 📌 Objectif : fournir une interface claire et interactive pour analyser une action (ticker) via **Yahoo Finance** et comparer **Buy & Hold** vs **Stratégie SMA**.

---

## ✨ Fonctionnalités

1) Chargement des données (Yahoo Finance)
 - Téléchargement des données **OHLCV** via `yfinance`
 - Paramètres :
  - **Ticker** (ex: `AAPL`, `TSLA`, `MSFT`)
  - **Période** (date début / date fin)
  - **Fréquence** : `1d` / `1wk` / `1mo`
- Mise en cache avec `@st.cache_data` pour accélérer les relances

2) Indicateurs techniques (optionnels)
- **SMA 20 / SMA 50**
- **RSI (14)**
- **Bandes de Bollinger (20, 2σ)**
- **MACD (12, 26, 9)**

3) Backtesting (optionnel)
Stratégie **SMA Crossover** :
- **Position = 1 (Long)** lorsque `SMA_short > SMA_long`
- **Position = 0 (Cash)** sinon  
Avec :
- Signaux **Buy / Sell** (marqueurs sur le graphe)
- Performance cumulée :
  - `Strategy_Cumulative`
  - `Buy_Hold`
- Drawdown comparatif (Stratégie vs Buy&Hold)

4) Dashboard complet
- KPIs :
  - **Rendement total (%)**
  - **Volatilité annualisée (%)**
  - **Sharpe ratio**
  - **Nombre de jours**
- Onglets :
  - 📈 Prix & Indicateurs (Plotly)
  - ⚔️ Backtest (Matplotlib)
  - 📊 Statistiques (JSON + distribution rendements)
- Affichage données brutes (dernieres lignes)
- Export CSV via bouton “Télécharger CSV”

---

Méthodologie & Calculs

 Rendements
- Rendement simple : `Return = Close.pct_change()`
- Rendement log : `Log_Return = log(Close / Close.shift(1))`

 Statistiques
- Rendement annualisé : `mean(Return) * 252`
- Volatilité annualisée : `std(Return) * sqrt(252)`
- Sharpe ratio : `ann_return / ann_vol` (si vol ≠ 0)

---

Technologies & Libraries
- **Python**
- **Streamlit**
- **yfinance**
- **pandas / numpy**
- **plotly**
- **matplotlib**
- **seaborn**
```bash

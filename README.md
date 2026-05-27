# 🚀 SOKA — Analyse Financière & Backtesting (Streamlit)

**SOKA** est une application web développée en **Python + Streamlit** pour l'analyse financière, l'affichage d'indicateurs techniques et le **backtesting** d'une stratégie simple basée sur le croisement de moyennes mobiles (SMA).

> 📌 Objectif : fournir une interface claire et interactive pour analyser une action (ticker) via **Yahoo Finance** et comparer **Buy & Hold** vs **Stratégie SMA**.

---

## ✨ Fonctionnalités

1. **Chargement des données (Yahoo Finance)**
   - Téléchargement des données **OHLCV** via `yfinance`
   - Paramètres : ticker (ex: `AAPL`, `TSLA`, `MSFT`), période, fréquence `1d` / `1wk` / `1mo`
   - Mise en cache avec `@st.cache_data` pour accélérer les relances
   - Validation du ticker avant tout appel réseau

2. **Indicateurs techniques** *(optionnels)*
   - SMA 20 / SMA 50
   - RSI (14)
   - Bandes de Bollinger (20, 2σ)
   - MACD (12, 26, 9)

3. **Backtesting** *(optionnel)*  
   Stratégie **SMA Crossover** :
   - Position = 1 (Long) lorsque `SMA_short > SMA_long`, sinon 0 (Cash)
   - Signaux Buy / Sell détectés par différence vectorisée (sans boucle Python)
   - Performance cumulée : `Strategy_Cumulative` vs `Buy_Hold`
   - Drawdown comparatif

4. **Dashboard complet**
   - KPIs : Rendement total, Volatilité annualisée, Sharpe ratio, Nombre de jours
   - Onglets : 📈 Prix & Indicateurs (Plotly) · ⚔️ Backtest (Matplotlib) · 📊 Statistiques
   - Export CSV

---

## 🚀 Lancement

```bash
git clone https://github.com/http-KAIJIN/SOKA_pr.git
cd SOKA_pr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 📐 Méthodologie & Calculs

**Rendements**
- Rendement simple : `Return = Close.pct_change()`
- Rendement log : `Log_Return = log(Close / Close.shift(1))`

**Statistiques** *(annualisées sur 252 jours de bourse)*
- Rendement annualisé : `mean(Return) * 252`
- Volatilité annualisée : `std(Return) * sqrt(252)`
- Sharpe ratio : `(ann_return - risk_free_rate) / ann_vol`

Le taux sans risque est configurable dans la barre latérale (défaut : 0 %).

---

## 🛠️ Technologies

| Bibliothèque | Rôle |
|---|---|
| Streamlit | Interface web |
| yfinance | Données de marché (Yahoo Finance) |
| pandas / numpy | Manipulation des données |
| Plotly | Graphiques interactifs |
| Matplotlib | Graphiques backtest |

---

## 🔒 Notes de sécurité & qualité

- Les tickers sont validés par expression régulière avant tout appel réseau.
- Aucune dépendance inutile (`seaborn`, `Pillow` supprimés).
- Les versions des dépendances sont épinglées (`requirements.txt`).

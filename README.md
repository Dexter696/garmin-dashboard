# 🏃 Garmin Trenér - AI-Powered Personal Training Coach

> Automatické stahování dat z Garmin Connect + AI analýza pro optimalizaci tréninku a regenerace

---

## 🎯 Co To Je?

Osobní AI trenér, který:
- ✅ Stahuje tvá data z Garmin Connect (aktivity, spánek, stress)
- ✅ Analyzuje korelace mezi tréninkem a regenerací
- ✅ Poskytuje AI-powered doporučení
- ✅ Predikuje optimální načasování tréninku

**Inspirováno:** Australským výzkumem o HRV, spánku a výkonu

---

## 📊 Současný Status

**Fáze:** ✅ Data Collection COMPLETED → 🔄 Ready for Analysis

### Co Máme:
- ✅ 50 tréninkových aktivit (182 km, 19.8 hod)
- ✅ 91 nocí spánku (sleep scores, fáze)
- ✅ 91 dní stress dat
- ✅ Propojený master dataset

### Co Chybí:
- ⏭️ Pokročilá analýza (korelace, trendy)
- ⏭️ Vizualizace a dashboard
- ⏭️ AI chat trenér

---

## 🚀 Quick Start

### 1. Přečti Dokumentaci

Než začneš, **POVINNĚ přečti:**

📄 **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Přehled projektu, co máme, co chceme  
📄 **[DATA_DESCRIPTION.md](DATA_DESCRIPTION.md)** - Detailní popis všech dat  
📄 **[NEXT_STEPS.md](NEXT_STEPS.md)** - Konkrétní kroky co dělat  
📄 **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Jak nainstalovat Claude Code  

### 2. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Nastav přihlašovací údaje (pokud ještě nemáš)
# Edituj .env soubor s tvým Garmin emailem a heslem
```

### 3. Spusť Claude Code

```bash
# Install Claude Code (if not installed)
npm install -g @anthropic-ai/claude-code

# Start Claude Code v projektu
cd "C:\Users\mirek\Desktop\Garmin Trener"
claude-code
```

### 4. Začni Pracovat

V Claude Code napiš:
```
Ahoj! Přečti si PROJECT_SUMMARY.md a začni s Task 2.1 z NEXT_STEPS.md
```

---

## 📁 Struktura Projektu

```
Garmin Trener/
│
├── 📄 README.md                  ← Tento soubor
├── 📄 PROJECT_SUMMARY.md         ← ZAČNI TADY!
├── 📄 DATA_DESCRIPTION.md        ← Popis dat
├── 📄 NEXT_STEPS.md              ← Co dělat dál
├── 📄 SETUP_GUIDE.md             ← Návod na setup
│
├── 🔒 .env                       ← Přihlašovací údaje (NECOMMITOVAT!)
├── 📦 requirements.txt           ← Python dependencies
│
├── 🐍 garmin_fixed.py            ← ✅ Fungující scraper aktivit
├── 🐍 garmin_health_fixed.py    ← ✅ Fungující health scraper
├── 🐍 garmin_aggressive_scraper.py  ← 🔍 Debug tool
│
└── 📂 garmin_data/               ← VŠECHNA DATA
    ├── activities_*.csv          ← 50 aktivit
    ├── activities_*.json         ← Raw data
    ├── master_dataset_*.csv      ← Propojený dataset
    │
    └── 📂 health/
        ├── sleep_*.csv           ← 91 nocí
        ├── stress_*.csv          ← 91 dní
        ├── debug_summary_2.json  ← Daily summary (parsovat!)
        ├── debug_vo2_1.json      ← VO2 Max (parsovat!)
        └── health_data_*.json    ← Complete dump
```

---

## 🎯 Priority Tasks

Pokud nevíš, kde začít:

1. **Task 2.1:** Parsuj debug soubory (15 min) → víc dat
2. **Task 2.2:** Rozšiř master dataset (30 min) → kompletní data
3. **Task 3.2:** Korelační analýza (1.5 hod) → první insights
4. **Task 4.3:** AI chat trenér (3 hod) → wow efekt!

**Detaily viz [NEXT_STEPS.md](NEXT_STEPS.md)**

---

## 📖 Dokumentace

### 📄 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
Komplexní přehled projektu - co máme, co chceme, jak to funguje.

### 📄 [DATA_DESCRIPTION.md](DATA_DESCRIPTION.md) 
Detailní popis všech datových souborů, struktura, metriky.

### 📄 [NEXT_STEPS.md](NEXT_STEPS.md)
Konkrétní kroky co dělat, prioritizované tasky, časové odhady.

### 📄 [SETUP_GUIDE.md](SETUP_GUIDE.md)
Instalace Claude Code, troubleshooting, best practices.

---

## 🤖 Tech Stack

**Data Collection:**
- Python 3.11
- garth - Garmin API client
- pandas - Data manipulation

**Analysis (Planned):**
- pandas, numpy - Data processing
- matplotlib, seaborn - Vizualizace
- scikit-learn - ML (optional)

**Dashboard (Planned):**
- React + Tailwind - UI
- Recharts - Interaktivní grafy
- Claude API - AI trenér chat

---

## 🔐 Bezpečnost

⚠️ **DŮLEŽITÉ:** Soubor `.env` obsahuje přihlašovací údaje! NIKDY nesdílej.

---

## 🆘 Troubleshooting

Viz [SETUP_GUIDE.md](SETUP_GUIDE.md) pro detailní návody.

---

## 👤 About

**Projekt:** Garmin Trenér  
**Autor:** Miroslav Sládek  
**Sport:** Běh + Kolo  
**Vytvořeno:** Únor 2026  

---

**Vytvořeno s ❤️ pro optimalizaci tréninku! 🏃‍♂️💪🤖**

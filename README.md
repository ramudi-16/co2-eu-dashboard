# European Car Registration Emissions Dashboard

## About
Interactive Streamlit dashboard analysing CO2 emissions
across 5 European nations from 2019 to 2023.

- 587,593 vehicle registrations
- Countries: Croatia, Finland, Greece, Ireland, Norway
- Source: European Environment Agency (EEA)
- Regulation: EU Regulation 2019/631

## Files
app.py : Streamlit Dashboard 
co2_data_clean.csv : Cleaned dataset used by the Dashboard 
clean_data.py : Data Cleaning Script 
co2_data.csv  : Orginal Dataset 
requirements.txt : Python Dependancies for deployment 

## Features
- 5 interactive filters
- 6 KPI cards
- 8 visualisations
- CSV download

## How to Run Locally
`pip install -r requirements.txt`
`streamlit run app.py`

## 📊 Key Sustainability Findings

**37% CO2 reduction** from 122.9 g/km (2019) to 77.4 g/km (2023)
**Norway** achieved 91.5% EV share by 2023 — EU leader
**Croatia** remains at 4.1% EV share — lowest in dataset
**Greece** (124.6 g/km) and **Croatia** (131.0 g/km) exceed EU limit
**41.8%** of all registrations are Electric or Plug-in Hybrid
**COVID-19 (2020)** caused temporary CO2 drop — not real improvement

## 🎨 Streamlit Widgets Used
`st.cache_data` : Cache dataset for performance 
`st.spinner` : Loading indicator 
`st.multiselect` : Country, fuel, weight, CO2 filters 
`st.slider` : Year range filter 
`st.selectbox` : Country fuel mix selector 
`st.radio` : Count vs percentage toggle 
`st.checkbox` : Show/hide statistics table 
`st.expander` : Collapsible data table 
`st.download_button` : CSV export 
`st.tabs` : Overview and Vehicle Explorer 
`st.columns` : 3-column chart layout 
`st.plotly_chart` :  Interactive chart display 
`st.dataframe` : Data table with column config 
`st.text_input` : Search box 
`st.warning` : COVID notice and empty filter 
`st.stop` : Prevent crash on empty filter 

## 🔧 Development Methodology
### Sprint 1: Data loading and sidebar filters

### Sprint 2: CO2 trend chart

### Sprint 3: EV market share chart

### Sprint 4: Fuel mix chart

### Sprint 5: Heatmap and scatter charts

### Sprint 6: Manufacturer chart and policy conclusion

## 🌐 Deployment

**Platform:** Streamlit Cloud
Repository : https://github.com/ramudi-16/eu-co2-dashboard |

Branch : main 

Main file : app.py 

## Live Dashboard
[Click here to open dashboard](https://eu-co2-dashboard.streamlit.app/)

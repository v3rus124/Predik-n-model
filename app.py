import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Načtení trénovaného modelu
model = xgb.XGBRegressor()
model.load_model("model.pkl")

st.title("Predikce potřebného tepla podle venkovní teploty")

df1 = pd.read_excel("prumerna_dodavka_tepla.xlsx")
df1 = df1.groupby(['mesic', 'hodina'])['dodavka_tepla'].mean()

uploaded_file = st.file_uploader("Nahrajte Excel soubor (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df["Teplota venkovní"] = df["Teplota venkovní"].astype(str).str.replace(',', '.').astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%y %H:%M")
    # Kontrola, zda soubor obsahuje správná data
    if "Teplota venkovní" not in df.columns or len(df) != 24:
        st.error("Soubor musí obsahovat sloupec 'Teplota venkovní' s 24 hodnotami.")
    else:
        # Přidání hodin a dnů v týdnu
        df["hodina"] = df["timestamp"].dt.hour
        df["den_v_tydnu"] = df["timestamp"].dt.dayofweek
        df["mesic"] = df["timestamp"].dt.month 
        df["Topna_sezona"] = df["mesic"].apply(lambda x: 1 if x in [10, 11, 12, 1, 2, 3, 4, 5] else 0)
        df["heat_demand_lag_1"] = df.apply(lambda x: df1[(x["mesic"], x["hodina"])], axis=1)
        
        df["month_sin"] = np.sin(2 * np.pi * df["mesic"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["mesic"] / 12)

        df["hour_sin"] = np.sin(2 * np.pi * df["hodina"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hodina"] / 24)

        df["day_of_week_sin"] = np.sin(2 * np.pi * df["den_v_tydnu"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["den_v_tydnu"] / 7)
        
        X_new = df[['Teplota venkovní', 'hodina', 'den_v_tydnu', 'mesic',"Topna_sezona", "heat_demand_lag_1", "month_sin", "month_cos", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"]]  # Zvol správné featury

        predictions = model.predict(X_new)
        # Predikce
        df["Predikce dodávky tepla"] = predictions
        
        # Zobrazení výsledků
        st.write("### Výsledky predikce:")
        st.dataframe(df[["timestamp", "Teplota venkovní", "Predikce dodávky tepla"]])
        
        # Graf
        fig, ax = plt.subplots()
        ax.plot(df["hodina"], df["Predikce dodávky tepla"], marker="o", linestyle="-", label="Predikovaná dodávka tepla")
        ax.set_xlabel("Hodina")
        ax.set_ylabel("Množství tepla (GJ/h)")
        ax.set_title("Predikce tepla pro následujících 24 hodin - EPRU")
        ax.legend()
        st.pyplot(fig)
# Zobrazení výsledku
#predikce = df["Predikce dodávky tepla"].sum()
#st.write(f"### Predikované množství tepla: {predikce:.2f} GJ/den")

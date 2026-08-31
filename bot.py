import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import requests
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="IA 24/7 Quantum Scanner", layout="wide")
st.title("🤖 Escáner de Arbitraje IA Automatizado (24/7)")
st.write("Servidor activo en segundo plano las 24 horas con refresco automático por hora.")

# Temporizador para auto-ejecutarse cada hora
st_autorefresh(interval=3600 * 1000, key="cron_trading_final")

st.sidebar.header("⚙️ Parámetros del Algoritmo")
universo = st.sidebar.multiselect("Universo:", ["IREN", "CIFR", "WULF", "SLNH", "CORZ"], default=["IREN", "CIFR", "WULF"])
capital_total = st.sidebar.number_input("Capital total (€):", min_value=100, value=1000)
objetivo_rendimiento = st.sidebar.slider("Objetivo Ganancia (%)", 1, 20, 10)

def enviar_alerta_automatica_telegram(mensaje):
    """URL estática y directa en texto plano para eliminar cualquier fallo de parseo"""
    texto_codificado = requests.utils.quote(mensaje)
    
    # Dirección web estática fija e indestructible
    url_fija = f"https://telegram.org{texto_codificado}"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url_fija, headers=headers, timeout=15)
        if response.status_code == 200:
            st.sidebar.success("📢 ¡Mensaje enviado con éxito!")
        else:
            st.sidebar.error(f"Telegram rechazó los datos: {response.text}")
    except Exception as e:
        st.sidebar.error(f"Fallo de conexión en red: {e}")

# Ejecución en datos de 1 hora
tf, per = "60m", "60d"
datos = yf.download(tickers=" ".join(universo), period=per, interval=tf, progress=False)

resultados_globales = []
diccionario_spreads = {}
diccionario_detalles = {}

if not datos.empty:
    if isinstance(datos.columns, pd.MultiIndex):
        precios_totales = datos.xs('Close', axis=1, level=0).dropna()
    else:
        precios_totales = datos['Close'].dropna()
        
    lista_pares = list(combinations(universo, 2))
    
    for t1, t2 in lista_pares:
        if t1 in precios_totales.columns and t2 in precios_totales.columns:
            df_par = precios_totales[[t1, t2]].dropna()
            if len(df_par) < 30: continue
            log_precios = np.log(df_par)
            
            res_joh = coint_johansen(log_precios, det_order=0, k_ar_diff=1)
            
            estadistico_traza = float(res_joh.lr1[0]) 
            valor_critico = float(res_joh.cvt[0, 0])
            
            if estadistico_traza > valor_critico:
                beta_t1 = float(res_joh.evec[0, 0])
                beta_t2 = float(res_joh.evec[1, 0])
                
                spread = (log_precios[t1] * beta_t1) + (log_precios[t2] * beta_t2)
                z_score = (spread - np.mean(spread)) / np.std(spread)
                z_actual = float(z_score.iloc[-1])
                puntuacion_calidad = estadistico_traza - valor_critico
                
                resultados_globales.append({
                    "Par": f"{t1} - {t2}", "Z-Score Actual": round(z_actual, 2),
                    "Fuerza": round(puntuacion_calidad, 4), "Estatus": "🚨 SEÑAL" if np.abs(z_actual) > 2.0 else "⚖️ Equilibrio"
                })
                diccionario_spreads[f"{t1} - {t2}"] = z_score
                diccionario_detalles[f"{t1} - {t2}"] = {
                    "p1": float(df_par[t1].iloc[-1]), "p2": float(df_par[t2].iloc[-1]),
                    "t1": t1, "t2": t2, "z": z_actual, "index": precios_totales.index
                }

if resultados_globales:
    st.subheader("📊 Estado de las Oportunidades Actuales (Auto-Refresh)")
    df_ranking = pd.DataFrame(resultados_globales).sort_values(by="Fuerza", ascending=False).set_index("Par")
    st.dataframe(df_ranking)
    
    for par, info in diccionario_detalles.items():
        z_verificar = info["z"]
        capital_usd = (capital_total / 2.0) * 1.10
        acc_t1 = round(capital_usd / info["p1"])
        acc_t2 = round(capital_usd / info["p2"])
        ganancia_est = capital_total * (objetivo_rendimiento / 100.0)
        
        if z_verificar > 2.0:
            msg = f"ALERTA TRADING IA: El par {par} se ha desviado a un Z-Score de {z_verificar:.2f}. PLAN: CORTO {acc_t1} de {info['t1']} y LARGO {acc_t2} de {info['t2']}."
            enviar_alerta_automatica_telegram(msg)
            st.error(f"🚨 ALERTA ENVIADA: {par}")
        elif z_verificar < -2.0:
            msg = f"ALERTA TRADING IA: El par {par} se ha desviado a un Z-Score de {z_verificar:.2f}. PLAN: LARGO {acc_t1} de {info['t1']} y CORTO {acc_t2} de {info['t2']}."
            enviar_alerta_automatica_telegram(msg)
            st.success(f"🟢 ALERTA ENVIADA: {par}")
            
    st.subheader("🔎 Visor de Control Gráfico")
    par_sel = st.selectbox("Selecciona un par para ver su histórico intradiario:", list(diccionario_spreads.keys()))
    if par_sel:
        df_g = pd.DataFrame({"Z-Score": diccionario_spreads[par_sel]}, index=diccionario_detalles[par_sel]["index"])
        st.line_chart(df_g)
else:
    st.warning("Buscando ineficiencias... Todo el sector IA se mantiene en equilibrio en esta hora.")

# === TEST FORZADO DE CONFIRMACIÓN ===
enviar_alerta_automatica_telegram("¡Conexión definitiva lograda! Tu robot bot.py está escuchando en la nube.")

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
st.write("El servidor se auto-ejecuta cada hora en segundo plano y te alertará al móvil ante ineficiencias.")

# CONFIGURACIÓN AUTOMÁTICA DEL TEMPORIZADOR (3600 segundos = 1 Hora)
st_autorefresh(interval=3600 * 1000, key="cron_trading_ia")

st.sidebar.header("⚙️ Parámetros del Algoritmo")
universo = st.sidebar.multiselect("Universo:", ["IREN", "CIFR", "WULF", "SLNH", "CORZ"], default=["IREN", "CIFR", "WULF"])
capital_total = st.sidebar.number_input("Capital total (€):", min_value=100, value=1000)
objetivo_rendimiento = st.sidebar.slider("Objetivo Ganancia (%)", 1, 20, 10)

def enviar_alerta_automatica_telegram(mensaje):
    """Manda notificaciones al móvil con una estructura de URL blindada contra errores de sintaxis"""
    # Separamos el prefijo 'bot' del código para evitar errores de interpretación del servidor
    token = "8948061031:AAF-hZXlXcoolKy9QZAwj2_gLTMr_GOWjZU"
    chat_id = "399072608"
    
    # Construcción limpia de la URL sin caracteres cruzados
    url_base = "https://telegram.org"
    url_final = url_base + "/bot" + token + "/sendMessage"
    
    payload = {"chat_id": str(chat_id), "text": mensaje, "parse_mode": "Markdown"}
    
    try:
        # Añadimos un agente de usuario estándar para que parezca una petición web normal de navegador
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.post(url_final, data=payload, headers=headers, timeout=15)
        
        # Mostramos el diagnóstico de éxito en la barra lateral en silencio para validar
        if response.status_code == 200:
            st.sidebar.success("📢 Mensaje enviado con éxito")
        else:
            st.sidebar.error(f"Telegram rechazó los datos: {response.text}")
    except Exception as e:
        st.sidebar.error(f"Error de red: {e}")

# Ejecución continua en datos intradiarios de 1 hora
tf, per = "60m", "60d"

# El algoritmo descarga datos de forma autónoma
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
            
            # Extraemos de forma matemática estricta usando posiciones
            estadistico_traza = float(res_joh.lr1[0]) 
            valor_critico = float(res_joh.cvt[0, 0])   # Confianza al 90% intradiario
            
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
    
    # Procesamiento pasivo automático de alertas críticas
    for par, info in diccionario_detalles.items():
        z_verificar = info["z"]
        capital_usd = (capital_total / 2.0) * 1.10
        acc_t1 = round(capital_usd / info["p1"])
        acc_t2 = round(capital_usd / info["p2"])
        ganancia_est = capital_total * (objetivo_rendimiento / 100.0)
        
        if z_verificar > 2.0:
            msg = f"🚨 *ALERTA DE TRADING IA ACTIVADA*\n\nEl par *{par}* se ha desviado a un Z-Score crítico de *{z_verificar:.2f}*.\n\n🔴 CORTO {acc_t1} de {info['t1']}\n🟢 LARGO {acc_t2} de {info['t2']}\n\n🎯 *Target:* +{ganancia_est:.2f}€ | ⚠️ *Stop Loss:* Z=3.50"
            enviar_alerta_automatica_telegram(msg)
            st.error(f"🚨 ALERTA ENVIADA: {par}")
            
        elif z_verificar < -2.0:
            msg = f"🚨 *ALERTA DE TRADING IA ACTIVADA*\n\nEl par *{par}* se ha desviado a un Z-Score crítico de *{z_verificar:.2f}*.\n\n🟢 LARGO {acc_t1} de {info['t1']}\n🔴 CORTO {acc_t2} de {info['t2']}\n\n🎯 *Target:* +{ganancia_est:.2f}€ | ⚠️ *Stop Loss:* Z=-3.50"
            enviar_alerta_automatica_telegram(msg)
            st.success(f"🟢 ALERTA ENVIADA: {par}")
            
    # Visor gráfico manual de apoyo
    st.subheader("🔎 Visor de Control Gráfico")
    par_sel = st.selectbox("Selecciona un par para ver su histórico intradiario:", list(diccionario_spreads.keys()))
    if par_sel:
        df_g = pd.DataFrame({"Z-Score": diccionario_spreads[par_sel]}, index=diccionario_detalles[par_sel]["index"])
        st.line_chart(df_g)
else:
    st.warning("Buscando ineficiencias... Todo el sector IA se mantiene en equilibrio en esta hora.")

# === TEST FORZADO AUTOMÁTICO DE CONFIRMACIÓN ===
# Borraremos estas dos líneas de abajo una vez que compruebes que suena tu móvil
enviar_alerta_automatica_telegram("🚀 ¡Prueba de Red Exitosa! El servidor y tu Telegram están vinculados sin errores.")

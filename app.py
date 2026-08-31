import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# === CONFIGURACIÓN DE ALERTAS DE TELEGRAM ===
TELEGRAM_TOKEN = "8948061031:AAF-hZXlXcoolKy9QZAwj2_gLTMr_GOWjZU"
TELEGRAM_CHAT_ID = "399072608"

def enviar_alerta_telegram(mensaje):
    """Manda el mensaje a tu móvil usando la ruta oficial de la API"""
    url = f"https://telegram.org{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        pass

st.set_page_config(page_title="Calculadora Cuántica de Pares", layout="wide")
st.title("💰 Monitor Sectorial con Alertas al Móvil")
st.write("Tu servidor está conectado a Yahoo Finance. Presiona el botón para escanear.")

capital_total = st.sidebar.number_input("Capital total a invertir (€):", min_value=100, value=1000)

if st.button("🧮 VERIFICAR MERCADO Y ENVIAR ALERTA"):
    st.info("Escaneando mercado actual...")
    
    # Descargamos los datos históricos
    datos = yf.download(tickers="IREN CIFR", period="1y", interval="1d")
    
    if not datos.empty:
        # --- SOLUCIÓN AL MULTI-INDEX DE YFINANCE ---
        # Si Yahoo Finance entrega las columnas agrupadas, las aplanamos
        if isinstance(datos.columns, pd.MultiIndex):
            # Nos quedamos solo con las filas que pertenecen al precio de Cierre ('Close')
            precios = datos.xs('Close', axis=1, level=0).dropna()
        else:
            precios = datos['Close'].dropna()
            
        log_precios = np.log(precios)
        
        # Matemáticas del Spread y Z-Score con Johansen
        res_johansen = coint_johansen(log_precios, det_order=0, k_ar_diff=1)
        
        # Extraemos el primer vector de cointegración (Hedge Ratio)
        beta = res_johansen.evec[:, 0]
        
        precio_actual_iren = float(precios['IREN'].iloc[-1])
        precio_actual_cifr = float(precios['CIFR'].iloc[-1])
        
        # Calculamos el spread dinámico exacto
        spread = (log_precios['IREN'] * beta[0]) + (log_precios['CIFR'] * beta[1])
        z_score = (spread - np.mean(spread)) / np.std(spread)
        z_actual = float(z_score.iloc[-1])
        
        # Gestión de capital para los 1,000€ asignados
        capital_usd = (capital_total / 2.0) * 1.10 
        acciones_iren = round(capital_usd / precio_actual_iren)
        acciones_cifr = round(capital_usd / precio_actual_cifr)
        
        st.subheader("📌 Diagnóstico del Spread Hoy")
        
        # --- LÓGICA DE ALERTA AUTOMÁTICA ---
        if z_actual > 2.0:
            msg = f"🚨 *ALERTA TRADING IA*\n\nEl Z-Score está disparado en *{z_actual:.2f}*.\n\n*Operación sugerida:*\n🔴 VENDER EN CORTO {acciones_iren} acciones de IREN\n🟢 COMPRAR {acciones_cifr} acciones de CIFR"
            st.error(msg)
            enviar_alerta_telegram(msg)
            
        elif z_actual < -2.0:
            msg = f"🚨 *ALERTA TRADING IA*\n\nEl Z-Score está hundido en *{z_actual:.2f}*.\n\n*Operación sugerida:*\n🟢 COMPRAR {acciones_iren} acciones de IREN\n🔴 VENDER EN CORTO {acciones_cifr} acciones de CIFR"
            st.success(msg)
            enviar_alerta_telegram(msg)
            
        else:
            st.info(f"⚖️ El par está en equilibrio (Z-Score: {z_actual:.2f}). No requiere operaciones directas.")
            # Confirmación forzada para verificar que el servidor y el bot se comunican
            enviar_alerta_telegram(f"🔔 ¡Botón pulsado! Servidor conectado con éxito. El Z-Score de equilibrio es {z_actual:.2f}.")
            
        # === GRÁFICO DE CONTROL AJUSTADO ===
        st.subheader("📈 Gráfico de Control Histórico")
        df_grafico = pd.DataFrame({"Z-Score del Spread": z_score}, index=precios.index)
        st.line_chart(df_grafico)
        
    else:
        st.error("No se pudieron obtener datos desde los servidores de Yahoo Finance.")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

st.set_page_config(page_title="Calculadora Cuántica de Pares", layout="wide")
st.title("💰 Calculadora de Órdenes: Estrategia 1,000€")
st.write("Calcula automáticamente el número exacto de acciones utilizando los precios reales del mercado.")

# 1. Configuración de Capital en la Barra Lateral
st.sidebar.header("💵 Tu Configuración Financiera")
capital_total = st.sidebar.number_input("Capital total a invertir (€):", min_value=100, max_value=100000, value=1000, step=100)

if st.button("🧮 CALCULAR TAMAÑO DE POSICIÓN EXACTO"):
    st.info("Obteniendo precios en tiempo real...")
    
    # Descargamos los datos históricos para Johansen y el precio actual
    datos = yf.download(tickers="IREN CIFR", period="1y", interval="1d")
    
    if not datos.empty:
        precios = datos['Close'].dropna()
        log_precios = np.log(precios)
        
        # Test de Johansen para extraer el Vector Beta exacto
        res_johansen = coint_johansen(log_precios, det_order=0, k_ar_diff=1)
        beta_iren = res_johansen.evec[0, 0]
        beta_cifr = res_johansen.evec[1, 0]
        
        # Obtener los últimos precios de mercado en dólares
        precio_actual_iren = float(precios['IREN'].iloc[-1])
        precio_actual_cifr = float(precios['CIFR'].iloc[-1])
        
        # Calcular el Z-Score actual
        spread = (log_precios['IREN'] * beta_iren) + (log_precios['CIFR'] * beta_cifr)
        z_score = (spread - np.mean(spread)) / np.std(spread)
        z_actual = float(z_score.iloc[-1])
        
        # --- LÓGICA DE DISTRIBUCIÓN DE CAPITAL ---
        # Dividimos el capital a la mitad (Neutralidad de mercado: 50% largo, 50% corto)
        capital_por_lado = capital_total / 2.0
        
        # Asumiendo un tipo de cambio aproximado de EUR a USD de 1.10 (para brokers de EEUU)
        capital_usd = capital_por_lado * 1.10 
        
        # Cálculo del número de acciones
        acciones_iren = round(capital_usd / precio_actual_iren)
        acciones_cifr = round(capital_usd / precio_actual_cifr)
        
        # 2. Despliegue de Resultados y Señales de Entrada
        st.subheader("📊 Diagnóstico del Spread Hoy")
        
        if z_actual > 2.0:
            st.error(f"🚨 SEÑAL ACTIVA: Z-Score en {z_actual:.2f} (IREN sobrevalorada / CIFR infravalorada)")
            st.markdown(f"""
            ### 📉 Plan de Ejecución para tus **{capital_total}€**:
            *   🔴 **VENTA EN CORTO (Short):** Vender **{acciones_iren} acciones** de **IREN** (Precio actual: ${precio_actual_iren:.2f})
            *   🟢 **COMPRA (Long):** Comprar **{acciones_cifr} acciones** de **CIFR** (Precio actual: ${precio_actual_cifr:.2f})
            """)
        elif z_actual < -2.0:
            st.success(f"🚨 SEÑAL ACTIVA: Z-Score en {z_actual:.2f} (IREN infravalorada / CIFR sobrevalorada)")
            st.markdown(f"""
            ### 📈 Plan de Ejecución para tus **{capital_total}€**:
            *   🟢 **COMPRA (Long):** Comprar **{acciones_iren} acciones** de **IREN** (Precio actual: ${precio_actual_iren:.2f})
            *   🔴 **VENTA EN CORTO (Short):** Vender **{acciones_cifr} acciones** de **CIFR** (Precio actual: ${precio_actual_cifr:.2f})
            """)
        else:
            st.info(f"⚖️ El par está en rango normal (Z-Score: {z_actual:.2f}). No se recomienda abrir posiciones aún.")
            st.write(f"Si forzaras la entrada ahora mismo con **{capital_total}€**, tu distribución neutral sería:")
            
            # Mostrar tabla informativa de precios
            df_ordenes = pd.DataFrame({
                "Precio Mercado ($)": [precio_actual_iren, precio_actual_cifr],
                "Asignación sugerida (€)": [capital_por_lado, capital_por_lado],
                "Cantidad de Acciones": [acciones_iren, acciones_cifr]
            }, index=["IREN", "CIFR"])
            st.dataframe(df_ordenes.style.format("{:.2f}", subset=["Precio Mercado ($)", "Asignación sugerida (€)"]))
            
        # Graficamos el histórico
        st.subheader("📈 Monitor de Entrada")
        st.line_chart(z_score)
        
    else:
        st.error("No se pudieron obtener datos.")

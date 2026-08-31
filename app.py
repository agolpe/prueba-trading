import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

st.set_page_config(page_title="Calculadora Cuántica de Pares", layout="wide")
st.title("💰 Monitor Sectorial con Enlace de Alertas al Móvil")
st.write("Tu servidor está conectado a Yahoo Finance. Presiona el botón para escanear.")

capital_total = st.sidebar.number_input("Capital total a invertir (€):", min_value=100, value=1000)

if st.button("🧮 VERIFICAR MERCADO Y GENERAR ÓRDENES"):
    st.info("Escaneando mercado actual...")
    
    # 1. Ingesta segura de datos
    datos = yf.download(tickers="IREN CIFR", period="1y", interval="1d")
    
    if not datos.empty:
        if isinstance(datos.columns, pd.MultiIndex):
            precios = datos.xs('Close', axis=1, level=0).dropna()
        else:
            precios = datos['Close'].dropna()
            
        log_precios = np.log(precios)
        
        # 2. Test de Johansen
        res_johansen = coint_johansen(log_precios, det_order=0, k_ar_diff=1)
        
        beta_iren = float(res_johansen.evec[0, 0])
        beta_cifr = float(res_johansen.evec[1, 0])
        
        precio_actual_iren = float(precios['IREN'].iloc[-1])
        precio_actual_cifr = float(precios['CIFR'].iloc[-1])
        
        # 3. Cálculo del Spread y Z-Score
        spread = (log_precios['IREN'] * beta_iren) + (log_precios['CIFR'] * beta_cifr)
        z_score = (spread - np.mean(spread)) / np.std(spread)
        z_actual = float(z_score.iloc[-1])
        
        # 4. Gestión de posiciones para tus 1,000€ asignados
        capital_usd = (capital_total / 2.0) * 1.10 
        acciones_iren = round(capital_usd / precio_actual_iren)
        acciones_cifr = round(capital_usd / precio_actual_cifr)
        
        st.subheader("📌 Diagnóstico del Spread Hoy")
        
        # Configuración base del enlace web de Telegram (Esquiva el bloqueo del servidor)
        token_bot = "8948061031:AAF-hZXlXcoolKy9QZAwj2_gLTMr_GOWjZU"
        chat_id_usuario = "399072608"
        
        if z_actual > 2.0:
            texto_alerta = f"🚨 ALERTA TRADING IA %0A%0AEl Z-Score está disparado en {z_actual:.2f}.%0A%0A*Operación sugerida:*%0A🔴 VENDER EN CORTO {acciones_iren} de IREN%0A🟢 COMPRAR {acciones_cifr} de CIFR"
            st.error(f"🚨 ALERTA ACTIVA: El Z-Score está en {z_actual:.2f}")
            
            # Botón web interactivo que hereda la URL que te dio "ok: true"
            url_telegram = f"https://api.telegram.org/bot{token_bot}/sendMessage?chat_id={chat_id_usuario}&text={texto_alerta}"
            st.link_button("📲 ENVIAR ORDEN AL MÓVIL", url_telegram)
            
        elif z_actual < -2.0:
            texto_alerta = f"🚨 ALERTA TRADING IA %0A%0AEl Z-Score está hundido en {z_actual:.2f}.%0A%0A*Operación sugerida:*%0A🟢 COMPRAR {acciones_iren} de IREN%0A🔴 VENDER EN CORTO {acciones_cifr} de CIFR"
            st.success(f"🚨 ALERTA ACTIVA: El Z-Score está en {z_actual:.2f}")
            
            url_telegram = f"https://api.telegram.org/bot{token_bot}/sendMessage?chat_id={chat_id_usuario}&text={texto_alerta}"
            st.link_button("📲 ENVIAR ORDEN AL MÓVIL", url_telegram)
            
        else:
            st.info(f"⚖️ El par está en equilibrio (Z-Score: {z_actual:.2f}). No requiere operaciones directas.")
            st.write(f"Si forzaras la entrada de equilibrio ahora mismo con **{capital_total}€**, tu distribución neutral sería:")
            
            df_ordenes = pd.DataFrame({
                "Precio Mercado ($)": [precio_actual_iren, precio_actual_cifr],
                "Asignación sugerida (€)": [capital_total / 2, capital_total / 2],
                "Cantidad de Acciones": [acciones_iren, acciones_cifr]
            }, index=["IREN", "CIFR"])
            st.dataframe(df_ordenes)
            
            # Enviamos un enlace de prueba de equilibrio
            texto_prueba = f"🔔 Sistema Online. El par IREN/CIFR está balanceado hoy (Z-Score: {z_actual:.2f})."
            url_telegram = f"https://api.telegram.org/bot{token_bot}/sendMessage?chat_id={chat_id_usuario}&text={texto_prueba}"
            st.link_button("📲 ENVIAR MENSAJE DE PRUEBA AL MÓVIL", url_telegram)
            
        # 5. Dibujo del Gráfico Histórico
        st.subheader("📈 Gráfico de Control Histórico")
        df_grafico = pd.DataFrame({"Z-Score del Spread": z_score}, index=precios.index)
        st.line_chart(df_grafico)
        
    else:
        st.error("No se pudieron obtener datos desde los servidores de Yahoo Finance.")

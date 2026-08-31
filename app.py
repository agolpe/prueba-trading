import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen

st.set_page_config(page_title="IA Pairs Trading Matrix", layout="wide")
st.title("⚡ Escáner de Arbitraje IA de Alta Frecuencia")
st.write("Análisis matricial intradiario, objetivos de beneficio y gestión de riesgos.")

# 1. Barra Lateral - Configuración del Universo y Riesgo
st.sidebar.header("⚙️ Configuración del Universo")
universo = st.sidebar.multiselect(
    "Compañías a escanear:",
    ["IREN", "CIFR", "WULF", "SLNH", "CORZ"],
    default=["IREN", "CIFR", "WULF"]
)

frecuencia = st.sidebar.selectbox(
    "Frecuencia de Datos Intradía:",
    ["1 Hora (Últimos 60 días)", "5 Minutos (Últimos 60 días)"]
)

capital_total = st.sidebar.number_input("Capital total a invertir (€):", min_value=100, value=1000)
objetivo_rendimiento = st.sidebar.slider("Objetivo de Ganancia (%)", 1, 20, 10)

# Mapeo de temporalidades oficiales de Yahoo Finance
tf = "60m" if "1 Hora" in frecuencia else "5m"

if st.sidebar.button("🚀 INICIAR ESCANEO SECTORIAL"):
    if len(universo) < 2:
        st.error("Por favor selecciona al menos 2 compañías para buscar combinaciones.")
    else:
        st.info(f"Descargando datos intradiarios y procesando combinaciones...")
        
        # Descarga masiva del universo seleccionado
        datos = yf.download(tickers=" ".join(universo), period="60d", interval=tf)
        
        if not datos.empty:
            # Aplanamos Multi-Index de yfinance si es necesario
            if isinstance(datos.columns, pd.MultiIndex):
                precios_totales = datos.xs('Close', axis=1, level=0).dropna()
            else:
                precios_totales = datos['Close'].dropna()
                
            lista_pares = list(combinations(universo, 2))
            resultados_globales = []
            diccionario_spreads = {}
            diccionario_detalles = {}
            
            # 2. Bucle de Análisis Estadístico Par por Par
            for t1, t2 in lista_pares:
                if t1 in precios_totales.columns and t2 in precios_totales.columns:
                    df_par = precios_totales[[t1, t2]].dropna()
                    if len(df_par) < 40: continue
                    
                    log_precios = np.log(df_par)
                    
                    # Test de Cointegración de Johansen
                    res_joh = coint_johansen(log_precios, det_order=0, k_ar_diff=1)
                    
                    # Verificamos si supera el valor crítico al 95% de confianza
                    estadistico_traza = res_joh.lr1[0]
                    valor_critico_95 = res_joh.cvt[0, 1]
                    esta_coint = estadistico_traza > valor_critico_95
                    
                    if esta_coint:
                        # Extraemos coeficientes numéricos planos
                        beta_t1 = float(res_joh.evec[0, 0])
                        beta_t2 = float(res_joh.evec[1, 0])
                        
                        # Cálculo del spread y Z-Score histórico intradiario
                        spread = (log_precios[t1] * beta_t1) + (log_precios[t2] * beta_t2)
                        z_score = (spread - np.mean(spread)) / np.std(spread)
                        z_actual = float(z_score.iloc[-1])
                        
                        # Puntuación de calidad del par (A mayor estadística de traza, más estable es la relación)
                        puntuacion_calidad = estadistico_traza - valor_critico_95
                        
                        resultados_globales.append({
                            "Par": f"{t1} - {t2}",
                            "Z-Score Actual": z_actual,
                            "Fuerza Cointegración": puntuacion_calidad,
                            "Estatus": "🚨 ¡SEÑAL ACTIVA!" if np.abs(z_actual) > 2.0 else "⚖️ Equilibrio"
                        })
                        
                        diccionario_spreads[f"{t1} - {t2}"] = z_score
                        diccionario_detalles[f"{t1} - {t2}"] = {
                            "p1": float(df_par[t1].iloc[-1]),
                            "p2": float(df_par[t2].iloc[-1]),
                            "t1": t1, "t2": t2, "z": z_actual
                        }
            
            # 3. Despliegue de Resultados y Ranking en Pantalla
            if resultados_globales:
                st.subheader("📊 Ranking de Oportunidades en el Sector IA")
                df_ranking = pd.DataFrame(resultados_globales).sort_values(by="Fuerza Cointegración", ascending=False).set_index("Par")
                st.dataframe(df_ranking)
                
                # Selector dinámico del par que el usuario desea operar
                st.subheader("🎯 Analizador de Ejecución y Gestión de Riesgo")
                par_elegido = st.selectbox("Selecciona el par que deseas analizar a fondo:", list(diccionario_spreads.keys()))
                
                if par_elegido:
                    info = diccionario_detalles[par_elegido]
                    z_grafico = diccionario_spreads[par_elegido]
                    
                    # Cálculo de acciones neutrales al mercado
                    capital_usd = (capital_total / 2.0) * 1.10
                    acciones_t1 = round(capital_usd / info["p1"])
                    acciones_t2 = round(capital_usd / info["p2"])
                    
                    # --- CÁLCULO ESTÁTICO DE RECOLECCIÓN DE BENEFICIOS Y STOP LOSS ---
                    # Dinámica de ganancias: buscamos el 10% del capital total
                    ganancia_euros = capital_total * (objetivo_rendimiento / 100.0)
                    
                    st.markdown(f"### 📋 Ficha de Operación: {par_elegido}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Z-Score de Entrada", f"{info['z']:.2f}")
                    with col2:
                        st.metric(f"🎯 Objetivo de Ganancia ({objetivo_rendimiento}%)", f"+{ganancia_euros:.2f} €")
                    with col3:
                        # Gestión del Stop Loss si el Z-score se desborda a un extremo crítico
                        st.metric("⚠️ Límite de Stop Loss Crítico", "Z = ±3.50")
                        
                    # Definición exacta del plan estratégico según el desvío intradiario
                    st.subheader("🛠️ Instrucciones de Entrada y Salida")
                    token_bot = "8948061031:AAF-hZXlXcoolKy9QZAwj2_gLTMr_GOWjZU"
                    chat_id_usuario = "399072608"
                    
                    if info["z"] > 2.0:
                        plan_texto = f"Entrada activada en {par_elegido}. Vender corto {acciones_t1} acciones de {info['t1']} y comprar {acciones_t2} acciones de {info['t2']}. Cierre con toma de ganancias cuando el Z-Score regrese a 0 (Ganancia estimada: {ganancia_euros}€). Ejecutar Stop Loss definitivo si el Z-Score rompe el nivel de 3.50."
                        st.error(f"🔴 ESTRATEGIA: VENTA DEL SPREAD ACTIVADA")
                        st.write(plan_texto)
                        
                        url_tel = f"https://telegram.org{token_bot}/sendMessage?chat_id={chat_id_usuario}&text=🚨 ORDEN ACTIVADA INTRADÍA: {plan_texto}"
                        st.link_button("📲 ENVIAR PLAN DE TRADING AL MÓVIL", url_tel)
                        
                    elif info["z"] < -2.0:
                        plan_texto = f"Entrada activada en {par_elegido}. Comprar {acciones_t1} acciones de {info['t1']} y vender corto {acciones_t2} acciones de {info['t2']}. Cierre con toma de ganancias cuando el Z-Score regrese a 0 (Ganancia estimada: {ganancia_euros}€). Ejecutar Stop Loss definitivo si el Z-Score cae por debajo de -3.50."
                        st.success(f"🟢 ESTRATEGIA: COMPRA DEL SPREAD ACTIVADA")
                        st.write(plan_texto)
                        
                        url_tel = f"https://telegram.org{token_bot}/sendMessage?chat_id={chat_id_usuario}&text=🚨 ORDEN ACTIVADA INTRADÍA: {plan_texto}"
                        st.link_button("📲 ENVIAR PLAN DE TRADING AL MÓVIL", url_tel)
                        
                    else:
                        plan_texto = f"El par {par_elegido} está balanceado en datos intradiarios. No se abren operaciones de arbitraje hoy."
                        st.info(f"⚖️ POSICIÓN: ESPERANDO DESVÍO DE MERCADO")
                        st.write(f"Si forzaras la entrada para ganar el {objetivo_rendimiento}%, comprarías {acciones_t1} de {info['t1']} y venderías {acciones_t2} de {info['t2']}, cerrando la operación estrictamente en Z = 0.")
                        
                        url_tel = f"https://telegram.org{token_bot}/sendMessage?chat_id={chat_id_usuario}&text=🔔 Alerta de Control: {plan_texto} (Z={info['z']:.2f})"
                        st.link_button("📲 ENVIAR REPORTE DE CONTROL AL MÓVIL", url_tel)
                        
                    # Gráfico de Control Intradiario
                    st.subheader("📈 Gráfico de Control Intradiario")
                    df_grafico = pd.DataFrame({"Z-Score del Spread": z_grafico}, index=precios_totales.index)
                    st.line_chart(df_grafico)
            else:
                st.warning("No se detectaron pares cointegrados al 95% entre las compañías seleccionadas en este intervalo de tiempo.")
        else:
            st.error("Error al descargar el histórico intradiario desde Yahoo Finance.")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen

st.set_page_config(page_title="IA Pairs Trading Matrix", layout="wide")
st.title("⚡ Escáner de Arbitraje IA Multitemporal")
st.write("Análisis matricial macro y micro, objetivos de beneficio y gestión de riesgos.")

# 1. Configuración del menú en la Barra Lateral
st.sidebar.header("⚙️ Configuración del Universo")
universo = st.sidebar.multiselect(
    "Compañías a escanear:",
    ["IREN", "CIFR", "WULF", "SLNH", "CORZ"],
    default=["IREN", "CIFR", "WULF"]
)

frecuencia = st.sidebar.selectbox(
    "Frecuencia de Datos / Horizonte:",
    [
        "Diario (Macro - Último 1 Año)", 
        "1 Hora (Intradía - Últimos 60 días)", 
        "5 Minutos (Alta Frecuencia - Últimos 60 días)"
    ]
)

capital_total = st.sidebar.number_input("Capital total a invertir (€):", min_value=100, value=1000)
objetivo_rendimiento = st.sidebar.slider("Objetivo de Ganancia (%)", 1, 20, 10)

# Mapeo dinámico de parámetros para Yahoo Finance
if "Diario" in frecuencia:
    tf, per = "1d", "1y"
elif "1 Hora" in frecuencia:
    tf, per = "60m", "60d"
else:
    tf, per = "5m", "60d"

if st.sidebar.button("🚀 INICIAR ESCANEO SECTORIAL"):
    if len(universo) < 2:
        st.error("Por favor selecciona al menos 2 compañías para buscar combinaciones.")
    else:
        st.info(f"Descargando datos ({frecuencia}) y procesando combinaciones...")
        
        # Descarga de datos históricos de los activos elegidos
        datos = yf.download(tickers=" ".join(universo), period=per, interval=tf)
        
        if not datos.empty:
            # Aplanamos el Multi-Index de yfinance de forma segura
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
                    if len(df_par) < 30: continue
                    
                    log_precios = np.log(df_par)
                    
                    # Ejecutamos el Test de Cointegración de Johansen
                    res_joh = coint_johansen(log_precios, det_order=0, k_ar_diff=1)
                    
                    # --- EXTRACCIÓN EXTRAESTRICTA DE VALORES CRÍTICOS (Solución al ValueError) ---
                    estadistico_traza = float(res_joh.lr1[0]) # Extraemos el valor del rango r=0
                    
                    if tf == "1d":
                        valor_critico = float(res_joh.cvt[0, 1])  # Columna 1 = Confianza al 95%
                        criterio_texto = "95% Confianza (Macro)"
                    else:
                        valor_critico = float(res_joh.cvt[0, 0])  # Columna 0 = Confianza al 90%
                        criterio_texto = "90% Confianza (Intradía)"
                        
                    esta_coint = estadistico_traza > valor_critico
                    
                    if esta_coint:
                        # Coordenadas matriciales fijas de los autovectores
                        beta_t1 = float(res_joh.evec[0, 0])
                        beta_t2 = float(res_joh.evec[1, 0])
                        
                        # Cálculo matemático del Spread y Z-Score
                        spread = (log_precios[t1] * beta_t1) + (log_precios[t2] * beta_t2)
                        z_score = (spread - np.mean(spread)) / np.std(spread)
                        z_actual = float(z_score.iloc[-1])
                        
                        puntuacion_calidad = estadistico_traza - valor_critico
                        
                        resultados_globales.append({
                            "Par": f"{t1} - {t2}",
                            "Z-Score Actual": round(z_actual, 2),
                            "Fuerza Relación": round(puntuacion_calidad, 4),
                            "Filtro Aplicado": criterio_texto,
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
                st.subheader(f"📊 Ranking de Estabilidad Sectorial ({frecuencia})")
                df_ranking = pd.DataFrame(resultados_globales).sort_values(by="Fuerza Relación", ascending=False).set_index("Par")
                st.dataframe(df_ranking)
                
                # 4. Sección de Gestión de Riesgo y Cálculo de Órdenes
                st.subheader("🎯 Analizador de Ejecución y Gestión de Riesgo")
                par_elegido = st.selectbox("Selecciona un par cointegrado para desplegar el plan operativo:", list(diccionario_spreads.keys()))
                
                if par_elegido:
                    info = diccionario_detalles[par_elegido]
                    z_grafico = diccionario_spreads[par_elegido]
                    
                    # Cálculo neutral monetario (€ a USD)
                    capital_usd = (capital_total / 2.0) * 1.10
                    acciones_t1 = round(capital_usd / info["p1"])
                    acciones_t2 = round(capital_usd / info["p2"])
                    
                    ganancia_euros = capital_total * (objetivo_rendimiento / 100.0)
                    
                    st.markdown(f"### 📋 Ficha Operativa Real: {par_elegido}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Z-Score Actual", f"{info['z']:.2f}")
                    with col2:
                        st.metric(f"🎯 Objetivo de Ganancia ({objetivo_rendimiento}%)", f"+{ganancia_euros:.2f} €")
                    with col3:
                        st.metric("⚠️ Límite de Stop Loss Crítico", "Z = ±3.50")
                        
                    # Configuración de enlaces a Telegram
                    token_bot = "8948061031:AAF-hZXlXcoolKy9QZAwj2_gLTMr_GOWjZU"
                    chat_id_usuario = "399072608"
                    
                    if info["z"] > 2.0:
                        plan_texto = f"Entrada en {par_elegido}. Vender corto {acciones_t1} acciones de {info['t1']} y comprar {acciones_t2} acciones de {info['t2']}. Cierre en Z=0 para asegurar +{ganancia_euros}€. Stop Loss definitivo si rompe Z=3.50."
                        st.error(f"🔴 ESTRATEGIA OPERATIVA: VENTA DEL SPREAD ACTIVADA")
                        st.write(plan_texto)
                        
                        url_tel = f"https://telegram.org{token_bot}/sendMessage?chat_id={chat_id_usuario}&text=🚨 ORDEN SUGERIDA: {plan_texto}"
                        st.link_button("📲 ENVIAR PLAN DE TRADING AL MÓVIL", url_tel)
                        
                    elif info["z"] < -2.0:
                        plan_texto = f"Entrada en {par_elegido}. Comprar {acciones_t1} acciones de {info['t1']} y vender corto {acciones_t2} acciones de {info['t2']}. Cierre en Z=0 para asegurar +{ganancia_euros}€. Stop Loss definitivo si cae de Z=-3.50."
                        st.success(f"🟢 ESTRATEGIA OPERATIVA: COMPRA DEL SPREAD ACTIVADA")
                        st.write(plan_texto)
                        
                        url_tel = f"https://telegram.org{token_bot}/sendMessage?chat_id={chat_id_usuario}&text=🚨 ORDEN SUGERIDA: {plan_texto}"
                        st.link_button("📲 ENVIAR PLAN DE TRADING AL MÓVIL", url_tel)
                        
                    else:
                        plan_texto = f"El par {par_elegido} está balanceado en este horizonte ({frecuencia}). No hay distorsiones arbitrales."
                        st.info(f"⚖️ POSICIÓN: ESPERANDO MOVIMIENTO DE BANDAS")
                        st.write(f"Para obtener el {objetivo_rendimiento}% de beneficio al regresar al centro (Z=0), el plan neutral asignará {acciones_t1} acciones de {info['t1']} y {acciones_t2} acciones de {info['t2']}.")
                        
                        url_tel = f"https://telegram.org{token_bot}/sendMessage?chat_id={chat_id_usuario}&text=🔔 Reporte de Control ({frecuencia}): {par_elegido} equilibrado en Z={info['z']:.2f}"
                        st.link_button("📲 ENVIAR REPORTE DE CONTROL AL MÓVIL", url_tel)
                        
                    # 5. Dibujo del Gráfico Histórico
                    st.subheader(f"📈 Gráfico de Control Histórico ({frecuencia})")
                    df_grafico = pd.DataFrame({"Z-Score del Spread": z_grafico}, index=precios_totales.index)
                    st.line_chart(df_grafico)
            else:
                st.warning("No se detectaron pares válidos que superen los criterios de cointegración elegidos en este periodo.")
        else:
            st.error("Error al descargar el histórico desde los servidores financieros de Yahoo Finance.")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen

st.set_page_config(page_title="IA Pairs Trading Matrix", layout="wide")
st.title("⚡ Escáner de Arbitraje IA Multitemporal")
st.write("Análisis matricial macro y micro, objetivos de beneficio y gestión de riesgos.")

if "datos_escaner" not in st.session_state:
    st.session_state["datos_escaner"] = None
if "contador_ejecucion" not in st.session_state:
    st.session_state["contador_ejecucion"] = 0

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

if "Diario" in frecuencia:
    tf, per = "1d", "1y"
elif "1 Hora" in frecuencia:
    tf, per = "60m", "60d"
else:
    tf, per = "5m", "60d"

if st.sidebar.button("🚀 INICIAR ESCANEO SECTORIAL"):
    if len(universo) < 2:
        st.error("Por favor selecciona al menos 2 compañías.")
    else:
        st.info(f"Descargando datos ({frecuencia})...")
        datos = yf.download(tickers=" ".join(universo), period=per, interval=tf)
        
        if not datos.empty:
            if isinstance(datos.columns, pd.MultiIndex):
                precios_totales = datos.xs('Close', axis=1, level=0).dropna()
            else:
                precios_totales = datos['Close'].dropna()
                
            lista_pares = list(combinations(universo, 2))
            resultados_globales = []
            diccionario_spreads = {}
            diccionario_detalles = {}
            
            for t1, t2 in lista_pares:
                if t1 in precios_totales.columns and t2 in precios_totales.columns:
                    df_par = precios_totales[[t1, t2]].dropna()
                    if len(df_par) < 30: continue
                    log_precios = np.log(df_par)
                    
                    res_joh = coint_johansen(log_precios, det_order=0, k_ar_diff=1)
                    estadistico_traza = float(res_joh.lr1[0]) 
                    
                    if tf == "1d":
                        valor_critico = float(res_joh.cvt[0, 1])  
                        criterio_texto = "95% Confianza (Macro)"
                    else:
                        valor_critico = float(res_joh.cvt[0, 0])  
                        criterio_texto = "90% Confianza (Intradía)"
                        
                    esta_coint = estadistico_traza > valor_critico
                    
                    if esta_coint:
                        beta_t1 = float(res_joh.evec[0, 0])
                        beta_t2 = float(res_joh.evec[1, 0])
                        
                        spread = (log_precios[t1] * beta_t1) + (log_precios[t2] * beta_t2)
                        z_score = (spread - np.mean(spread)) / np.std(spread)
                        z_actual = float(z_score.iloc[-1])
                        puntuacion_calidad = estadistico_traza - valor_critico
                        
                        resultados_globales.append({
                            "Par": f"{t1} - {t2}",
                            "Z-Score Actual": round(z_actual, 2),
                            "Fuerza Relación": round(puntuacion_calidad, 4),
                            "Filtro Aplicado": criterio_texto,
                            "Estatus": "🚨 SEÑAL" if np.abs(z_actual) > 2.0 else "⚖️ Equilibrio"
                        })
                        diccionario_spreads[f"{t1} - {t2}"] = z_score
                        diccionario_detalles[f"{t1} - {t2}"] = {
                            "p1": float(df_par[t1].iloc[-1]), "p2": float(df_par[t2].iloc[-1]),
                            "t1": t1, "t2": t2, "z": z_actual, "precios_index": precios_totales.index
                        }
            if resultados_globales:
                st.session_state["datos_escaner"] = {"ranking": resultados_globales, "spreads": diccionario_spreads, "detalles": diccionario_detalles}
                st.session_state["contador_ejecucion"] += 1
            else:
                st.session_state["datos_escaner"] = None
                st.warning("No se detectaron pares válidos.")
if st.session_state["datos_escaner"] is not None:
    datos_actuales = st.session_state["datos_escaner"]
    
    st.subheader(f"📊 Ranking de Estabilidad Sectorial ({frecuencia})")
    df_ranking = pd.DataFrame(datos_actuales["ranking"]).sort_values(by="Fuerza Relación", ascending=False).set_index("Par")
    st.dataframe(df_ranking)
    
    st.subheader("🎯 Analizador de Execution y Gestión de Riesgo")
    par_elegido = st.selectbox(
        "Selecciona un par cointegrado:", 
        list(datos_actuales["spreads"].keys()),
        key=f"selector_pares_{st.session_state['contador_ejecucion']}"
    )
    
    if par_elegido and par_elegido in datos_actuales["detalles"]:
        info = datos_actuales["detalles"][par_elegido]
        z_grafico = datos_actuales["spreads"][par_elegido]
        
        capital_usd = (capital_total / 2.0) * 1.10
        acciones_t1 = round(capital_usd / info["p1"])
        acciones_t2 = round(capital_usd / info["p2"])
        ganancia_euros = capital_total * (objetivo_rendimiento / 100.0)
        
        st.markdown(f"### 📋 Ficha Operativa Real: {par_elegido}")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Z-Score Actual", f"{info['z']:.2f}")
        with col2: st.metric(f"🎯 Ganancia Target ({objetivo_rendimiento}%)", f"+{ganancia_euros:.2f} €")
        with col3: st.metric("⚠️ Límite Stop Loss", "Z = ±3.50")
            
        token_bot = "8948061031:AAF-hZXlXcoolKy9QZAwj2_gLTMr_GOWjZU"
        chat_id_usuario = "399072608"
        
        if info["z"] > 2.0:
            plan_texto = f"Entrada {par_elegido}. Corto {acciones_t1} de {info['t1']} y Largo {acciones_t2} de {info['t2']}. Cierre en Z=0 (+{ganancia_euros}€). Stop Loss en Z=3.50."
            st.error(f"🔴 ESTRATEGIA: VENTA DEL SPREAD ACTIVADA")
            url_tel = f"https://telegram.org{token_bot}/sendMessage?chat_id={chat_id_usuario}&text={plan_texto}"
            st.link_button("📲 ENVIAR ORDEN AL MÓVIL", url_tel)
        elif info["z"] < -2.0:
            plan_texto = f"Entrada {par_elegido}. Largo {acciones_t1} de {info['t1']} y Corto {acciones_t2} de {info['t2']}. Cierre en Z=0 (+{ganancia_euros}€). Stop Loss en Z=-3.50."
            st.success(f"🟢 ESTRATEGIA: COMPRA DEL SPREAD ACTIVADA")
            url_tel = f"https://telegram.org{token_bot}/sendMessage?chat_id={chat_id_usuario}&text={plan_texto}"
            st.link_button("📲 ENVIAR ORDEN AL MÓVIL", url_tel)
        else:
            plan_texto = f"El par {par_elegido} está equilibrado. No operar."
            st.info(f"⚖️ POSICIÓN: ESPERANDO MOVIMIENTO")
            st.write(f"Para ganar {ganancia_euros}€ en Z=0, el plan asignará {acciones_t1} acciones de {info['t1']} y {acciones_t2} de {info['t2']}.")
            url_tel = f"https://telegram.org{token_bot}/sendMessage?chat_id={chat_id_usuario}&text=Reporte: {par_elegido} en equilibrio (Z={info['z']:.2f})"
            st.link_button("📲 ENVIAR REPORTE AL MÓVIL", url_tel)
            
        st.subheader(f"📈 Gráfico de Control Histórico ({frecuencia})")
        df_grafico = pd.DataFrame({"Z-Score del Spread": z_grafico}, index=info["precios_index"])
        st.line_chart(df_grafico)

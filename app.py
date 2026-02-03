"""
Domin Cideam.es - Generador de Planes de Entrenamiento
Planes de entrenamiento personalizados basados en análisis de datos Garmin.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import json
import struct
import zipfile

# Import local modules
from training_zones import TrainingZones
from plan_generator import TrainingPlanGenerator
from garmin_analyzer import GarminDataAnalyzer, PowerProfileAnalyzer
from garmin_fit_exporter import GarminFitExporter
from calendar_view import TrainingCalendarView
from pdf_exporter import TrainingPlanPDFExporter

# Page configuration
st.set_page_config(
    page_title="Domin Cideam.es - Plan de Entrenamiento",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .zone-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .workout-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🏃 Domin Cideam.es - Generador de Planes de Entrenamiento</h1>', unsafe_allow_html=True)

    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Perfil del Usuario")

        # Required inputs
        st.subheader("Información Requerida")

        age = st.number_input("Edad (años)", min_value=10, max_value=100, value=35, step=1)
        gender = st.selectbox("Género", options=["Masculino", "Femenino"], index=0)
        gender = "male" if gender == "Masculino" else "female"
        weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        vo2_max = st.number_input("VO2 Máx (ml/kg/min)", min_value=20.0, max_value=90.0, value=45.0, step=0.5)
        max_hr = st.number_input("Frecuencia Cardíaca Máxima (ppm)", min_value=120, max_value=220, value=185, step=1)
        resting_hr = st.number_input("FC en Reposo (ppm)", min_value=30, max_value=100, value=60, step=1)

        st.subheader("Preferencia de Entrenamiento")
        training_intensity = st.selectbox(
            "Estilo de Intensidad",
            options=[
                "Suave/Zona 2 (Baja intensidad, base aeróbica)",
                "Moderado/Tempo (Entrenamiento mixto con series tempo)",
                "Alta Intensidad (Intervalos y trabajo de velocidad)"
            ],
            index=2
        )

        # Map selection to code
        intensity_map = {
            "Suave/Zona 2 (Baja intensidad, base aeróbica)": "easy",
            "Moderado/Tempo (Entrenamiento mixto con series tempo)": "moderate",
            "Alta Intensidad (Intervalos y trabajo de velocidad)": "high"
        }
        selected_intensity = intensity_map[training_intensity]

        st.subheader("Tipos de Entrenamiento Preferidos")
        st.caption("Selecciona los tipos de entrenamiento que más te gustan")

        pref_easy_runs = st.checkbox("Rodajes Suaves", value=True)
        pref_tempo_runs = st.checkbox("Series Tempo", value=True)
        pref_intervals = st.checkbox("Intervalos", value=True)
        pref_long_runs = st.checkbox("Tiradas Largas", value=True)
        pref_hill_runs = st.checkbox("Cuestas", value=False)
        pref_fartlek = st.checkbox("Fartlek", value=False)

        preferred_workouts = {
            "easy": pref_easy_runs,
            "tempo": pref_tempo_runs,
            "intervals": pref_intervals,
            "long": pref_long_runs,
            "hills": pref_hill_runs,
            "fartlek": pref_fartlek
        }

        # Optional inputs
        st.subheader("Opcional: Marcas Personales (Últimos 4-5 meses)")
        st.caption("⚠️ Introduce solo tiempos de los últimos 4-5 meses para mayor precisión")

        col1, col2 = st.columns(2)
        with col1:
            best_5k_min = st.number_input("Tiempo 5K (min)", min_value=0, max_value=60, value=0, key="5k_min")
        with col2:
            best_5k_sec = st.number_input("Tiempo 5K (seg)", min_value=0, max_value=59, value=0, key="5k_sec")

        col3, col4 = st.columns(2)
        with col3:
            best_10k_min = st.number_input("Tiempo 10K (min)", min_value=0, max_value=120, value=0, key="10k_min")
        with col4:
            best_10k_sec = st.number_input("Tiempo 10K (seg)", min_value=0, max_value=59, value=0, key="10k_sec")

        best_5k_time = f"{best_5k_min}:{best_5k_sec:02d}" if best_5k_min > 0 else None
        best_10k_time = f"{best_10k_min}:{best_10k_sec:02d}" if best_10k_min > 0 else None

        # Number of training days per week
        st.subheader("Calendario de Entrenamiento")
        training_days_per_week = st.slider("Días de Entrenamiento por Semana", min_value=3, max_value=7, value=4)

    # Main content area
    tab1, tab2, tab2b, tab2c, tab2d, tab3, tab4, tab4b, tab5 = st.tabs([
        "📁 Subir Datos",
        "📈 Análisis de Forma",
        "🏆 Fitness Score",
        "⚡ Power Profile",
        "🎯 Dashboard Avanzado",
        "🏃 Zonas de Entrenamiento",
        "📅 Plan de Entrenamiento",
        "📆 Calendario Interactivo",
        "📤 Exportar"
    ])

    # Initialize session state
    if 'garmin_data' not in st.session_state:
        st.session_state.garmin_data = None
    if 'fitness_status' not in st.session_state:
        st.session_state.fitness_status = None
    if 'fitness_score' not in st.session_state:
        st.session_state.fitness_score = None
    if 'training_plan' not in st.session_state:
        st.session_state.training_plan = None
    if 'advanced_metrics' not in st.session_state:
        st.session_state.advanced_metrics = None

    # Tab 1: Upload Garmin Data
    with tab1:
        st.header("📁 Subir Historial de Entrenamiento Garmin")
        st.markdown("""
        Sube tu archivo CSV de historial de entrenamiento de Garmin para analizar tu nivel de forma actual.

        **Columnas esperadas:**
        - `date` / `fecha`: Fecha de la actividad
        - `distance` / `distancia`: Distancia en kilómetros
        - `duration` / `duracion`: Duración (HH:MM:SS o minutos)
        - `average_heart_rate` / `fc_media`: Frecuencia cardíaca media (ppm)
        - `calories` / `calorias`: Calorías quemadas
        - `activity_type` / `tipo`: Tipo de actividad (Carrera, Caminata, etc.)
        """)

        uploaded_file = st.file_uploader(
            "Elige tu archivo CSV de Garmin",
            type=['csv'],
            help="Exporta tus actividades desde Garmin Connect"
        )

        if uploaded_file is not None:
            try:
                analyzer = GarminDataAnalyzer()
                df = analyzer.load_csv(uploaded_file)
                st.session_state.garmin_data = df

                st.success(f"✅ ¡Se cargaron correctamente {len(df)} actividades!")

                # Show preview
                st.subheader("Vista Previa de Datos")
                st.dataframe(df.head(10), use_container_width=True)

                # Analyze fitness
                st.session_state.fitness_status = analyzer.analyze_fitness(
                    df, max_hr=max_hr, resting_hr=resting_hr, gender=gender
                )

                # Calculate fitness score with TRIMP
                st.session_state.fitness_score = analyzer.calculate_fitness_score(
                    df, max_hr=max_hr, resting_hr=resting_hr, age=age, gender=gender
                )

                # Calculate advanced metrics
                st.session_state.advanced_metrics = analyzer.calculate_advanced_metrics(
                    df, max_hr=max_hr, resting_hr=resting_hr, age=age, gender=gender
                )

            except ValueError as e:
                st.error(f"❌ Error al cargar el archivo: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error inesperado al procesar el archivo: {str(e)}")
                st.info("💡 Asegúrate de que el archivo sea un CSV exportado de Garmin Connect")
        else:
            st.info("👆 Sube un archivo CSV exportado de Garmin Connect para comenzar el análisis")

    # Tab 2: Fitness Analysis
    with tab2:
        st.header("📈 Análisis de Forma Física General")

        if st.session_state.garmin_data is not None and st.session_state.fitness_status is not None:
            fitness = st.session_state.fitness_status
            df = st.session_state.garmin_data

            # Key metrics - Row 1
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Nivel de Forma", fitness['fitness_level'])
            with col2:
                hours = fitness['avg_weekly_duration'] / 60
                st.metric("Tiempo Semanal", f"{hours:.1f} h")
            with col3:
                st.metric("Actividades Totales", f"{fitness['total_activities']}")
            with col4:
                if fitness.get('avg_heart_rate', 0) > 0:
                    st.metric("FC Media", f"{fitness['avg_heart_rate']:.0f} ppm")
                else:
                    st.metric("Calorías Semanales", f"{fitness.get('avg_weekly_calories', 0):.0f}")

            # Key metrics - Row 2
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Distancia Semanal", f"{fitness['avg_weekly_distance']:.1f} km")
            with col6:
                if fitness['avg_pace'] > 0:
                    st.metric("Ritmo Medio (Carrera)", f"{fitness['avg_pace']:.2f} min/km")
                else:
                    st.metric("Ritmo Medio", "N/A")
            with col7:
                st.metric("Carga de Entrenamiento", f"{fitness['training_load']:.0f} UA/sem")
            with col8:
                st.metric("Carrera más Larga", f"{fitness.get('longest_run', 0):.1f} km")

            # Activity breakdown
            if fitness.get('activity_breakdown'):
                st.subheader("📊 Desglose por Tipo de Actividad")
                activity_data = fitness['activity_breakdown']
                if activity_data:
                    fig_activities = px.pie(
                        values=list(activity_data.values()),
                        names=list(activity_data.keys()),
                        title='Distribución de Actividades (Últimas 8 semanas)'
                    )
                    fig_activities.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_activities, use_container_width=True)

            # Charts
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.subheader("Tendencia de Duración Semanal")
                if 'duration_minutes' in df.columns:
                    weekly_data = df.groupby(pd.Grouper(key='date', freq='W'))['duration_minutes'].sum().reset_index()
                    weekly_data['hours'] = weekly_data['duration_minutes'] / 60
                    fig_duration = px.bar(weekly_data, x='date', y='hours',
                                           title='Horas de Entrenamiento por Semana',
                                           labels={'hours': 'Horas', 'date': 'Semana'})
                    fig_duration.update_traces(marker_color='#1E88E5')
                    st.plotly_chart(fig_duration, use_container_width=True)

            with col_chart2:
                st.subheader("Distribución de Frecuencia Cardíaca")
                if 'average_heart_rate' in df.columns:
                    valid_hr = df[df['average_heart_rate'].notna()]
                    if len(valid_hr) > 0:
                        fig_hr = px.histogram(valid_hr, x='average_heart_rate', nbins=30,
                                             title='Distribución de Frecuencia Cardíaca',
                                             labels={'average_heart_rate': 'FC Media (ppm)'})
                        fig_hr.update_traces(marker_color='#E91E63')
                        st.plotly_chart(fig_hr, use_container_width=True)

            # Training load analysis
            st.subheader("📈 Análisis de Carga de Entrenamiento")
            col_load1, col_load2, col_load3 = st.columns(3)

            with col_load1:
                st.metric("Carga Semanal Actual", f"{fitness['training_load']:.0f} UA",
                         help="Unidades Arbitrarias basadas en duración e intensidad")
            with col_load2:
                st.metric("Incremento Recomendado", f"{fitness['recommended_increase']:.0f}%",
                         help="Tasa de progresión segura según tu nivel actual")
            with col_load3:
                st.metric("Semanas Analizadas", f"{fitness['weeks_analyzed']}")
        else:
            st.warning("⚠️ Por favor sube los datos de Garmin en la pestaña 'Subir Datos' primero")

    # Tab 2b: Fitness Score
    with tab2b:
        st.header("🏆 Fitness Score - Análisis Avanzado")

        if st.session_state.fitness_score is not None:
            fs = st.session_state.fitness_score

            # Score principal con gauge
            col_score1, col_score2, col_score3 = st.columns([2, 2, 2])

            with col_score1:
                # Gauge de Fitness Score
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=fs['fitness_score'],
                    title={'text': "Fitness Score", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#1E88E5"},
                        'steps': [
                            {'range': [0, 25], 'color': "#ffebee"},
                            {'range': [25, 50], 'color': "#fff3e0"},
                            {'range': [50, 75], 'color': "#e8f5e9"},
                            {'range': [75, 100], 'color': "#e3f2fd"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': fs['fitness_score']
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_score2:
                st.subheader("📊 Tu Posición")
                st.metric("Percentil Poblacional", f"{fs['percentile']}%")
                st.info(f"**{fs['percentile_label']}** - Estás por encima del {fs['percentile']}% de la población de tu edad y género.")

                # Gráfico de percentil
                fig_percentile = go.Figure()
                fig_percentile.add_trace(go.Bar(
                    x=['Tu posición'],
                    y=[fs['percentile']],
                    marker_color='#1E88E5',
                    text=[f"{fs['percentile']}%"],
                    textposition='outside'
                ))
                fig_percentile.update_layout(
                    yaxis_range=[0, 100],
                    yaxis_title="Percentil",
                    height=200,
                    showlegend=False
                )
                st.plotly_chart(fig_percentile, use_container_width=True)

            with col_score3:
                st.subheader("⚡ Estado de Forma")
                st.metric("CTL (Forma)", f"{fs['ctl']:.1f}", help="Chronic Training Load - Tu forma física acumulada")
                st.metric("ATL (Fatiga)", f"{fs['atl']:.1f}", help="Acute Training Load - Tu fatiga reciente")
                st.metric("TSB (Balance)", f"{fs['tsb']:.1f}", help="Training Stress Balance - Diferencia entre forma y fatiga")

                # Estado de forma con color
                tsb_color = "green" if fs['tsb'] > 0 else "orange" if fs['tsb'] > -15 else "red"
                st.markdown(f"<div style='padding:10px; background-color:{tsb_color}20; border-radius:5px; border-left:4px solid {tsb_color};'>{fs['form_status']}</div>", unsafe_allow_html=True)

            # Gráfico de evolución temporal
            st.subheader("📈 Evolución del Fitness Score")

            if fs.get('evolution_data') and len(fs['evolution_data']) > 0:
                evolution_df = pd.DataFrame(fs['evolution_data'])
                evolution_df['date'] = pd.to_datetime(evolution_df['date'])

                fig_evolution = go.Figure()

                # Fitness Score
                fig_evolution.add_trace(go.Scatter(
                    x=evolution_df['date'],
                    y=evolution_df['fitness_score'],
                    mode='lines',
                    name='Fitness Score',
                    line=dict(color='#1E88E5', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(30, 136, 229, 0.1)'
                ))

                fig_evolution.update_layout(
                    title='Evolución de tu Fitness Score',
                    xaxis_title='Fecha',
                    yaxis_title='Fitness Score',
                    yaxis_range=[0, 100],
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_evolution, use_container_width=True)

                # Gráfico CTL/ATL/TSB
                st.subheader("📊 Métricas de Carga de Entrenamiento")

                fig_pmc = go.Figure()

                fig_pmc.add_trace(go.Scatter(
                    x=evolution_df['date'],
                    y=evolution_df['ctl'],
                    mode='lines',
                    name='CTL (Forma)',
                    line=dict(color='#4CAF50', width=2)
                ))

                fig_pmc.add_trace(go.Scatter(
                    x=evolution_df['date'],
                    y=evolution_df['atl'],
                    mode='lines',
                    name='ATL (Fatiga)',
                    line=dict(color='#FF5722', width=2)
                ))

                fig_pmc.add_trace(go.Scatter(
                    x=evolution_df['date'],
                    y=evolution_df['tsb'],
                    mode='lines',
                    name='TSB (Balance)',
                    line=dict(color='#9C27B0', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(156, 39, 176, 0.1)'
                ))

                fig_pmc.update_layout(
                    title='Performance Management Chart (PMC)',
                    xaxis_title='Fecha',
                    yaxis_title='Valor',
                    height=350,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_pmc, use_container_width=True)

            # Estadísticas adicionales
            st.subheader("📋 Resumen de Entrenamiento")
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

            with col_stats1:
                st.metric("TRIMP Total", f"{fs['total_trimp']:.0f}")
            with col_stats2:
                st.metric("TRIMP Semanal Medio", f"{fs['avg_weekly_trimp']:.0f}")
            with col_stats3:
                progress_delta = f"+{fs['progress_pct']:.1f}%" if fs['progress_pct'] > 0 else f"{fs['progress_pct']:.1f}%"
                st.metric("Progreso", progress_delta)
            with col_stats4:
                st.metric("Edad/Género", f"{fs['age']} años / {'♂' if fs['gender'] == 'male' else '♀'}")

            # Explicación del sistema
            with st.expander("ℹ️ ¿Cómo se calcula el Fitness Score?"):
                st.markdown("""
                ### Metodología de Cálculo

                El **Fitness Score** se basa en el sistema TRIMP (Training Impulse) desarrollado por Banister:

                - **TRIMP** = Duración × Fracción de FC Reserva × Factor de Intensidad
                - **CTL (Chronic Training Load)**: Media móvil exponencial de 42 días del TRIMP diario
                - **ATL (Acute Training Load)**: Media móvil exponencial de 7 días del TRIMP diario
                - **TSB (Training Stress Balance)**: CTL - ATL

                ### Interpretación del TSB

                | TSB | Estado | Recomendación |
                |-----|--------|---------------|
                | > 25 | Muy descansado | Riesgo de pérdida de forma |
                | 10 a 25 | Fresco | Óptimo para competir |
                | -10 a 10 | Forma óptima | Buen balance entrenamiento/recuperación |
                | -25 a -10 | Fatigado | Considerar reducir carga |
                | < -25 | Muy fatigado | Riesgo de sobreentrenamiento |

                ### Percentiles Poblacionales

                La comparación se realiza contra datos poblacionales ajustados por edad y género,
                basados en estudios epidemiológicos de actividad física.
                """)
        else:
            st.warning("⚠️ Por favor sube los datos de Garmin en la pestaña 'Subir Datos' primero para ver tu Fitness Score")

    # Tab 2c: Power Profile (Watts/kg)
    with tab2c:
        st.header("⚡ Power Profile - Análisis de Potencia")
        st.markdown("""
        Evalúa tu nivel de ciclismo basado en tu potencia (watts) y potencia relativa (watts/kg).
        Basado en las tablas de referencia de **Coggan/Allen** utilizadas mundialmente.
        """)

        # Crear analizador de potencia
        power_analyzer = PowerProfileAnalyzer(weight=weight, gender=gender, age=age)

        # Intentar extraer datos de potencia del CSV cargado
        power_from_csv = None
        if st.session_state.garmin_data is not None:
            power_from_csv = power_analyzer.extract_power_from_dataframe(st.session_state.garmin_data)

        # Mostrar datos detectados del CSV
        if power_from_csv and power_from_csv['has_power_data']:
            st.success(f"✅ **Datos de potencia detectados en tu historial de Garmin**")

            col_csv1, col_csv2, col_csv3 = st.columns(3)
            with col_csv1:
                st.metric("Actividades con potencia", power_from_csv['total_activities_with_power'])
            with col_csv2:
                st.metric("FTP Estimado", f"{power_from_csv['estimated_ftp']:.0f} W")
            with col_csv3:
                st.metric("Potencia Máxima", f"{power_from_csv['max_power']:.0f} W")

            # Usar valores del CSV como defaults
            default_ftp = int(power_from_csv['estimated_ftp']) if power_from_csv['estimated_ftp'] > 0 else 200
            default_5s = int(power_from_csv['best_efforts']['5s']) if power_from_csv['best_efforts']['5s'] > 0 else 0
            default_1min = int(power_from_csv['best_efforts']['1min']) if power_from_csv['best_efforts']['1min'] > 0 else 0
            default_5min = int(power_from_csv['best_efforts']['5min']) if power_from_csv['best_efforts']['5min'] > 0 else 0

            st.info("💡 Los valores se han pre-rellenado con los datos de tu historial. Puedes ajustarlos si conoces tus valores exactos.")
        else:
            default_ftp = 200
            default_5s = 0
            default_1min = 0
            default_5min = 0
            if st.session_state.garmin_data is not None:
                st.warning("⚠️ No se encontraron datos de potencia en tu historial de Garmin. Introduce los valores manualmente.")
            else:
                st.info("💡 Sube tu historial de Garmin en la pestaña 'Subir Datos' para detectar automáticamente los datos de potencia, o introduce los valores manualmente.")

        st.divider()

        # Inputs de potencia
        col_power1, col_power2 = st.columns(2)

        with col_power1:
            st.subheader("📊 Datos de Potencia")
            ftp_input = st.number_input(
                "FTP (Functional Threshold Power) en watts",
                min_value=0, max_value=600, value=default_ftp,
                help="Potencia que puedes mantener durante ~1 hora. Si no lo conoces, usa el 95% de tu potencia media en un test de 20 minutos."
            )

            power_5s = st.number_input(
                "Potencia máxima 5 segundos (Sprint)",
                min_value=0, max_value=2500, value=default_5s,
                help="Tu potencia máxima en un sprint de 5 segundos"
            )

            power_1min = st.number_input(
                "Potencia máxima 1 minuto",
                min_value=0, max_value=1000, value=default_1min,
                help="Tu potencia máxima sostenida durante 1 minuto"
            )

            power_5min = st.number_input(
                "Potencia máxima 5 minutos",
                min_value=0, max_value=600, value=default_5min,
                help="Tu potencia máxima sostenida durante 5 minutos (esfuerzo VO2max)"
            )

        with col_power2:
            st.subheader("⚙️ Configuración")
            st.info(f"**Peso actual:** {weight} kg (configurado en la barra lateral)")
            st.info(f"**Género:** {'Masculino' if gender == 'male' else 'Femenino'}")
            st.info(f"**Edad:** {age} años")

            # Mostrar historial de potencia si existe
            if power_from_csv and power_from_csv['power_history']:
                with st.expander("📈 Ver historial de potencia"):
                    history_df = pd.DataFrame(power_from_csv['power_history'])
                    if not history_df.empty:
                        # Mostrar últimas 10 actividades
                        st.dataframe(history_df.tail(10)[['date', 'avg_power', 'max_power', 'duration_minutes']],
                                    use_container_width=True)

        # Analizar potencia
        if ftp_input > 0:
            # Análisis FTP
            ftp_analysis = power_analyzer.analyze_ftp(ftp_input)

            # Análisis de perfil completo si hay más datos
            if power_5s > 0 or power_1min > 0 or power_5min > 0:
                full_profile = power_analyzer.analyze_power_profile(
                    power_5s=power_5s,
                    power_1min=power_1min,
                    power_5min=power_5min,
                    ftp=ftp_input
                )
            else:
                full_profile = None

            # Mostrar resultados principales
            st.divider()
            st.subheader("🏆 Tu Nivel de Potencia")

            col_res1, col_res2, col_res3 = st.columns(3)

            with col_res1:
                # Gauge de watts/kg
                fig_wpkg = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ftp_analysis['watts_per_kg'],
                    number={'suffix': " w/kg", 'font': {'size': 40}},
                    title={'text': "Potencia Relativa (FTP)", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 7], 'tickwidth': 1},
                        'bar': {'color': "#FF6B00"},
                        'steps': [
                            {'range': [0, 2.5], 'color': "#ffebee"},
                            {'range': [2.5, 3.5], 'color': "#fff3e0"},
                            {'range': [3.5, 4.5], 'color': "#e8f5e9"},
                            {'range': [4.5, 5.5], 'color': "#e3f2fd"},
                            {'range': [5.5, 7], 'color': "#f3e5f5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': ftp_analysis['watts_per_kg']
                        }
                    }
                ))
                fig_wpkg.update_layout(height=280)
                st.plotly_chart(fig_wpkg, use_container_width=True)

            with col_res2:
                st.metric("FTP Absoluto", f"{ftp_analysis['ftp_watts']:.0f} W")
                st.metric("Categoría", ftp_analysis['category'])
                st.info(f"**{ftp_analysis['category_description']}**")

                if ftp_analysis['watts_to_next'] > 0:
                    st.caption(f"📈 Necesitas **+{ftp_analysis['watts_to_next']:.0f}W** para alcanzar '{ftp_analysis['next_category']}'")

            with col_res3:
                st.metric("Percentil", f"{ftp_analysis['percentile']}%")
                st.metric("Percentil Ajustado por Edad", f"{ftp_analysis['age_adjusted_percentile']}%")
                st.metric("VO2max Estimado", f"{ftp_analysis['estimated_vo2max']:.1f} ml/kg/min")

            # Zonas de potencia
            st.divider()
            st.subheader("🎯 Zonas de Potencia (Modelo Coggan)")

            zones_data = ftp_analysis['power_zones']
            zone_colors_power = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF9800', '#FF5722', '#F44336']

            # Crear gráfico de barras para zonas
            fig_zones = go.Figure()

            for i, (zone_name, zone_data) in enumerate(zones_data.items()):
                max_val = min(zone_data['max'], ftp_input * 2)  # Limitar para visualización
                fig_zones.add_trace(go.Bar(
                    name=zone_name,
                    x=[zone_name.split(' - ')[0]],
                    y=[max_val - zone_data['min']],
                    base=zone_data['min'],
                    marker_color=zone_colors_power[i],
                    text=f"{zone_data['min']}-{max_val}W",
                    textposition='inside',
                    hovertemplate=f"<b>{zone_name}</b><br>{zone_data['min']}-{max_val}W<br>{zone_data['description']}<extra></extra>"
                ))

            fig_zones.update_layout(
                title="Zonas de Potencia basadas en tu FTP",
                yaxis_title="Potencia (watts)",
                height=350,
                showlegend=False,
                barmode='stack'
            )
            st.plotly_chart(fig_zones, use_container_width=True)

            # Tabla de zonas
            with st.expander("📋 Ver tabla detallada de zonas"):
                zones_table = []
                for zone_name, zone_data in zones_data.items():
                    max_display = zone_data['max'] if zone_data['max'] < 9999 else "Máx"
                    zones_table.append({
                        'Zona': zone_name,
                        'Rango (W)': f"{zone_data['min']} - {max_display}",
                        'Descripción': zone_data['description']
                    })
                st.table(pd.DataFrame(zones_table))

            # Perfil completo si hay datos adicionales
            if full_profile and len(full_profile.get('powers', {})) > 1:
                st.divider()
                st.subheader("📊 Perfil de Potencia Completo")

                col_profile1, col_profile2 = st.columns([2, 1])

                with col_profile1:
                    # Gráfico radar del perfil
                    powers = full_profile['powers']
                    categories = []
                    values = []

                    duration_labels = {'5s': 'Sprint (5s)', '1min': 'Anaeróbico (1min)',
                                      '5min': 'VO2max (5min)', 'ftp': 'FTP (60min)'}

                    for duration in ['5s', '1min', '5min', 'ftp']:
                        if duration in powers:
                            categories.append(duration_labels[duration])
                            values.append(powers[duration]['percentile'])

                    if len(categories) >= 3:
                        fig_radar = go.Figure()

                        fig_radar.add_trace(go.Scatterpolar(
                            r=values + [values[0]],  # Cerrar el polígono
                            theta=categories + [categories[0]],
                            fill='toself',
                            fillcolor='rgba(255, 107, 0, 0.3)',
                            line=dict(color='#FF6B00', width=2),
                            name='Tu perfil'
                        ))

                        # Añadir referencia de percentil 50
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[50] * (len(categories) + 1),
                            theta=categories + [categories[0]],
                            line=dict(color='gray', width=1, dash='dash'),
                            name='Promedio (P50)'
                        ))

                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100])
                            ),
                            showlegend=True,
                            title="Perfil de Potencia (Percentiles)",
                            height=400
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                with col_profile2:
                    st.metric("Tipo de Ciclista", full_profile['rider_type'])
                    st.metric("Score General", f"{full_profile['overall_score']:.0f}/100")

                    if full_profile['strengths']:
                        st.success(f"💪 **Fortalezas:** {', '.join(full_profile['strengths'])}")
                    if full_profile['weaknesses']:
                        st.warning(f"📈 **A mejorar:** {', '.join(full_profile['weaknesses'])}")

                    # Tabla de potencias
                    st.markdown("**Detalle por duración:**")
                    for duration, data in powers.items():
                        st.caption(f"• {duration_labels.get(duration, duration)}: {data['watts']:.0f}W ({data['watts_per_kg']:.2f} w/kg) - P{data['percentile']}")

            # Gráfico de evolución de potencia si hay historial
            if power_from_csv and power_from_csv['power_history'] and len(power_from_csv['power_history']) > 1:
                st.divider()
                st.subheader("📈 Evolución de Potencia")

                history_df = pd.DataFrame(power_from_csv['power_history'])
                history_df['date'] = pd.to_datetime(history_df['date'])

                fig_power_evolution = go.Figure()

                # Potencia promedio
                fig_power_evolution.add_trace(go.Scatter(
                    x=history_df['date'],
                    y=history_df['avg_power'],
                    mode='lines+markers',
                    name='Potencia Media',
                    line=dict(color='#FF6B00', width=2),
                    marker=dict(size=6)
                ))

                # Potencia máxima si existe
                if 'max_power' in history_df.columns and history_df['max_power'].sum() > 0:
                    fig_power_evolution.add_trace(go.Scatter(
                        x=history_df['date'],
                        y=history_df['max_power'],
                        mode='lines+markers',
                        name='Potencia Máxima',
                        line=dict(color='#E91E63', width=2),
                        marker=dict(size=6)
                    ))

                # Línea de FTP actual
                fig_power_evolution.add_hline(
                    y=ftp_input,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"FTP: {ftp_input}W"
                )

                fig_power_evolution.update_layout(
                    title='Evolución de Potencia en tus Actividades',
                    xaxis_title='Fecha',
                    yaxis_title='Potencia (watts)',
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_power_evolution, use_container_width=True)

            # Recomendaciones de entrenamiento
            st.divider()
            st.subheader("💡 Recomendaciones de Entrenamiento")

            recommendations = ftp_analysis['training_recommendations']
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

            # Tabla de referencia
            with st.expander("📚 Tabla de Referencia de Categorías (Coggan/Allen)"):
                st.markdown(f"**Tabla para {'Hombres' if gender == 'male' else 'Mujeres'}:**")

                ref_table = []
                categories = power_analyzer.power_categories
                for cat_name, cat_data in categories.items():
                    ref_table.append({
                        'Categoría': cat_name,
                        'W/kg mínimo': f"{cat_data['min_wpkg']:.2f}",
                        f'Watts ({weight}kg)': f"{cat_data['min_wpkg'] * weight:.0f}",
                        'Descripción': cat_data['description']
                    })
                st.table(pd.DataFrame(ref_table))

                st.markdown("""
                ### Notas sobre las categorías:
                - **FTP (Functional Threshold Power)**: Potencia máxima sostenible durante ~1 hora
                - Los valores son para FTP, no para potencias de corta duración
                - Las categorías están basadas en datos de ciclistas de todo el mundo
                - El percentil ajustado por edad considera el declive natural de potencia (~1%/año después de 35)
                """)
        else:
            st.info("👆 Introduce tu FTP (Functional Threshold Power) para comenzar el análisis")

            with st.expander("❓ ¿Cómo calcular mi FTP?"):
                st.markdown("""
                ### Métodos para estimar tu FTP:

                1. **Test de 20 minutos**:
                   - Calienta 10-15 minutos
                   - Pedalea 20 minutos al máximo esfuerzo sostenible
                   - Tu FTP ≈ 95% de la potencia media de esos 20 minutos

                2. **Test de rampa**:
                   - Comienza a baja potencia y aumenta cada minuto
                   - Continúa hasta el agotamiento
                   - Tu FTP ≈ 75% de la potencia del último minuto completado

                3. **Desde Zwift/TrainerRoad**: Estas plataformas calculan tu FTP automáticamente

                4. **Estimación desde carrera de 1 hora**: Tu potencia media en una contrarreloj de 1 hora es aproximadamente tu FTP
                """)

    # Tab 2d: Advanced Dashboard
    with tab2d:
        st.header("🎯 Dashboard Avanzado - Estado de Entrenamiento")

        if st.session_state.advanced_metrics is not None:
            am = st.session_state.advanced_metrics

            # Recomendación del día destacada
            st.markdown("### 📋 Recomendación del Día")
            recommendation = am.get('daily_recommendation', 'Sube datos para obtener recomendaciones')

            # Determinar color según contenido
            if '🛑' in recommendation or 'DESCANSO' in recommendation:
                rec_color = '#ffebee'
                border_color = '#f44336'
            elif '⚠️' in recommendation or 'SUAVE' in recommendation:
                rec_color = '#fff3e0'
                border_color = '#ff9800'
            else:
                rec_color = '#e8f5e9'
                border_color = '#4caf50'

            st.markdown(f"""
            <div style="padding: 20px; background-color: {rec_color}; border-left: 5px solid {border_color};
                        border-radius: 8px; margin-bottom: 20px; font-size: 1.1em;">
                {recommendation}
            </div>
            """, unsafe_allow_html=True)

            # Alertas activas
            alerts = am.get('alerts', [])
            if alerts:
                st.markdown("### ⚠️ Alertas Activas")
                for alert in alerts:
                    alert_type = alert.get('type', 'info')
                    if alert_type == 'danger':
                        st.error(f"{alert.get('icon', '')} **{alert.get('title', '')}**: {alert.get('message', '')}")
                    elif alert_type == 'warning':
                        st.warning(f"{alert.get('icon', '')} **{alert.get('title', '')}**: {alert.get('message', '')}")
                    else:
                        st.info(f"{alert.get('icon', '')} **{alert.get('title', '')}**: {alert.get('message', '')}")

            st.divider()

            # Métricas principales en 4 columnas
            st.markdown("### 📊 Métricas de Riesgo y Carga")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                acwr = am.get('acwr', 0)
                acwr_delta = "Óptimo" if 0.8 <= acwr <= 1.3 else "Fuera de rango"
                st.metric("ACWR", f"{acwr:.2f}", acwr_delta)
                st.caption(am.get('acwr_status', ''))

            with col2:
                ramp = am.get('ramp_rate', 0)
                ramp_delta = f"{ramp:+.1f} pts/sem"
                st.metric("Ramp Rate", f"{ramp:.1f}", ramp_delta)
                st.caption(am.get('ramp_status', ''))

            with col3:
                monotony = am.get('monotony', 0)
                st.metric("Monotonía", f"{monotony:.2f}")
                st.caption(am.get('monotony_status', ''))

            with col4:
                strain = am.get('strain', 0)
                strain_status = "Alto" if strain > 4000 else "Normal"
                st.metric("Strain", f"{strain:.0f}", strain_status)

            # Gráfico de ACWR con zonas
            st.markdown("### 📈 Zona de Riesgo ACWR")

            fig_acwr = go.Figure()

            # Zonas de fondo
            fig_acwr.add_shape(type="rect", x0=0, x1=1, y0=0, y1=0.8,
                              fillcolor="rgba(255, 193, 7, 0.3)", line_width=0)
            fig_acwr.add_shape(type="rect", x0=0, x1=1, y0=0.8, y1=1.3,
                              fillcolor="rgba(76, 175, 80, 0.3)", line_width=0)
            fig_acwr.add_shape(type="rect", x0=0, x1=1, y0=1.3, y1=1.5,
                              fillcolor="rgba(255, 152, 0, 0.3)", line_width=0)
            fig_acwr.add_shape(type="rect", x0=0, x1=1, y0=1.5, y1=2.0,
                              fillcolor="rgba(244, 67, 54, 0.3)", line_width=0)

            # Indicador actual
            fig_acwr.add_trace(go.Indicator(
                mode="gauge+number",
                value=acwr,
                gauge={
                    'axis': {'range': [0, 2], 'tickwidth': 1},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 0.8], 'color': "rgba(255, 193, 7, 0.5)"},
                        {'range': [0.8, 1.3], 'color': "rgba(76, 175, 80, 0.5)"},
                        {'range': [1.3, 1.5], 'color': "rgba(255, 152, 0, 0.5)"},
                        {'range': [1.5, 2], 'color': "rgba(244, 67, 54, 0.5)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': acwr
                    }
                },
                title={'text': "ACWR (Acute:Chronic Workload Ratio)"}
            ))

            fig_acwr.update_layout(height=250)
            st.plotly_chart(fig_acwr, use_container_width=True)

            st.caption("🟡 < 0.8: Desentrenamiento | 🟢 0.8-1.3: Óptimo | 🟠 1.3-1.5: Precaución | 🔴 > 1.5: Alto riesgo")

            # Gráficos de Evolución Temporal
            metrics_evo = am.get('metrics_evolution', {})

            if metrics_evo and metrics_evo.get('acwr'):
                st.markdown("### 📈 Evolución ACWR y Carga")

                acwr_data = metrics_evo.get('acwr', [])
                if acwr_data:
                    evo_df = pd.DataFrame(acwr_data)
                    evo_df['date'] = pd.to_datetime(evo_df['date'])

                    # Gráfico de ACWR con zonas de riesgo
                    fig_acwr_evo = go.Figure()

                    # Zonas de fondo
                    fig_acwr_evo.add_hrect(y0=0, y1=0.8, fillcolor="rgba(255, 193, 7, 0.2)",
                                           line_width=0, annotation_text="Desentrenamiento",
                                           annotation_position="top left")
                    fig_acwr_evo.add_hrect(y0=0.8, y1=1.3, fillcolor="rgba(76, 175, 80, 0.2)",
                                           line_width=0, annotation_text="Óptimo")
                    fig_acwr_evo.add_hrect(y0=1.3, y1=1.5, fillcolor="rgba(255, 152, 0, 0.2)",
                                           line_width=0, annotation_text="Precaución")
                    fig_acwr_evo.add_hrect(y0=1.5, y1=2.0, fillcolor="rgba(244, 67, 54, 0.2)",
                                           line_width=0, annotation_text="Alto riesgo")

                    # Línea de ACWR
                    fig_acwr_evo.add_trace(go.Scatter(
                        x=evo_df['date'], y=evo_df['acwr'],
                        mode='lines+markers', name='ACWR',
                        line=dict(color='#1E88E5', width=2),
                        marker=dict(size=4)
                    ))

                    fig_acwr_evo.update_layout(
                        title='Evolución del ACWR',
                        xaxis_title='Fecha', yaxis_title='ACWR',
                        height=350, yaxis=dict(range=[0, 2]),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_acwr_evo, use_container_width=True)

                    # Gráfico CTL/ATL/TSB
                    fig_ctl = go.Figure()
                    fig_ctl.add_trace(go.Scatter(
                        x=evo_df['date'], y=evo_df['ctl'],
                        mode='lines', name='CTL (Fitness)',
                        line=dict(color='#4CAF50', width=2)
                    ))
                    fig_ctl.add_trace(go.Scatter(
                        x=evo_df['date'], y=evo_df['atl'],
                        mode='lines', name='ATL (Fatiga)',
                        line=dict(color='#F44336', width=2)
                    ))
                    fig_ctl.add_trace(go.Scatter(
                        x=evo_df['date'], y=evo_df['tsb'],
                        mode='lines', name='TSB (Forma)',
                        fill='tozeroy', line=dict(color='#2196F3', width=1),
                        fillcolor='rgba(33, 150, 243, 0.3)'
                    ))
                    fig_ctl.add_hline(y=0, line_dash="dash", line_color="gray")

                    fig_ctl.update_layout(
                        title='Performance Management Chart (CTL/ATL/TSB)',
                        xaxis_title='Fecha', yaxis_title='Puntos',
                        height=350, hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig_ctl, use_container_width=True)

            # Gráfico de Ramp Rate
            ramp_data = metrics_evo.get('ramp_rate', [])
            if ramp_data:
                st.markdown("### 📊 Evolución Ramp Rate")
                ramp_df = pd.DataFrame(ramp_data)
                ramp_df['date'] = pd.to_datetime(ramp_df['date'])

                # Colorear barras según zona
                colors = ['#4CAF50' if 0 <= r <= 7 else '#F44336' if r > 7 else '#FFC107'
                          for r in ramp_df['ramp_rate']]

                fig_ramp = go.Figure()
                fig_ramp.add_trace(go.Bar(
                    x=ramp_df['date'], y=ramp_df['ramp_rate'],
                    marker_color=colors, name='Ramp Rate'
                ))
                fig_ramp.add_hline(y=7, line_dash="dash", line_color="red",
                                   annotation_text="Límite seguro")
                fig_ramp.add_hline(y=0, line_dash="solid", line_color="gray")

                fig_ramp.update_layout(
                    title='Ramp Rate Semanal (cambio de CTL)',
                    xaxis_title='Fecha', yaxis_title='Puntos/semana',
                    height=300
                )
                st.plotly_chart(fig_ramp, use_container_width=True)

            st.divider()

            # Eficiencia y VO2max
            st.markdown("### 🏃 Rendimiento Aeróbico")
            col_ef1, col_ef2, col_ef3, col_ef4 = st.columns(4)

            with col_ef1:
                ef = am.get('efficiency_factor', 0)
                ef_trend = am.get('ef_trend', 'stable')
                trend_icon = "📈" if ef_trend == 'improving' else "📉" if ef_trend == 'declining' else "➡️"
                st.metric("Efficiency Factor", f"{ef:.3f}", f"{trend_icon} {am.get('ef_change_pct', 0):.1f}%")

            with col_ef2:
                decoupling = am.get('decoupling', 0)
                if decoupling < 3:
                    dec_status = "Excelente"
                    dec_delta_color = "normal"
                elif decoupling < 5:
                    dec_status = "Bueno"
                    dec_delta_color = "normal"
                elif decoupling < 8:
                    dec_status = "Moderado"
                    dec_delta_color = "off"
                else:
                    dec_status = "Mejorable"
                    dec_delta_color = "inverse"
                st.metric("Decoupling", f"{decoupling:.1f}%", dec_status, delta_color=dec_delta_color)

            with col_ef3:
                vo2max = am.get('vo2max_estimated', 0)
                st.metric("VO2max Estimado", f"{vo2max:.1f} ml/kg/min")
                st.caption(am.get('vo2max_category', ''))

            with col_ef4:
                vi = am.get('variability_index', 0)
                if vi > 0:
                    st.metric("Variability Index", f"{vi:.2f}")
                    st.caption("Ciclismo")
                else:
                    if_val = am.get('intensity_factor', 0)
                    st.metric("Intensity Factor", f"{if_val:.2f}")

            # Gráficos de evolución de rendimiento
            col_evo1, col_evo2 = st.columns(2)

            # Gráfico de Efficiency Factor
            ef_data = metrics_evo.get('efficiency_factor', [])
            if ef_data:
                with col_evo1:
                    ef_df = pd.DataFrame(ef_data)
                    ef_df['date'] = pd.to_datetime(ef_df['date'])

                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(
                        x=ef_df['date'], y=ef_df['ef'],
                        mode='lines+markers', name='EF',
                        line=dict(color='#9C27B0', width=2),
                        marker=dict(size=6),
                        hovertemplate='%{x}<br>EF: %{y:.3f}<br>Ritmo: %{customdata[0]:.2f} min/km<br>Distancia: %{customdata[1]:.1f} km',
                        customdata=ef_df[['pace', 'distance']].values
                    ))

                    # Línea de tendencia
                    if len(ef_df) >= 3:
                        z = np.polyfit(range(len(ef_df)), ef_df['ef'], 1)
                        p = np.poly1d(z)
                        fig_ef.add_trace(go.Scatter(
                            x=ef_df['date'], y=p(range(len(ef_df))),
                            mode='lines', name='Tendencia',
                            line=dict(color='rgba(156, 39, 176, 0.4)', width=2, dash='dash')
                        ))

                    fig_ef.update_layout(
                        title='Evolución Efficiency Factor (Running)',
                        xaxis_title='Fecha', yaxis_title='EF (velocidad/FC)',
                        height=350, showlegend=True
                    )
                    st.plotly_chart(fig_ef, use_container_width=True)

            # Gráfico de Monotony y Strain
            mono_data = metrics_evo.get('monotony_strain', [])
            if mono_data:
                with col_evo2:
                    mono_df = pd.DataFrame(mono_data)
                    mono_df['date'] = pd.to_datetime(mono_df['date'])

                    fig_mono = make_subplots(specs=[[{"secondary_y": True}]])

                    # Monotony
                    fig_mono.add_trace(go.Scatter(
                        x=mono_df['date'], y=mono_df['monotony'],
                        mode='lines+markers', name='Monotonía',
                        line=dict(color='#FF9800', width=2),
                        marker=dict(size=4)
                    ), secondary_y=False)

                    # Línea de referencia monotonía
                    fig_mono.add_hline(y=2.0, line_dash="dash", line_color="red",
                                       annotation_text="Límite monotonía", secondary_y=False)

                    # Strain como barras
                    fig_mono.add_trace(go.Bar(
                        x=mono_df['date'], y=mono_df['strain'],
                        name='Strain', marker_color='rgba(244, 67, 54, 0.5)',
                        opacity=0.6
                    ), secondary_y=True)

                    fig_mono.update_layout(
                        title='Monotonía y Strain Semanal',
                        height=350, hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    fig_mono.update_yaxes(title_text="Monotonía", secondary_y=False)
                    fig_mono.update_yaxes(title_text="Strain", secondary_y=True)
                    st.plotly_chart(fig_mono, use_container_width=True)

            # Gráfico de métricas de potencia (ciclismo)
            power_data = metrics_evo.get('power_metrics', [])
            if power_data:
                st.markdown("### 🚴 Evolución Métricas de Potencia")
                power_df = pd.DataFrame(power_data)
                power_df['date'] = pd.to_datetime(power_df['date'])

                col_pow1, col_pow2 = st.columns(2)

                with col_pow1:
                    fig_np = go.Figure()
                    fig_np.add_trace(go.Scatter(
                        x=power_df['date'], y=power_df['np'],
                        mode='lines+markers', name='NP (Normalizada)',
                        line=dict(color='#E91E63', width=2)
                    ))
                    fig_np.add_trace(go.Scatter(
                        x=power_df['date'], y=power_df['avg_power'],
                        mode='lines+markers', name='Potencia Media',
                        line=dict(color='#03A9F4', width=2)
                    ))
                    fig_np.update_layout(
                        title='Potencia por Actividad',
                        xaxis_title='Fecha', yaxis_title='Watts',
                        height=300, hovermode='x unified'
                    )
                    st.plotly_chart(fig_np, use_container_width=True)

                with col_pow2:
                    fig_tss = go.Figure()
                    fig_tss.add_trace(go.Bar(
                        x=power_df['date'], y=power_df['tss'],
                        name='TSS', marker_color='#673AB7'
                    ))
                    fig_tss.update_layout(
                        title='TSS por Actividad',
                        xaxis_title='Fecha', yaxis_title='TSS',
                        height=300
                    )
                    st.plotly_chart(fig_tss, use_container_width=True)

                # VI e IF
                fig_vi_if = go.Figure()
                fig_vi_if.add_trace(go.Scatter(
                    x=power_df['date'], y=power_df['vi'],
                    mode='lines+markers', name='Variability Index',
                    line=dict(color='#009688', width=2)
                ))
                fig_vi_if.add_trace(go.Scatter(
                    x=power_df['date'], y=power_df['if'],
                    mode='lines+markers', name='Intensity Factor',
                    line=dict(color='#FF5722', width=2)
                ))
                fig_vi_if.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                    annotation_text="IF=1.0 (umbral FTP)")
                fig_vi_if.update_layout(
                    title='Variability Index e Intensity Factor',
                    xaxis_title='Fecha', yaxis_title='Valor',
                    height=300, hovermode='x unified'
                )
                st.plotly_chart(fig_vi_if, use_container_width=True)

            # Gráfico de VO2max
            vo2_data = metrics_evo.get('vo2max', [])
            if vo2_data:
                st.markdown("### 🫁 Evolución VO2max Estimado")
                vo2_df = pd.DataFrame(vo2_data)

                fig_vo2 = go.Figure()
                fig_vo2.add_trace(go.Scatter(
                    x=vo2_df['date'], y=vo2_df['vo2max'],
                    mode='lines+markers', name='VO2max',
                    line=dict(color='#E91E63', width=3),
                    marker=dict(size=10),
                    hovertemplate='%{x}<br>VO2max: %{y:.1f} ml/kg/min<br>Mejor ritmo: %{customdata[0]:.2f} min/km',
                    customdata=vo2_df[['best_pace']].values
                ))

                # Zonas de referencia
                fig_vo2.add_hrect(y0=55, y1=70, fillcolor="rgba(76, 175, 80, 0.2)",
                                  annotation_text="Excelente", annotation_position="right")
                fig_vo2.add_hrect(y0=45, y1=55, fillcolor="rgba(139, 195, 74, 0.2)",
                                  annotation_text="Muy bueno", annotation_position="right")
                fig_vo2.add_hrect(y0=35, y1=45, fillcolor="rgba(255, 193, 7, 0.2)",
                                  annotation_text="Bueno", annotation_position="right")

                fig_vo2.update_layout(
                    title='Evolución VO2max Estimado (por mes)',
                    xaxis_title='Mes', yaxis_title='VO2max (ml/kg/min)',
                    height=350
                )
                st.plotly_chart(fig_vo2, use_container_width=True)

            st.divider()

            # Predicciones de carrera
            st.markdown("### 🏆 Predicciones de Carrera")
            predictions = am.get('race_predictions', {})

            if predictions and 'base_race' in predictions:
                base = predictions.get('base_race', {})
                st.info(f"📊 Basado en tu carrera de **{base.get('distance', 'N/A')}** en **{base.get('time', 'N/A')}** ({base.get('date', '')})")

                col_p1, col_p2, col_p3, col_p4 = st.columns(4)

                with col_p1:
                    p5k = predictions.get('5K', {})
                    st.metric("5K", p5k.get('time', 'N/A'))
                    st.caption(p5k.get('pace', ''))

                with col_p2:
                    p10k = predictions.get('10K', {})
                    st.metric("10K", p10k.get('time', 'N/A'))
                    st.caption(p10k.get('pace', ''))

                with col_p3:
                    phm = predictions.get('Media Maratón', {})
                    st.metric("Media Maratón", phm.get('time', 'N/A'))
                    st.caption(phm.get('pace', ''))

                with col_p4:
                    pm = predictions.get('Maratón', {})
                    st.metric("Maratón", pm.get('time', 'N/A'))
                    st.caption(pm.get('pace', ''))
            else:
                st.info("💡 Realiza una carrera de 5K o 10K para obtener predicciones de tiempos")

            st.divider()

            # Explicación detallada de métricas
            st.markdown("### 📚 Guía de Métricas")

            with st.expander("🔴 ACWR - Acute:Chronic Workload Ratio", expanded=False):
                st.markdown("""
                #### ¿Qué es?
                El **ACWR** (Ratio de Carga Aguda:Crónica) compara tu carga de entrenamiento reciente
                (últimos 7 días) con tu carga habitual (últimos 28 días). Es el indicador más importante
                para **prevenir lesiones**.

                #### ¿Cómo se calcula?
                ```
                ACWR = Carga Aguda (7 días) / Carga Crónica (28 días)
                ```

                #### Interpretación
                | Valor | Zona | Significado | Acción |
                |-------|------|-------------|--------|
                | < 0.8 | 🟡 Baja | Desentrenamiento, pérdida de forma | Aumentar carga gradualmente |
                | 0.8 - 1.0 | 🟢 Óptima baja | Mantenimiento, recuperación activa | Ideal para semanas de descarga |
                | 1.0 - 1.3 | 🟢 Óptima alta | Progresión segura, adaptación | Zona ideal para mejorar |
                | 1.3 - 1.5 | 🟠 Precaución | Riesgo moderado de lesión | Monitorizar síntomas |
                | > 1.5 | 🔴 Peligro | Alto riesgo de lesión (2-4x mayor) | Reducir carga inmediatamente |

                #### Evidencia científica
                Estudios de Gabbett (2016) y Hulin (2014) demuestran que mantener el ACWR entre
                0.8-1.3 reduce el riesgo de lesión hasta un 50% en atletas de resistencia.

                #### Consejos prácticos
                - **No aumentes** la carga semanal más del 10% respecto a la semana anterior
                - Después de una semana de descanso, **vuelve gradualmente** (no al 100%)
                - Si estás lesionado, el ACWR será alto al volver; planifica una vuelta progresiva
                """)

            with st.expander("📈 Ramp Rate - Tasa de Progresión", expanded=False):
                st.markdown("""
                #### ¿Qué es?
                El **Ramp Rate** mide cuántos puntos de CTL (Chronic Training Load) ganas o pierdes
                por semana. Indica si estás progresando demasiado rápido o demasiado lento.

                #### ¿Cómo se calcula?
                ```
                Ramp Rate = CTL actual - CTL hace 7 días
                ```

                #### Interpretación
                | Valor | Significado | Riesgo |
                |-------|-------------|--------|
                | < 0 | Pérdida de forma | Desentrenamiento si es prolongado |
                | 0 - 3 | Mantenimiento | Bajo riesgo, progresión lenta |
                | 3 - 5 | Progresión moderada | Óptimo para la mayoría |
                | 5 - 7 | Progresión agresiva | Aceptable para atletas experimentados |
                | > 7 | Progresión excesiva | Alto riesgo de sobreentrenamiento |

                #### Consejos prácticos
                - **Principiantes**: Mantén el Ramp Rate entre 3-5 puntos/semana
                - **Avanzados**: Puedes tolerar 5-7 puntos/semana en bloques cortos
                - **Recuperación**: Un Ramp Rate negativo es normal en semanas de descarga
                - **Precompetición**: Reduce a 0-2 puntos/semana las 2 semanas antes de una carrera
                """)

            with st.expander("🔄 Monotonía y Strain - Método de Foster", expanded=False):
                st.markdown("""
                #### ¿Qué es la Monotonía?
                La **Monotonía** mide cuán repetitivo es tu entrenamiento. Un entrenamiento muy
                similar día tras día aumenta el riesgo de lesiones por sobreuso.

                #### ¿Cómo se calcula?
                ```
                Monotonía = Media diaria de carga / Desviación estándar de carga
                ```

                #### Interpretación de Monotonía
                | Valor | Significado |
                |-------|-------------|
                | < 1.5 | ✅ Excelente variabilidad |
                | 1.5 - 2.0 | ✅ Buena variabilidad |
                | 2.0 - 2.5 | ⚠️ Poca variabilidad |
                | > 2.5 | 🔴 Entrenamiento muy monótono |

                #### ¿Qué es el Strain?
                El **Strain** (tensión) combina la carga total con la monotonía para estimar
                el estrés acumulado en tu cuerpo.

                ```
                Strain = Carga semanal total × Monotonía
                ```

                #### Interpretación de Strain
                | Valor | Significado |
                |-------|-------------|
                | < 2000 | Carga baja, bajo riesgo |
                | 2000 - 4000 | Carga moderada, riesgo normal |
                | 4000 - 6000 | Carga alta, monitorizar |
                | > 6000 | Carga muy alta, riesgo de enfermedad/lesión |

                #### Consejos prácticos
                - **Varía la intensidad**: Alterna días duros y suaves
                - **Varía la duración**: No hagas siempre la misma distancia
                - **Incluye descanso**: Al menos 1-2 días de descanso o muy suave por semana
                - **Semanas de descarga**: Cada 3-4 semanas, reduce la carga un 30-40%
                """)

            with st.expander("⚡ Efficiency Factor (EF) - Factor de Eficiencia", expanded=False):
                st.markdown("""
                #### ¿Qué es?
                El **Efficiency Factor** mide cuánta velocidad (o potencia) produces por cada
                latido de tu corazón. Es un indicador directo de tu **eficiencia aeróbica**.

                #### ¿Cómo se calcula?
                **Para running:**
                ```
                EF = Velocidad (metros/minuto) / Frecuencia Cardíaca Media
                ```

                **Para ciclismo:**
                ```
                EF = Potencia Normalizada / Frecuencia Cardíaca Media
                ```

                #### Interpretación
                | Tendencia | Significado |
                |-----------|-------------|
                | 📈 Subiendo | Tu base aeróbica está mejorando |
                | ➡️ Estable | Mantenimiento de forma |
                | 📉 Bajando | Fatiga acumulada o pérdida de forma |

                #### Valores típicos (running)
                | Nivel | EF aproximado |
                |-------|---------------|
                | Principiante | 0.6 - 0.8 |
                | Intermedio | 0.8 - 1.0 |
                | Avanzado | 1.0 - 1.2 |
                | Élite | > 1.2 |

                #### Consejos prácticos
                - **Compara contigo mismo**: El EF absoluto varía mucho entre personas
                - **Usa rutas similares**: El terreno afecta mucho al EF
                - **Condiciones similares**: Calor, viento y altitud afectan al EF
                - **Mejora con Z2**: El entrenamiento en Zona 2 es el más efectivo para mejorar EF
                """)

            with st.expander("💔 Decoupling - Desacoplamiento Cardíaco", expanded=False):
                st.markdown("""
                #### ¿Qué es?
                El **Decoupling** mide cuánto se "desacopla" tu frecuencia cardíaca de tu
                rendimiento durante un entrenamiento largo. Indica la **resistencia de tu
                sistema aeróbico**.

                #### ¿Cómo se calcula?
                ```
                Decoupling = ((EF primera mitad - EF segunda mitad) / EF primera mitad) × 100
                ```

                #### Interpretación
                | Valor | Significado | Base aeróbica |
                |-------|-------------|---------------|
                | < 3% | Excelente | Muy desarrollada |
                | 3% - 5% | Bueno | Bien desarrollada |
                | 5% - 8% | Aceptable | En desarrollo |
                | > 8% | Mejorable | Necesita más trabajo de base |

                #### ¿Por qué ocurre el decoupling?
                1. **Deshidratación**: Menos volumen sanguíneo → FC más alta
                2. **Agotamiento de glucógeno**: Cambio a metabolismo de grasas
                3. **Fatiga muscular**: Menos eficiencia mecánica
                4. **Termorregulación**: El cuerpo desvía sangre a la piel

                #### Consejos prácticos
                - **Test de decoupling**: Haz un rodaje largo (90+ min) a ritmo constante
                - **Objetivo**: Mantener decoupling < 5% en rodajes de 2+ horas
                - **Mejora con volumen**: Más kilómetros en Z2 reducen el decoupling
                - **Nutrición**: Hidratación y carbohidratos durante el ejercicio ayudan
                """)

            with st.expander("🫁 VO2max - Consumo Máximo de Oxígeno", expanded=False):
                st.markdown("""
                #### ¿Qué es?
                El **VO2max** es la cantidad máxima de oxígeno que tu cuerpo puede utilizar
                durante el ejercicio intenso. Es el indicador más importante de **capacidad
                aeróbica** y predictor de rendimiento en resistencia.

                #### ¿Cómo se estima?
                Usamos el método **VDOT de Jack Daniels**, basado en tus tiempos de carrera:
                ```
                VDOT ≈ 80 - (ritmo_min/km × 6.5) × factor_distancia
                ```

                #### Clasificación por edad y género
                **Hombres (ml/kg/min):**
                | Edad | Pobre | Regular | Bueno | Muy bueno | Excelente | Élite |
                |------|-------|---------|-------|-----------|-----------|-------|
                | 20-29 | <38 | 38-43 | 44-51 | 52-56 | 57-62 | >62 |
                | 30-39 | <35 | 35-40 | 41-48 | 49-54 | 55-60 | >60 |
                | 40-49 | <32 | 32-37 | 38-45 | 46-52 | 53-58 | >58 |
                | 50-59 | <29 | 29-34 | 35-42 | 43-49 | 50-55 | >55 |

                **Mujeres (ml/kg/min):**
                | Edad | Pobre | Regular | Bueno | Muy bueno | Excelente | Élite |
                |------|-------|---------|-------|-----------|-----------|-------|
                | 20-29 | <32 | 32-37 | 38-43 | 44-49 | 50-55 | >55 |
                | 30-39 | <29 | 29-34 | 35-40 | 41-46 | 47-52 | >52 |
                | 40-49 | <26 | 26-31 | 32-37 | 38-43 | 44-49 | >49 |
                | 50-59 | <23 | 23-28 | 29-34 | 35-40 | 41-46 | >46 |

                #### ¿Cómo mejorar el VO2max?
                1. **Intervalos VO2max**: 3-5 min al 95-100% FCmax, con recuperación igual
                2. **Tempo runs**: 20-40 min al 85-90% FCmax
                3. **Volumen base**: Más kilómetros en Z2 aumentan el VO2max gradualmente
                4. **Consistencia**: El VO2max mejora ~5-15% en 8-12 semanas de entrenamiento

                #### Limitaciones de la estimación
                - Es una **estimación**, no una medición directa
                - Más precisa con carreras de 5K-10K recientes
                - Puede variar ±3-5 ml/kg/min respecto a test de laboratorio
                """)

            with st.expander("🚴 Variability Index (VI) e Intensity Factor (IF)", expanded=False):
                st.markdown("""
                #### Variability Index (VI)
                El **VI** mide cuán variable es tu potencia durante un entrenamiento de ciclismo.

                ```
                VI = Potencia Normalizada / Potencia Media
                ```

                | Valor | Tipo de entrenamiento |
                |-------|----------------------|
                | 1.00 - 1.02 | Contrarreloj, rodillo muy constante |
                | 1.02 - 1.06 | Ruta llana, grupo organizado |
                | 1.06 - 1.13 | Ruta con subidas, grupo variable |
                | 1.13 - 1.20 | Criterium, carrera con ataques |
                | > 1.20 | MTB, carrera muy variable |

                #### Intensity Factor (IF)
                El **IF** compara tu potencia normalizada con tu FTP (umbral funcional).

                ```
                IF = Potencia Normalizada / FTP
                ```

                | Valor | Intensidad | Tipo de entrenamiento |
                |-------|------------|----------------------|
                | < 0.75 | Recuperación | Rodaje suave, Z1-Z2 |
                | 0.75 - 0.85 | Resistencia | Fondo largo, Z2-Z3 |
                | 0.85 - 0.95 | Tempo | Ritmo sostenido, Z3-Z4 |
                | 0.95 - 1.05 | Umbral | Esfuerzo de ~1 hora, Z4 |
                | 1.05 - 1.15 | VO2max | Intervalos duros, Z5 |
                | > 1.15 | Anaeróbico | Sprints, esfuerzos cortos |

                #### TSS (Training Stress Score)
                El **TSS** combina duración e intensidad para cuantificar la carga:
                ```
                TSS = (duración_horas × IF² × 100)
                ```

                | TSS | Recuperación necesaria |
                |-----|----------------------|
                | < 150 | Recuperación en 24h |
                | 150 - 300 | Algo de fatiga residual al día siguiente |
                | 300 - 450 | Fatiga notable, 2 días para recuperar |
                | > 450 | Fatiga severa, varios días de recuperación |
                """)

            with st.expander("🏆 Predicciones de Carrera - Fórmula de Riegel", expanded=False):
                st.markdown("""
                #### ¿Cómo funcionan las predicciones?
                Usamos la **fórmula de Riegel** (1981), validada científicamente para predecir
                tiempos de carrera basándose en un resultado conocido.

                ```
                T2 = T1 × (D2 / D1)^1.06
                ```

                Donde:
                - T1 = Tiempo conocido
                - D1 = Distancia conocida
                - T2 = Tiempo predicho
                - D2 = Distancia objetivo
                - 1.06 = Factor de fatiga (varía entre 1.05-1.08)

                #### Precisión de las predicciones
                | Predicción | Precisión típica |
                |------------|------------------|
                | 5K → 10K | ±1-2% |
                | 10K → Media Maratón | ±2-3% |
                | Media → Maratón | ±3-5% |
                | 5K → Maratón | ±5-8% |

                #### Factores que afectan la precisión
                1. **Distancia base**: Cuanto más cercana a la objetivo, más precisa
                2. **Recencia**: Carreras de los últimos 2-3 meses son más relevantes
                3. **Condiciones**: Temperatura, altitud, viento afectan el rendimiento
                4. **Experiencia**: Corredores experimentados predicen mejor distancias largas
                5. **Tipo de corredor**: Velocistas vs fondistas tienen diferentes factores

                #### Consejos para usar las predicciones
                - **Sé conservador**: En tu primera maratón, añade 5-10% al tiempo predicho
                - **Practica el ritmo**: Entrena al ritmo objetivo antes de la carrera
                - **Nutrición**: En distancias > 90 min, la nutrición es clave
                - **Tapering**: Descansa adecuadamente antes de la carrera objetivo
                """)

            with st.expander("📊 CTL, ATL y TSB - El Modelo PMC", expanded=False):
                st.markdown("""
                #### El Modelo de Gestión del Rendimiento (PMC)
                El **Performance Management Chart** usa tres métricas clave para modelar
                tu estado de forma y fatiga.

                #### CTL - Chronic Training Load (Fitness)
                Tu **forma física** acumulada. Media móvil exponencial de 42 días de TRIMP.
                ```
                CTL_hoy = CTL_ayer × 0.976 + TRIMP_hoy × 0.024
                ```

                | CTL | Nivel |
                |-----|-------|
                | < 40 | Principiante / Desentrenado |
                | 40 - 70 | Recreativo activo |
                | 70 - 100 | Aficionado serio |
                | 100 - 130 | Competidor amateur |
                | > 130 | Élite / Profesional |

                #### ATL - Acute Training Load (Fatigue)
                Tu **fatiga** reciente. Media móvil exponencial de 7 días de TRIMP.
                ```
                ATL_hoy = ATL_ayer × 0.857 + TRIMP_hoy × 0.143
                ```

                #### TSB - Training Stress Balance (Form)
                Tu **forma actual** = Fitness - Fatiga
                ```
                TSB = CTL - ATL
                ```

                | TSB | Estado | Recomendación |
                |-----|--------|---------------|
                | < -30 | Muy fatigado | Descanso obligatorio |
                | -30 a -10 | Fatigado | Entrenamiento suave |
                | -10 a +5 | Forma óptima | Ideal para competir |
                | +5 a +15 | Fresco | Bueno para competir |
                | +15 a +25 | Muy fresco | Posible pérdida de forma |
                | > +25 | Desentrenado | Necesitas entrenar más |

                #### Planificación con TSB
                - **Competición importante**: TSB entre +5 y +15
                - **Entrenamiento normal**: TSB entre -20 y +5
                - **Bloque de carga**: TSB puede bajar a -30
                - **Semana de descarga**: Subir TSB 10-15 puntos
                """)

            with st.expander("❤️ TRIMP - Training Impulse", expanded=False):
                st.markdown("""
                #### ¿Qué es TRIMP?
                El **TRIMP** (Training Impulse) es la unidad base para cuantificar la carga
                de entrenamiento. Combina duración e intensidad en un solo número.

                #### Fórmula de Banister
                ```
                TRIMP = Duración (min) × ΔHR × Factor_intensidad

                Donde:
                ΔHR = (FC_media - FC_reposo) / (FC_máx - FC_reposo)
                Factor_hombres = 0.64 × e^(1.92 × ΔHR)
                Factor_mujeres = 0.86 × e^(1.67 × ΔHR)
                ```

                #### Valores típicos de TRIMP
                | Actividad | TRIMP aproximado |
                |-----------|------------------|
                | Rodaje suave 30 min | 20-40 |
                | Rodaje moderado 60 min | 60-100 |
                | Entrenamiento intenso 60 min | 100-150 |
                | Carrera 10K competición | 80-120 |
                | Media maratón | 150-250 |
                | Maratón | 300-500 |

                #### Carga semanal recomendada
                | Nivel | TRIMP semanal |
                |-------|---------------|
                | Principiante | 200-400 |
                | Intermedio | 400-700 |
                | Avanzado | 700-1000 |
                | Élite | 1000-1500+ |

                #### Limitaciones del TRIMP
                - No captura bien el entrenamiento de fuerza
                - Puede subestimar intervalos muy cortos
                - Requiere datos precisos de FC
                - No diferencia entre tipos de estrés (muscular vs cardiovascular)
                """)
        else:
            st.warning("⚠️ Por favor sube los datos de Garmin en la pestaña 'Subir Datos' primero")
            st.info("""
            ### 📊 Métricas Avanzadas Disponibles

            Una vez subas tus datos, tendrás acceso a:

            - **ACWR (Acute:Chronic Workload Ratio)**: Prevención de lesiones
            - **Ramp Rate**: Tasa de progresión segura
            - **Monotonía y Strain**: Gestión de fatiga (Foster)
            - **Efficiency Factor**: Eficiencia aeróbica
            - **Decoupling**: Resistencia aeróbica
            - **VO2max Estimado**: Capacidad aeróbica
            - **Predicciones de Carrera**: Tiempos estimados para 5K, 10K, Media y Maratón
            - **Alertas Automáticas**: Avisos de riesgo de lesión o sobreentrenamiento
            - **Recomendación Diaria**: Qué tipo de entrenamiento hacer hoy
            """)

    # Tab 3: Training Zones
    with tab3:
        st.header("🎯 Zonas de Entrenamiento Personalizadas")

        zones_calculator = TrainingZones(max_hr, vo2_max, age, weight)
        zones = zones_calculator.calculate_zones()

        # Display zones
        st.subheader("Zonas de Frecuencia Cardíaca")

        zone_colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF5722']
        zone_names = ['Zona 1: Recuperación', 'Zona 2: Base Aeróbica', 'Zona 3: Tempo',
                      'Zona 4: Umbral', 'Zona 5: VO2 Máx']
        zone_descriptions = [
            'Actividad muy ligera, rodajes de recuperación',
            'Carrera fácil, quema de grasas, construcción de resistencia',
            'Esfuerzo moderado, mejora de capacidad aeróbica',
            'Esfuerzo intenso, entrenamiento de umbral de lactato',
            'Esfuerzo máximo, desarrollo de velocidad y potencia'
        ]

        for i, (zone_name, zone_data) in enumerate(zones.items()):
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    st.markdown(f"**{zone_names[i]}**")
                with col2:
                    st.markdown(f"**{zone_data['min_hr']} - {zone_data['max_hr']} ppm**")
                with col3:
                    st.progress((i + 1) / 5)
                st.caption(zone_descriptions[i])
                st.divider()

        # Zone visualization
        fig_zones = go.Figure()
        for i, (zone_name, zone_data) in enumerate(zones.items()):
            fig_zones.add_trace(go.Bar(
                name=zone_names[i],
                x=[zone_names[i]],
                y=[zone_data['max_hr'] - zone_data['min_hr']],
                base=zone_data['min_hr'],
                marker_color=zone_colors[i],
                text=f"{zone_data['min_hr']}-{zone_data['max_hr']}",
                textposition='inside'
            ))

        fig_zones.update_layout(
            title='Visualización de Zonas de Frecuencia Cardíaca',
            yaxis_title='Frecuencia Cardíaca (ppm)',
            showlegend=True,
            barmode='group'
        )
        st.plotly_chart(fig_zones, use_container_width=True)

        # Pace zones if PR available
        if best_5k_time or best_10k_time:
            st.subheader("⏱️ Zonas de Ritmo")
            pace_zones = zones_calculator.calculate_pace_zones(best_5k_time, best_10k_time)

            pace_translations = {
                'Easy Pace': 'Ritmo Suave',
                'Marathon Pace': 'Ritmo Maratón',
                'Threshold Pace': 'Ritmo Umbral',
                'Interval Pace': 'Ritmo Intervalos',
                'Repetition Pace': 'Ritmo Repeticiones',
                'Recuperación': 'Recuperación'
            }

            pace_colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF9800', '#FF5722']
            pace_icons = ['🚶', '🏃', '🏃‍♂️', '⚡', '🔥', '🚀']

            for i, (zone_type, pace_data) in enumerate(pace_zones.items()):
                translated = pace_translations.get(zone_type, zone_type)
                color = pace_colors[i % len(pace_colors)]
                icon = pace_icons[i % len(pace_icons)]

                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 3])
                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, {color}22, transparent);
                                    padding: 8px 12px; border-left: 4px solid {color}; border-radius: 4px;">
                            <span style="font-size: 1.1em;">{icon} <strong>{translated}</strong></span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 8px;">
                            <span style="font-size: 1.3em; font-weight: bold; color: {color};">
                                {pace_data['pace_min']} - {pace_data['pace_max']}
                            </span>
                            <span style="font-size: 0.9em; color: #666;"> /km</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div style="padding: 8px; color: #555; font-style: italic;">
                            {pace_data['description']}
                        </div>
                        """, unsafe_allow_html=True)
                st.divider()

            # Pace zones visualization chart
            fig_pace = go.Figure()
            pace_names = [pace_translations.get(z, z) for z in pace_zones.keys()]

            for i, (zone_type, pace_data) in enumerate(pace_zones.items()):
                # Convert pace to seconds for visualization
                pace_min_parts = pace_data['pace_min'].split(':')
                pace_max_parts = pace_data['pace_max'].split(':')
                pace_min_sec = int(pace_min_parts[0]) * 60 + int(pace_min_parts[1])
                pace_max_sec = int(pace_max_parts[0]) * 60 + int(pace_max_parts[1])

                fig_pace.add_trace(go.Bar(
                    name=pace_translations.get(zone_type, zone_type),
                    x=[pace_translations.get(zone_type, zone_type)],
                    y=[pace_max_sec - pace_min_sec],
                    base=pace_min_sec,
                    marker_color=pace_colors[i % len(pace_colors)],
                    text=f"{pace_data['pace_min']} - {pace_data['pace_max']}",
                    textposition='inside',
                    hovertemplate=f"<b>{pace_translations.get(zone_type, zone_type)}</b><br>" +
                                  f"Ritmo: {pace_data['pace_min']} - {pace_data['pace_max']} /km<br>" +
                                  f"{pace_data['description']}<extra></extra>"
                ))

            fig_pace.update_layout(
                title='📊 Visualización de Zonas de Ritmo',
                yaxis_title='Ritmo (segundos/km)',
                showlegend=False,
                barmode='group',
                yaxis=dict(
                    tickmode='array',
                    tickvals=[180, 210, 240, 270, 300, 330, 360, 390, 420, 450],
                    ticktext=['3:00', '3:30', '4:00', '4:30', '5:00', '5:30', '6:00', '6:30', '7:00', '7:30']
                )
            )
            st.plotly_chart(fig_pace, use_container_width=True)

    # Tab 4: Training Plan
    with tab4:
        st.header("📅 Generador de Plan de Entrenamiento")

        col_goal1, col_goal2 = st.columns(2)

        with col_goal1:
            target_date = st.date_input(
                "Fecha Objetivo de la Carrera",
                min_value=datetime.now().date() + timedelta(days=28),
                value=datetime.now().date() + timedelta(days=84)
            )

        with col_goal2:
            distance_goal = st.selectbox(
                "Distancia de la Carrera",
                options=["10K", "15K", "Media Maratón (21K)", "Maratón (42K)"],
                index=2
            )

        # Map distance
        distance_map = {
            "10K": 10,
            "15K": 15,
            "Media Maratón (21K)": 21.1,
            "Maratón (42K)": 42.2
        }
        target_distance = distance_map[distance_goal]

        weeks_to_race = (target_date - datetime.now().date()).days // 7
        st.info(f"📆 **{weeks_to_race} semanas** hasta tu carrera!")

        if weeks_to_race < 4:
            st.warning("⚠️ Se recomiendan al menos 4 semanas para una preparación adecuada")

        if st.button("🎯 Generar Plan de Entrenamiento", type="primary", use_container_width=True):
            # Get fitness status or use defaults
            if st.session_state.fitness_status:
                current_fitness = st.session_state.fitness_status
            else:
                current_fitness = {
                    'avg_weekly_distance': 20,
                    'fitness_level': 'Principiante',
                    'training_load': 50
                }

            # Generate plan
            zones_calc = TrainingZones(max_hr, vo2_max, age, weight)
            zones = zones_calc.calculate_zones()

            generator = TrainingPlanGenerator(
                target_distance=target_distance,
                target_date=target_date,
                current_fitness=current_fitness,
                training_zones=zones,
                intensity_preference=selected_intensity,
                preferred_workouts=preferred_workouts,
                training_days_per_week=training_days_per_week,
                best_5k_time=best_5k_time,
                best_10k_time=best_10k_time
            )

            plan = generator.generate_plan()
            st.session_state.training_plan = plan
            st.success("✅ ¡Plan de entrenamiento generado correctamente!")

        # Display plan
        if st.session_state.training_plan:
            plan = st.session_state.training_plan

            st.subheader("📊 Resumen del Plan de Entrenamiento")

            # Summary metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                total_distance = sum(w['total_distance'] for w in plan['weeks'])
                st.metric("Distancia Total de Entrenamiento", f"{total_distance:.1f} km")
            with col_m2:
                st.metric("Semanas Totales", len(plan['weeks']))
            with col_m3:
                st.metric("Distancia Semana Pico", f"{plan['peak_week_distance']:.1f} km")

            # Weekly volume chart
            weekly_distances = [w['total_distance'] for w in plan['weeks']]
            week_numbers = [f"Semana {i+1}" for i in range(len(plan['weeks']))]

            fig_plan = px.bar(
                x=week_numbers,
                y=weekly_distances,
                title='Volumen de Entrenamiento Semanal',
                labels={'x': 'Semana', 'y': 'Distancia (km)'}
            )

            # Highlight recovery weeks
            colors = ['#FF9800' if w.get('is_recovery') else '#1E88E5' for w in plan['weeks']]
            fig_plan.update_traces(marker_color=colors)
            st.plotly_chart(fig_plan, use_container_width=True)

            # Detailed weekly view
            st.subheader("📋 Calendario Semanal")

            selected_week = st.selectbox(
                "Seleccionar Semana",
                options=list(range(1, len(plan['weeks']) + 1)),
                format_func=lambda x: f"Semana {x}" + (" (Recuperación)" if plan['weeks'][x-1].get('is_recovery') else "")
            )

            week_data = plan['weeks'][selected_week - 1]

            # Week header
            week_type = "🔄 Semana de Recuperación" if week_data.get('is_recovery') else "💪 Semana de Entrenamiento"
            st.markdown(f"### {week_type}")
            st.markdown(f"**Semana {selected_week}** | Total: {week_data['total_distance']:.1f} km")

            # Daily workouts
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            days_es = {'Monday': 'Lun', 'Tuesday': 'Mar', 'Wednesday': 'Mié', 'Thursday': 'Jue',
                       'Friday': 'Vie', 'Saturday': 'Sáb', 'Sunday': 'Dom'}

            workout_translations = {
                'Rest': 'Descanso',
                'Easy Run': 'Rodaje Suave',
                'Tempo Run': 'Series Tempo',
                'Intervals': 'Intervalos',
                'Long Run': 'Tirada Larga',
                'Recovery Run': 'Rodaje Recuperación',
                'Hill Repeats': 'Cuestas',
                'Fartlek': 'Fartlek'
            }

            cols = st.columns(7)
            for i, day in enumerate(days):
                with cols[i]:
                    workout = week_data['workouts'].get(day, {'type': 'Rest', 'distance': 0})

                    workout_type = workout.get('type', 'Rest')
                    workout_type_es = workout_translations.get(workout_type, workout_type)
                    workout_colors = {
                        'Rest': '#9E9E9E',
                        'Easy Run': '#4CAF50',
                        'Tempo Run': '#FF9800',
                        'Intervals': '#F44336',
                        'Long Run': '#2196F3',
                        'Recovery Run': '#8BC34A',
                        'Hill Repeats': '#9C27B0',
                        'Fartlek': '#00BCD4'
                    }

                    color = workout_colors.get(workout_type, '#607D8B')

                    st.markdown(f"""
                    <div style='background-color: {color}; color: white; padding: 10px;
                                border-radius: 8px; text-align: center; min-height: 120px;'>
                        <strong>{days_es[day]}</strong><br>
                        <small>{workout_type_es}</small><br>
                        {workout.get('distance', 0):.1f} km<br>
                        <small>Zona {workout.get('zone', '-')}</small>
                    </div>
                    """, unsafe_allow_html=True)

            # Workout details - Enhanced with segments
            days_full_es = {'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                           'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
            st.subheader("📝 Detalles del Entrenamiento")

            for day, workout in week_data['workouts'].items():
                if workout.get('type') != 'Rest':
                    workout_type_es = workout.get('type_es', workout_translations.get(workout.get('type', 'Workout'), workout.get('type', 'Entrenamiento')))
                    with st.expander(f"📌 {days_full_es.get(day, day)}: {workout_type_es} ({workout.get('distance', 0):.1f} km)"):

                        # Resumen general
                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            st.markdown(f"**🎯 Objetivo:** {workout.get('description', 'N/A')}")
                            if workout.get('structure'):
                                st.markdown(f"**📋 Estructura:** {workout.get('structure')}")
                        with col_w2:
                            st.markdown(f"**❤️ Zona FC principal:** {workout.get('zone', 'N/A')}")
                            st.markdown(f"**💓 Rango FC:** {workout.get('hr_min', 'N/A')} - {workout.get('hr_max', 'N/A')} ppm")

                        # Segmentos detallados
                        segments = workout.get('segments', [])
                        if segments:
                            st.markdown("---")
                            st.markdown("#### 🔄 Segmentos del Entrenamiento")

                            for idx, segment in enumerate(segments, 1):
                                segment_color = '#4CAF50' if segment.get('zone', 2) in [1, 2] else '#FF9800' if segment.get('zone') == 3 else '#F44336'

                                st.markdown(f"""
                                <div style='background-color: {segment_color}15; border-left: 4px solid {segment_color};
                                            padding: 12px; margin: 8px 0; border-radius: 0 8px 8px 0;'>
                                    <strong style='color: {segment_color};'>{idx}. {segment.get('name', 'Segmento')}</strong>
                                </div>
                                """, unsafe_allow_html=True)

                                col_s1, col_s2, col_s3 = st.columns(3)

                                with col_s1:
                                    if segment.get('reps'):
                                        rep_dist = segment.get('rep_distance', segment.get('rep_duration', ''))
                                        unit = 'm' if segment.get('rep_distance') else 's'
                                        st.markdown(f"🔁 **Repeticiones:** {segment['reps']}x {rep_dist}{unit}")
                                    if segment.get('distance_km'):
                                        st.markdown(f"📏 **Distancia:** {segment['distance_km']:.1f} km")
                                    if segment.get('duration_min'):
                                        st.markdown(f"⏱️ **Duración:** {segment['duration_min']:.0f} min")
                                    if segment.get('incline'):
                                        st.markdown(f"⛰️ **Pendiente:** {segment['incline']}")

                                with col_s2:
                                    st.markdown(f"🏃 **Ritmo:** {segment.get('pace', 'N/A')}")
                                    if segment.get('pace_range') and segment.get('pace_range') != segment.get('pace'):
                                        st.markdown(f"📊 **Rango:** {segment['pace_range']}")
                                    if segment.get('fast_pace'):
                                        st.markdown(f"⚡ **Ritmo rápido:** {segment['fast_pace']}")
                                    if segment.get('slow_pace'):
                                        st.markdown(f"🚶 **Ritmo suave:** {segment['slow_pace']}")

                                with col_s3:
                                    st.markdown(f"❤️ **FC:** {segment.get('hr_min', 'N/A')} - {segment.get('hr_max', 'N/A')} ppm")
                                    st.markdown(f"🎯 **Zona:** {segment.get('zone', 'N/A')}")

                                # Mostrar recuperación con más detalle
                                if segment.get('recovery_duration') or segment.get('rest_after'):
                                    recovery_info = []
                                    if segment.get('recovery_duration'):
                                        rec_time = segment['recovery_duration']
                                        if rec_time >= 60:
                                            time_str = f"{rec_time // 60}:{rec_time % 60:02d}" if rec_time % 60 else f"{rec_time // 60} min"
                                        else:
                                            time_str = f"{rec_time}s"
                                        recovery_info.append(time_str)
                                    if segment.get('recovery_distance'):
                                        recovery_info.append(f"{segment['recovery_distance']}m")
                                    if segment.get('recovery_type'):
                                        recovery_info.append(segment['recovery_type'])

                                    if recovery_info:
                                        st.markdown(f"💤 **Recuperación:** {' '.join(recovery_info)}")
                                    elif segment.get('rest_after'):
                                        st.markdown(f"💤 **Recuperación:** {segment['rest_after']}")

                                if segment.get('notes'):
                                    st.info(f"💡 {segment['notes']}")

                                st.markdown("")  # Spacer

    # Tab 4b: Interactive Calendar
    with tab4b:
        st.header("📆 Calendario de Entrenamiento Interactivo")

        if st.session_state.training_plan:
            plan = st.session_state.training_plan

            # Create athlete profile for calendar
            athlete_profile = {
                'age': age,
                'weight': weight,
                'max_hr': max_hr,
                'resting_hr': resting_hr,
                'vo2_max': vo2_max
            }

            # Initialize calendar view
            calendar_view = TrainingCalendarView(plan, athlete_profile)

            # Render the interactive calendar
            calendar_view.render_calendar_view()

            st.divider()

            # Legend
            st.subheader("🎨 Leyenda de Colores")
            legend_cols = st.columns(4)

            workout_legend = [
                ('Descanso', '#E0E0E0'),
                ('Rodaje Suave', '#4CAF50'),
                ('Series Tempo', '#FF9800'),
                ('Intervalos', '#F44336'),
                ('Tirada Larga', '#2196F3'),
                ('Recuperación', '#8BC34A'),
                ('Cuestas', '#9C27B0'),
                ('Fartlek', '#00BCD4')
            ]

            for i, (name, color) in enumerate(workout_legend):
                with legend_cols[i % 4]:
                    st.markdown(f"""
                    <div style='display: flex; align-items: center; margin: 4px 0;'>
                        <div style='width: 20px; height: 20px; background: {color};
                                    border-radius: 4px; margin-right: 8px;'></div>
                        <span>{name}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Primero genera un plan de entrenamiento en la pestaña 'Plan de Entrenamiento'")
            st.info("💡 Una vez generado el plan, aquí podrás ver un calendario interactivo con todos tus entrenamientos.")

    # Tab 5: Export
    with tab5:
        st.header("📤 Exportar Plan de Entrenamiento")

        if st.session_state.training_plan:
            plan = st.session_state.training_plan

            col_exp1, col_exp2, col_exp3 = st.columns(3)

            with col_exp1:
                st.subheader("📄 Exportar CSV")
                # Convert to DataFrame
                export_data = []
                for week in plan['weeks']:
                    week_num = week['week_number']
                    for day, workout in week['workouts'].items():
                        export_data.append({
                            'Semana': week_num,
                            'Día': day,
                            'Tipo de Entrenamiento': workout.get('type', 'Descanso'),
                            'Distancia (km)': workout.get('distance', 0),
                            'Zona FC': workout.get('zone', ''),
                            'FC Mín': workout.get('hr_min', ''),
                            'FC Máx': workout.get('hr_max', ''),
                            'Descripción': workout.get('description', '')
                        })

                export_df = pd.DataFrame(export_data)
                csv_buffer = export_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    "⬇️ Descargar CSV",
                    data=csv_buffer,
                    file_name="plan_entrenamiento.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_exp2:
                st.subheader("📊 Exportar Excel")
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='Plan de Entrenamiento', index=False)

                    # Add summary sheet
                    summary_data = {
                        'Métrica': ['Semanas Totales', 'Distancia Total', 'Distancia Semana Pico',
                                   'Carrera Objetivo', 'Fecha Objetivo'],
                        'Valor': [len(plan['weeks']), f"{sum(w['total_distance'] for w in plan['weeks']):.1f} km",
                                  f"{plan['peak_week_distance']:.1f} km", distance_goal,
                                  str(target_date)]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Resumen', index=False)

                st.download_button(
                    "⬇️ Descargar Excel",
                    data=excel_buffer.getvalue(),
                    file_name="plan_entrenamiento.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with col_exp3:
                st.subheader("⌚ Archivos Garmin FIT")
                st.caption("Exportar entrenamientos en formato Garmin FIT")

                selected_week_export = st.selectbox(
                    "Seleccionar Semana a Exportar",
                    options=list(range(1, len(plan['weeks']) + 1)),
                    format_func=lambda x: f"Semana {x}",
                    key="export_week"
                )

                if st.button("⬇️ Generar Entrenamientos Garmin", use_container_width=True):
                    exporter = GarminFitExporter()
                    week_data = plan['weeks'][selected_week_export - 1]

                    try:
                        zip_buffer = exporter.export_week_to_fit(week_data, selected_week_export)

                        st.download_button(
                            "⬇️ Descargar Archivos FIT (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"entrenamientos_garmin_semana_{selected_week_export}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                        st.success("¡Archivos FIT generados! Súbelos a Garmin Connect.")
                    except Exception as e:
                        st.error(f"Error al generar archivos FIT: {str(e)}")

            # PDF Export
            st.divider()
            st.subheader("📕 Exportar PDF Profesional")
            st.caption("Genera un PDF con diseño profesional incluyendo perfil del atleta, zonas y calendario completo")

            col_pdf1, col_pdf2 = st.columns(2)
            with col_pdf1:
                pdf_start_week = st.number_input(
                    "Semana Inicial",
                    min_value=1,
                    max_value=len(plan['weeks']),
                    value=1,
                    key="pdf_start_week"
                )
            with col_pdf2:
                pdf_end_week = st.number_input(
                    "Semana Final",
                    min_value=1,
                    max_value=len(plan['weeks']),
                    value=len(plan['weeks']),
                    key="pdf_end_week"
                )

            pdf_include_details = st.checkbox("Incluir detalles de segmentos", value=True, key="pdf_details")

            if st.button("📕 Generar PDF", type="primary", use_container_width=True):
                try:
                    # Create athlete profile
                    athlete_profile = {
                        'age': age,
                        'weight': weight,
                        'max_hr': max_hr,
                        'resting_hr': resting_hr,
                        'vo2_max': vo2_max
                    }

                    # Get training zones
                    zones_calc = TrainingZones(max_hr, vo2_max, age, weight)
                    zones = zones_calc.calculate_zones()

                    # Format HR zones for PDF
                    training_zones_formatted = {}
                    for i, (zone_name, zone_data) in enumerate(zones.items(), 1):
                        training_zones_formatted[f'zone_{i}'] = {
                            'hr_min': zone_data['min_hr'],
                            'hr_max': zone_data['max_hr'],
                        }

                    # Add additional profile data for pace zones calculation
                    athlete_profile['best_5k_time'] = best_5k_time
                    athlete_profile['best_10k_time'] = best_10k_time

                    # Add fitness score data if available
                    if st.session_state.fitness_score is not None:
                        athlete_profile['fitness_score'] = st.session_state.fitness_score

                    # Generate PDF with pace zones
                    pdf_exporter = TrainingPlanPDFExporter(
                        plan,
                        athlete_profile,
                        training_zones_formatted
                    )

                    pdf_bytes = pdf_exporter.generate_pdf(
                        start_week=pdf_start_week,
                        end_week=pdf_end_week,
                        include_details=pdf_include_details
                    )

                    st.download_button(
                        "⬇️ Descargar PDF",
                        data=pdf_bytes,
                        file_name="plan_entrenamiento.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ ¡PDF generado correctamente!")

                except ImportError:
                    st.error("❌ Por favor instala reportlab: pip install reportlab")
                except Exception as e:
                    st.error(f"❌ Error al generar PDF: {str(e)}")

            # JSON Export for full plan
            st.divider()
            st.subheader("🔧 Exportar Plan Completo (JSON)")
            json_data = json.dumps(plan, indent=2, default=str)
            st.download_button(
                "⬇️ Descargar Plan Completo (JSON)",
                data=json_data,
                file_name="plan_entrenamiento_completo.json",
                mime="application/json"
            )
        else:
            st.warning("⚠️ Primero genera un plan de entrenamiento en la pestaña 'Plan de Entrenamiento'")

if __name__ == "__main__":
    main()


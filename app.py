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
from io import BytesIO
import json
import struct
import zipfile

# Import local modules
from training_zones import TrainingZones
from plan_generator import TrainingPlanGenerator
from garmin_analyzer import GarminDataAnalyzer
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
    tab1, tab2, tab2b, tab3, tab4, tab4b, tab5 = st.tabs([
        "📁 Subir Datos",
        "📈 Análisis de Forma",
        "🏆 Fitness Score",
        "🎯 Zonas de Entrenamiento",
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
                st.session_state.fitness_status = analyzer.analyze_fitness(df, max_hr)

                # Calculate fitness score with TRIMP
                st.session_state.fitness_score = analyzer.calculate_fitness_score(
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


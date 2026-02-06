"""
Calendar View Module - Domin Cideam.es
Vista de calendario interactivo para planes de entrenamiento.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import plotly.graph_objects as go
import calendar


class TrainingCalendarView:
    """Vista de calendario interactivo para el plan de entrenamiento."""

    # Colores para tipos de entrenamiento
    WORKOUT_COLORS = {
        # Running
        'Rest': '#E0E0E0',
        'Easy Run': '#4CAF50',
        'Tempo Run': '#FF9800',
        'Intervals': '#F44336',
        'Long Run': '#2196F3',
        'Recovery Run': '#8BC34A',
        'Hill Repeats': '#9C27B0',
        'Fartlek': '#00BCD4',
        # Cycling
        'Endurance Ride': '#43A047',
        'Recovery Ride': '#81C784',
        'Long Ride': '#1565C0',
        'Tempo Ride': '#EF6C00',
        'Sweet Spot': '#FFA000',
        'Threshold Intervals': '#C62828',
        'VO2max Intervals': '#AD1457',
        'Hill Ride': '#6A1B9A',
        # Swimming
        'Easy Swim': '#00ACC1',
        'Threshold Swim': '#00838F',
        'Interval Swim': '#006064',
        'Open Water Swim': '#0097A7',
        # Triathlon Brick/Transition
        'Sweet Spot Ride': '#FFA000',
        'Intervals Run': '#F44336',
        'Brick Workout': '#D81B60',
        'Transition Practice': '#7B1FA2',
    }

    # Traducciones al español
    WORKOUT_TRANSLATIONS = {
        # Running
        'Rest': 'Descanso',
        'Easy Run': 'Rodaje Suave',
        'Tempo Run': 'Series Tempo',
        'Intervals': 'Intervalos',
        'Long Run': 'Tirada Larga',
        'Recovery Run': 'Recuperación',
        'Hill Repeats': 'Cuestas',
        'Fartlek': 'Fartlek',
        # Cycling
        'Endurance Ride': 'Rodaje Resistencia',
        'Recovery Ride': 'Rodaje Recuperación',
        'Long Ride': 'Fondo Largo Bici',
        'Tempo Ride': 'Rodaje Tempo',
        'Sweet Spot': 'Sweet Spot',
        'Threshold Intervals': 'Intervalos Umbral',
        'VO2max Intervals': 'Intervalos VO2max',
        'Hill Ride': 'Subidas',
        # Swimming
        'Easy Swim': 'Natación Suave',
        'Threshold Swim': 'Natación Umbral',
        'Interval Swim': 'Natación Intervalos',
        'Open Water Swim': 'Aguas Abiertas',
        # Triathlon
        'Sweet Spot Ride': 'Sweet Spot Bici',
        'Intervals Run': 'Series Carrera',
        'Brick Workout': 'Ent. Brick',
        'Transition Practice': 'Transiciones',
    }

    DAYS_ES = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }

    DAYS_SHORT_ES = {
        'Monday': 'Lun', 'Tuesday': 'Mar', 'Wednesday': 'Mié',
        'Thursday': 'Jue', 'Friday': 'Vie', 'Saturday': 'Sáb', 'Sunday': 'Dom'
    }

    def __init__(self, training_plan: dict, athlete_profile: dict = None):
        """
        Inicializa la vista de calendario.

        Args:
            training_plan: Plan de entrenamiento generado
            athlete_profile: Perfil del atleta (opcional)
        """
        self.plan = training_plan
        self.profile = athlete_profile or {}
        self._build_calendar_data()

    def _build_calendar_data(self):
        """Construye la estructura de datos del calendario."""
        self.calendar_data = {}
        self.weekly_summaries = []

        for week in self.plan.get('weeks', []):
            week_num = week['week_number']
            start_date = week.get('start_date', datetime.now().date())

            if isinstance(start_date, datetime):
                start_date = start_date.date()

            week_summary = {
                'week_number': week_num,
                'phase': week.get('phase', 'build'),
                'is_recovery': week.get('is_recovery', False),
                'total_distance': week.get('total_distance', 0),
                'start_date': start_date,
                'workouts_count': 0,
                'total_duration_min': 0
            }

            # Mapear días a fechas
            day_offsets = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }

            for day_name, workout in week.get('workouts', {}).items():
                offset = day_offsets.get(day_name, 0)
                workout_date = start_date + timedelta(days=offset)

                self.calendar_data[workout_date] = {
                    'workout': workout,
                    'week_number': week_num,
                    'day_name': day_name,
                    'phase': week.get('phase', 'build'),
                    'is_recovery_week': week.get('is_recovery', False)
                }

                if workout.get('type') != 'Rest':
                    week_summary['workouts_count'] += 1
                    distance = workout.get('distance', 0)
                    # Calcular duración desde segmentos si están disponibles
                    segments = workout.get('segments', [])
                    duration = 0
                    if segments:
                        for seg in segments:
                            if seg.get('duration_min'):
                                duration += seg['duration_min']
                            elif seg.get('distance_km'):
                                pace_min = self._parse_pace(seg.get('pace', ''))
                                if pace_min:
                                    duration += seg['distance_km'] * pace_min
                                else:
                                    duration += seg['distance_km'] * 6  # fallback ~6 min/km
                    else:
                        duration = distance * 6  # fallback ~6 min/km
                    week_summary['total_duration_min'] += duration

            self.weekly_summaries.append(week_summary)

    def render_calendar_view(self):
        """Renderiza la vista completa del calendario."""
        st.header("📅 Calendario de Entrenamiento Interactivo")

        # Selector de vista
        view_type = st.radio(
            "Tipo de Vista",
            ["Vista Mensual", "Vista Semanal", "Vista Lista"],
            horizontal=True
        )

        if view_type == "Vista Mensual":
            self._render_monthly_view()
        elif view_type == "Vista Semanal":
            self._render_weekly_view()
        else:
            self._render_list_view()

        # Panel de detalle del día seleccionado
        self._render_day_detail_panel()

    def _render_monthly_view(self):
        """Renderiza la vista mensual del calendario."""
        if not self.weekly_summaries:
            st.warning("No hay datos de entrenamiento disponibles")
            return

        # Determinar meses disponibles
        all_dates = sorted(self.calendar_data.keys())
        if not all_dates:
            return

        start_month = all_dates[0].replace(day=1)
        end_month = all_dates[-1].replace(day=1)

        # Selector de mes
        months_available = []
        current = start_month
        while current <= end_month:
            months_available.append(current)
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        month_names_es = [
            '', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
        ]

        selected_month_idx = st.selectbox(
            "Seleccionar Mes",
            range(len(months_available)),
            format_func=lambda i: f"{month_names_es[months_available[i].month]} {months_available[i].year}"
        )

        selected_month = months_available[selected_month_idx]

        # Renderizar calendario del mes
        self._render_month_calendar(selected_month, month_names_es)

    def _render_month_calendar(self, month_date, month_names_es):
        """Renderiza el calendario de un mes específico."""
        year = month_date.year
        month = month_date.month

        st.subheader(f"📅 {month_names_es[month]} {year}")

        # Cabecera de días
        day_headers = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        cols = st.columns(7)
        for i, header in enumerate(day_headers):
            with cols[i]:
                st.markdown(f"**{header}**")

        # Obtener días del mes
        cal = calendar.Calendar(firstweekday=0)
        month_days = cal.monthdayscalendar(year, month)

        for week_row in month_days:
            cols = st.columns(7)
            for i, day in enumerate(week_row):
                with cols[i]:
                    if day == 0:
                        st.markdown("&nbsp;", unsafe_allow_html=True)
                    else:
                        current_date = month_date.replace(day=day)
                        self._render_calendar_day(current_date, day)

    def _render_calendar_day(self, date, day_number):
        """Renderiza un día individual del calendario."""
        day_data = self.calendar_data.get(date)

        if day_data:
            workout = day_data['workout']
            workout_type = workout.get('type', 'Rest')
            color = self.WORKOUT_COLORS.get(workout_type, '#9E9E9E')
            distance = workout.get('distance', 0)

            # Botón interactivo para el día
            button_key = f"day_{date.isoformat()}"

            if workout_type == 'Rest':
                st.markdown(f"""
                <div style='background: {color}; padding: 8px; border-radius: 8px;
                            text-align: center; min-height: 70px; cursor: pointer;'
                     onclick="document.getElementById('{button_key}').click()">
                    <strong>{day_number}</strong><br>
                    <small style='color: #666;'>Descanso</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                type_es = self.WORKOUT_TRANSLATIONS.get(workout_type, workout_type)[:10]
                sport = workout.get('sport', '')
                sport_icon = {'swim': '🏊', 'bike': '🚲', 'run': '🏃', 'brick': '🔄'}.get(sport, '')
                dist_label = f"{distance:.1f}km" if distance else ""
                st.markdown(f"""
                <div style='background: {color}; color: white; padding: 8px;
                            border-radius: 8px; text-align: center; min-height: 70px;'>
                    <strong>{day_number}</strong><br>
                    <small>{sport_icon} {type_es}</small><br>
                    <small>{dist_label}</small>
                </div>
                """, unsafe_allow_html=True)

            if st.button("📋", key=button_key, help=f"Ver detalles del {day_number}"):
                st.session_state.selected_calendar_day = date
        else:
            # Día sin entrenamiento programado
            st.markdown(f"""
            <div style='background: #FAFAFA; padding: 8px; border-radius: 8px;
                        text-align: center; min-height: 70px; color: #CCC;'>
                <strong>{day_number}</strong>
            </div>
            """, unsafe_allow_html=True)

    def _render_weekly_view(self):
        """Renderiza la vista semanal con más detalle."""
        st.subheader("📊 Vista Semanal Detallada")

        if not self.weekly_summaries:
            st.warning("No hay datos de entrenamiento")
            return

        # Selector de semana
        week_options = [
            f"Semana {w['week_number']}" + (" (Recup.)" if w['is_recovery'] else "")
            for w in self.weekly_summaries
        ]

        selected_week_idx = st.selectbox(
            "Seleccionar Semana",
            range(len(week_options)),
            format_func=lambda i: week_options[i]
        )

        week_summary = self.weekly_summaries[selected_week_idx]
        week_data = self.plan['weeks'][selected_week_idx]

        # Resumen de la semana
        self._render_week_summary(week_summary, week_data)

        # Tarjetas de entrenamientos
        self._render_week_workout_cards(week_data)


    def _render_week_summary(self, week_summary, week_data):
        """Renderiza el resumen de una semana."""
        phase_names = {
            'base': '🏗️ Fase Base',
            'build': '📈 Fase Construcción',
            'peak': '🏔️ Fase Pico',
            'taper': '🎯 Tapering'
        }

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📏 Distancia Total",
                f"{week_summary['total_distance']:.1f} km"
            )

        with col2:
            st.metric(
                "🏃 Entrenamientos",
                f"{week_summary['workouts_count']} días"
            )

        with col3:
            duration_hours = week_summary['total_duration_min'] / 60
            st.metric(
                "⏱️ Duración Est.",
                f"{duration_hours:.1f} h"
            )

        with col4:
            phase = phase_names.get(week_summary['phase'], week_summary['phase'])
            recovery_badge = " 🔄" if week_summary['is_recovery'] else ""
            st.metric("📊 Fase", f"{phase}{recovery_badge}")

    def _render_week_workout_cards(self, week_data):
        """Renderiza tarjetas de entrenamiento para cada día de la semana."""
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        st.markdown("### 📋 Entrenamientos de la Semana")

        for day in days_order:
            workout = week_data['workouts'].get(day, {'type': 'Rest'})
            workout_type = workout.get('type', 'Rest')

            if workout_type == 'Rest':
                continue

            day_es = self.DAYS_ES.get(day, day)
            type_es = workout.get('type_es', self.WORKOUT_TRANSLATIONS.get(workout_type, workout_type))
            color = self.WORKOUT_COLORS.get(workout_type, '#607D8B')
            sport = workout.get('sport', '')
            sport_icon = {'swim': '🏊', 'bike': '🚲', 'run': '🏃', 'brick': '🔄'}.get(sport, '📌')

            dist = workout.get('distance', 0)
            dist_str = f"{dist:.1f} km" if dist else ""

            with st.expander(
                f"{sport_icon} {day_es}: {type_es} {('- ' + dist_str) if dist_str else ''}",
                expanded=False
            ):
                # Información general
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**📏 Distancia:** {workout.get('distance', 0):.1f} km")
                    st.markdown(f"**🎯 Objetivo:** {workout.get('description', 'N/A')}")

                with col2:
                    st.markdown(f"**❤️ Zona FC:** {workout.get('zone', '-')}")
                    st.markdown(f"**💓 FC:** {workout.get('hr_min', '-')} - {workout.get('hr_max', '-')} ppm")

                with col3:
                    if workout.get('structure'):
                        st.markdown(f"**📋 Estructura:** {workout['structure']}")

                # Segmentos detallados
                segments = workout.get('segments', [])
                if segments:
                    st.markdown("---")
                    st.markdown("#### 🔄 Segmentos del Entrenamiento")

                    for idx, segment in enumerate(segments, 1):
                        self._render_segment(segment, idx)

    def _render_segment(self, segment, index):
        """Renderiza un segmento de entrenamiento."""
        zone = segment.get('zone', 2)
        if zone in [1, 2]:
            seg_color = '#4CAF50'
        elif zone == 3:
            seg_color = '#FF9800'
        else:
            seg_color = '#F44336'

        st.markdown(f"""
        <div style='background-color: {seg_color}15; border-left: 4px solid {seg_color};
                    padding: 12px; margin: 8px 0; border-radius: 0 8px 8px 0;'>
            <strong style='color: {seg_color};'>{index}. {segment.get('name', 'Segmento')}</strong>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            if segment.get('distance_km'):
                st.markdown(f"📏 **Distancia:** {segment['distance_km']:.1f} km")
            if segment.get('duration_min'):
                st.markdown(f"⏱️ **Duración:** {segment['duration_min']:.0f} min")
            if segment.get('reps'):
                rep_val = segment.get('rep_distance', segment.get('rep_duration', ''))
                unit = 'm' if segment.get('rep_distance') else 's'
                st.markdown(f"🔁 **Reps:** {segment['reps']} x {rep_val}{unit}")

        with col2:
            pace = segment.get('pace', 'N/A')
            # Detect power-based segments (contain 'W' for watts)
            if isinstance(pace, str) and ('W' in pace or 'FTP' in pace):
                st.markdown(f"⚡ **Potencia:** {pace}")
            elif isinstance(pace, str) and '/100m' in pace:
                st.markdown(f"🏊 **Ritmo:** {pace}")
            else:
                st.markdown(f"🏃 **Ritmo:** {pace}")
            if segment.get('pace_range'):
                st.markdown(f"📊 **Rango:** {segment['pace_range']}")

        with col3:
            hr_min = segment.get('hr_min', '-')
            hr_max = segment.get('hr_max', '-')
            if hr_min and hr_max and hr_min != '' and hr_max != '':
                st.markdown(f"❤️ **FC:** {hr_min} - {hr_max} ppm")
            st.markdown(f"🎯 **Zona:** {segment.get('zone', '-')}")

        if segment.get('rest_after'):
            st.markdown(f"💤 **Recuperación:** {segment['rest_after']}")

        if segment.get('notes'):
            st.info(f"💡 {segment['notes']}")

    def _render_list_view(self):
        """Renderiza la vista de lista de entrenamientos."""
        st.subheader("📋 Lista de Entrenamientos")

        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            workout_filter = st.multiselect(
                "Filtrar por Tipo",
                list(self.WORKOUT_TRANSLATIONS.values()),
                default=[]
            )
        with col2:
            zone_filter = st.multiselect(
                "Filtrar por Zona FC",
                [1, 2, 3, 4, 5],
                default=[]
            )

        # Listar entrenamientos
        for date in sorted(self.calendar_data.keys()):
            day_data = self.calendar_data[date]
            workout = day_data['workout']
            workout_type = workout.get('type', 'Rest')

            if workout_type == 'Rest':
                continue

            type_es = workout.get('type_es', self.WORKOUT_TRANSLATIONS.get(workout_type, workout_type))

            # Aplicar filtros
            if workout_filter and type_es not in workout_filter:
                continue
            if zone_filter and workout.get('zone') not in zone_filter:
                continue

            day_es = self.DAYS_ES.get(day_data['day_name'], day_data['day_name'])
            color = self.WORKOUT_COLORS.get(workout_type, '#607D8B')

            st.markdown(f"""
            <div style='background: linear-gradient(90deg, {color}30, white);
                        padding: 12px; margin: 8px 0; border-radius: 8px;
                        border-left: 4px solid {color};'>
                <strong>{date.strftime('%d/%m/%Y')} - {day_es}</strong> |
                Semana {day_data['week_number']} |
                <span style='color: {color}; font-weight: bold;'>{type_es}</span> |
                {workout.get('distance', 0):.1f} km |
                Zona {workout.get('zone', '-')}
            </div>
            """, unsafe_allow_html=True)

    def _render_day_detail_panel(self):
        """Renderiza el panel de detalle del día seleccionado."""
        if 'selected_calendar_day' not in st.session_state:
            return

        selected_date = st.session_state.selected_calendar_day
        day_data = self.calendar_data.get(selected_date)

        if not day_data:
            return

        workout = day_data['workout']
        workout_type = workout.get('type', 'Rest')

        st.markdown("---")
        st.subheader(f"📋 Detalle: {selected_date.strftime('%d/%m/%Y')}")

        if workout_type == 'Rest':
            st.info("🧘 Día de descanso - Recuperación activa o descanso completo")
        else:
            type_es = workout.get('type_es', self.WORKOUT_TRANSLATIONS.get(workout_type, workout_type))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🏃 Tipo:** {type_es}")
                st.markdown(f"**📏 Distancia:** {workout.get('distance', 0):.1f} km")
                st.markdown(f"**🎯 Objetivo:** {workout.get('description', 'N/A')}")

            with col2:
                st.markdown(f"**❤️ Zona FC:** {workout.get('zone', '-')}")
                st.markdown(f"**💓 Rango FC:** {workout.get('hr_min', '-')} - {workout.get('hr_max', '-')} ppm")
                if workout.get('structure'):
                    st.markdown(f"**📋 Estructura:** {workout['structure']}")

            # Segmentos
            segments = workout.get('segments', [])
            if segments:
                st.markdown("#### 🔄 Segmentos")
                for idx, segment in enumerate(segments, 1):
                    self._render_segment(segment, idx)

    @staticmethod
    def _parse_pace(pace_str: str) -> float:
        """Parsea un ritmo como '5:30/km' y retorna el valor decimal en min/km, o 0 si no es válido."""
        if not pace_str or not isinstance(pace_str, str):
            return 0
        try:
            clean = pace_str.replace('/km', '').strip()
            parts = clean.split(':')
            if len(parts) == 2:
                return int(parts[0]) + int(parts[1]) / 60
        except (ValueError, IndexError):
            pass
        return 0

    def get_calendar_data_for_export(self) -> dict:
        """Retorna los datos del calendario formateados para exportación."""
        return {
            'calendar_data': self.calendar_data,
            'weekly_summaries': self.weekly_summaries,
            'plan': self.plan,
            'profile': self.profile
        }


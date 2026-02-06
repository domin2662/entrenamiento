"""
Cycling Plan Generator - Domin Cideam.es
Genera planes de entrenamiento de ciclismo periodizados basados en FTP y objetivos.
Incluye entrenamientos detallados con potencia, cadencia y zonas específicas.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math


class CyclingPlanGenerator:
    """Genera planes de entrenamiento de ciclismo personalizados con detalle completo."""

    # Límites de seguridad de km semanales según nivel de forma física
    WEEKLY_KM_LIMITS = {
        'Principiante': {'min': 50, 'max': 200, 'default': 100},
        'Intermedio': {'min': 80, 'max': 350, 'default': 180},
        'Avanzado': {'min': 120, 'max': 500, 'default': 280},
        'Élite': {'min': 150, 'max': 700, 'default': 400},
    }

    # Límites de seguridad de desnivel semanal (metros) según nivel
    WEEKLY_ELEVATION_LIMITS = {
        'Principiante': {'min': 0, 'max': 1500, 'default': 500},
        'Intermedio': {'min': 0, 'max': 3000, 'default': 1200},
        'Avanzado': {'min': 0, 'max': 5000, 'default': 2500},
        'Élite': {'min': 0, 'max': 8000, 'default': 4000},
    }

    def __init__(
        self,
        target_event: str,
        target_date: datetime,
        current_fitness: dict,
        training_zones: dict,
        ftp: int,
        power_zones: dict,
        intensity_preference: str = "moderate",
        preferred_workouts: dict = None,
        training_days_per_week: int = 4,
        weekly_km: Optional[float] = None,
        weekly_elevation_gain: Optional[int] = None
    ):
        """
        Inicializa el generador de planes de ciclismo.

        Args:
            target_event: Tipo de evento ('gran_fondo', 'criterium', 'contrarreloj', 'etapas')
            target_date: Fecha objetivo del evento
            current_fitness: Diccionario con métricas de forma física actual
            training_zones: Diccionario con zonas de FC
            ftp: FTP actual en watts
            power_zones: Diccionario con zonas de potencia
            intensity_preference: "easy", "moderate", o "high"
            preferred_workouts: Diccionario de tipos de entrenamiento preferidos
            training_days_per_week: Días de entrenamiento (3-6)
            weekly_km: Kilómetros totales semanales deseados (None = automático)
            weekly_elevation_gain: Desnivel positivo acumulado semanal en metros (None = automático)
        """
        self.target_event = target_event
        self.target_date = target_date
        self.current_fitness = current_fitness
        self.training_zones = training_zones
        self.ftp = ftp
        self.power_zones = power_zones
        self.intensity_preference = intensity_preference
        self.preferred_workouts = preferred_workouts or {
            "endurance": True, "tempo": True, "sweet_spot": True,
            "threshold": True, "vo2max": False, "sprints": False
        }
        self.training_days_per_week = min(6, max(3, training_days_per_week))

        today = datetime.now().date()
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        self.weeks_to_event = max(4, (target_date - today).days // 7)

        self.base_weekly_hours = current_fitness.get('avg_weekly_hours', 6)

        # Aplicar límites de seguridad a km y desnivel según nivel de forma
        fitness_level = current_fitness.get('fitness_level', 'Principiante')
        self._apply_safe_volume(weekly_km, weekly_elevation_gain, fitness_level)

        self._set_training_parameters()

    def _apply_safe_volume(self, weekly_km: Optional[float], weekly_elevation_gain: Optional[int], fitness_level: str):
        """Aplica límites de seguridad a los volúmenes semanales según nivel de forma."""
        km_limits = self.WEEKLY_KM_LIMITS.get(fitness_level, self.WEEKLY_KM_LIMITS['Principiante'])
        elev_limits = self.WEEKLY_ELEVATION_LIMITS.get(fitness_level, self.WEEKLY_ELEVATION_LIMITS['Principiante'])

        if weekly_km is not None:
            self.weekly_km = max(km_limits['min'], min(km_limits['max'], weekly_km))
        else:
            self.weekly_km = km_limits['default']

        if weekly_elevation_gain is not None:
            self.weekly_elevation_gain = max(elev_limits['min'], min(elev_limits['max'], weekly_elevation_gain))
        else:
            self.weekly_elevation_gain = elev_limits['default']

        # Derivar velocidad media estimada (km/h) a partir de km y horas base
        # Se usará para convertir horas a km en la generación de entrenamientos
        self.avg_speed_kmh = self.weekly_km / max(1, self.base_weekly_hours)

    def get_volume_limits(self) -> dict:
        """Devuelve los límites de seguridad actuales según el nivel de forma del usuario."""
        fitness_level = self.current_fitness.get('fitness_level', 'Principiante')
        return {
            'km': self.WEEKLY_KM_LIMITS.get(fitness_level, self.WEEKLY_KM_LIMITS['Principiante']),
            'elevation': self.WEEKLY_ELEVATION_LIMITS.get(fitness_level, self.WEEKLY_ELEVATION_LIMITS['Principiante']),
            'current_weekly_km': self.weekly_km,
            'current_weekly_elevation': self.weekly_elevation_gain,
            'fitness_level': fitness_level,
        }

    def _set_training_parameters(self):
        """Establece parámetros de entrenamiento según evento e intensidad."""
        event_params = {
            'gran_fondo': {'peak_hours': 14, 'long_ride_max_h': 5.0, 'tss_peak': 800},
            'criterium': {'peak_hours': 10, 'long_ride_max_h': 3.0, 'tss_peak': 600},
            'contrarreloj': {'peak_hours': 10, 'long_ride_max_h': 3.5, 'tss_peak': 550},
            'etapas': {'peak_hours': 16, 'long_ride_max_h': 5.5, 'tss_peak': 900}
        }
        params = event_params.get(self.target_event, event_params['gran_fondo'])
        self.peak_weekly_hours = params['peak_hours']
        self.long_ride_max_hours = params['long_ride_max_h']
        self.tss_peak = params['tss_peak']

        if self.intensity_preference == "easy":
            self.easy_pct = 0.80
            self.intensity_pct = 0.20
        elif self.intensity_preference == "high":
            self.easy_pct = 0.60
            self.intensity_pct = 0.40
        else:
            self.easy_pct = 0.70
            self.intensity_pct = 0.30

    def _calculate_phases(self) -> list:
        """Calcula las fases de periodización."""
        total = self.weeks_to_event
        if total <= 6:
            return [('build', total - 2), ('taper', 2)]
        taper = 2
        peak = max(2, int(total * 0.15))
        build = max(3, int(total * 0.35))
        base = total - build - peak - taper
        phases = []
        if base > 0:
            phases.append(('base', base))
        phases.append(('build', build))
        phases.append(('peak', peak))
        phases.append(('taper', taper))
        return phases

    def _is_recovery_week(self, week_num: int, phase: str) -> bool:
        """Determina si es semana de recuperación."""
        if phase == 'taper':
            return False
        return week_num % 4 == 0



    def _calculate_week_volume(self, phase: str, progress: float, is_recovery: bool) -> dict:
        """Calcula el volumen objetivo (km y desnivel) para una semana según fase y progreso."""
        # Factor de volumen según fase (porcentaje del pico)
        if phase == 'base':
            vol_factor = 0.6 + 0.2 * progress   # 60-80% del pico
        elif phase == 'build':
            vol_factor = 0.75 + 0.20 * progress  # 75-95% del pico
        elif phase == 'peak':
            vol_factor = 0.95 + 0.05 * progress  # 95-100% del pico
        else:  # taper
            vol_factor = 0.7 - 0.3 * progress    # 70-40% del pico

        if is_recovery:
            vol_factor *= 0.65

        target_km = round(self.weekly_km * vol_factor, 1)
        target_elevation = round(self.weekly_elevation_gain * vol_factor)

        return {
            'target_km': target_km,
            'target_elevation': target_elevation,
            'vol_factor': vol_factor,
        }

    def generate_plan(self) -> dict:
        """Genera el plan de entrenamiento completo de ciclismo."""
        weeks = []
        phases = self._calculate_phases()
        week_start_date = datetime.now().date()

        week_num = 0
        for phase, num_weeks in phases:
            for w in range(num_weeks):
                week_num += 1
                is_recovery = self._is_recovery_week(week_num, phase)

                progress = w / max(1, num_weeks - 1) if num_weeks > 1 else 1.0
                if phase == 'base':
                    target_hours = self.base_weekly_hours + (self.peak_weekly_hours * 0.7 - self.base_weekly_hours) * progress
                elif phase == 'build':
                    target_hours = self.peak_weekly_hours * (0.7 + 0.25 * progress)
                elif phase == 'peak':
                    target_hours = self.peak_weekly_hours * (0.95 + 0.05 * progress)
                else:
                    target_hours = self.peak_weekly_hours * (0.7 - 0.3 * progress)

                if is_recovery:
                    target_hours *= 0.65

                # Calcular volumen semanal (km y desnivel)
                week_volume = self._calculate_week_volume(phase, progress, is_recovery)

                week_workouts = self._generate_week_workouts(
                    week_num, target_hours, phase, is_recovery, week_volume
                )

                actual_hours = sum(
                    sum(s.get('duration_min', 0) for s in wo.get('segments', [])) / 60
                    for wo in week_workouts.values() if wo.get('type') != 'Rest'
                )
                estimated_distance = round(
                    sum(wo.get('distance', 0) for wo in week_workouts.values()), 1
                )
                total_elevation = sum(
                    wo.get('elevation_gain', 0) for wo in week_workouts.values()
                )

                weeks.append({
                    'week_number': week_num,
                    'phase': phase,
                    'is_recovery': is_recovery,
                    'target_hours': target_hours,
                    'total_distance': estimated_distance,
                    'total_elevation': total_elevation,
                    'target_km': week_volume['target_km'],
                    'target_elevation': week_volume['target_elevation'],
                    'start_date': week_start_date,
                    'workouts': week_workouts
                })
                week_start_date += timedelta(days=7)

        return {
            'weeks': weeks,
            'total_weeks': week_num,
            'target_event': self.target_event,
            'target_date': self.target_date,
            'ftp': self.ftp,
            'weekly_km': self.weekly_km,
            'weekly_elevation_gain': self.weekly_elevation_gain,
            'peak_week_distance': round(self.weekly_km, 1),
            'sport_type': 'cycling',
            'phases': phases
        }

    # Proporción de desnivel por tipo de entrenamiento
    _ELEVATION_DISTRIBUTION = {
        'Long Ride': 0.35,
        'Hill Ride': 0.40,
        'Endurance Ride': 0.10,
        'Tempo Ride': 0.05,
        'Sweet Spot': 0.03,
        'Threshold Intervals': 0.03,
        'VO2max Intervals': 0.02,
        'Recovery Ride': 0.02,
    }

    def _generate_week_workouts(self, week_num: int, target_hours: float, phase: str,
                                 is_recovery: bool, week_volume: dict = None) -> dict:
        """Genera entrenamientos de ciclismo para una semana."""
        workouts = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        schedule = self._create_workout_schedule(phase, is_recovery, week_num)

        target_km = week_volume['target_km'] if week_volume else self.weekly_km
        target_elevation = week_volume['target_elevation'] if week_volume else self.weekly_elevation_gain

        # Calcular cuántos entrenamientos hay (excluyendo Rest)
        active_types = [schedule.get(d, 'Rest') for d in days if schedule.get(d, 'Rest') != 'Rest']

        # Calcular peso total de elevación para normalizar la distribución
        total_elev_weight = sum(self._ELEVATION_DISTRIBUTION.get(t, 0.05) for t in active_types)

        remaining_hours = target_hours
        remaining_km = target_km
        for day in days:
            workout_type = schedule.get(day, 'Rest')
            if workout_type == 'Rest':
                workouts[day] = {
                    'type': 'Rest', 'type_es': 'Descanso', 'distance': 0,
                    'elevation_gain': 0,
                    'zone': '', 'hr_min': '', 'hr_max': '',
                    'description': 'Día de descanso completo o recuperación activa',
                    'structure': None, 'segments': []
                }
                continue

            dur_h = self._calculate_workout_duration(workout_type, remaining_hours, target_hours, phase)
            dur_min = dur_h * 60
            detailed = self._build_workout(workout_type, dur_min, phase, week_num)
            zone_info = self._get_zone_for_cycling_workout(workout_type)

            # Distribuir km proporcionalmente según duración
            distance = round(dur_h * self.avg_speed_kmh, 1)
            distance = min(distance, remaining_km) if remaining_km > 0 else round(dur_h * 28, 1)

            # Distribuir desnivel según tipo de entrenamiento
            elev_weight = self._ELEVATION_DISTRIBUTION.get(workout_type, 0.05)
            elevation = round(target_elevation * (elev_weight / max(0.01, total_elev_weight)))

            workouts[day] = {
                'type': workout_type,
                'type_es': detailed.get('type_es', workout_type),
                'distance': distance,
                'elevation_gain': elevation,
                'zone': zone_info['zone_num'],
                'hr_min': zone_info.get('min_hr', ''),
                'hr_max': zone_info.get('max_hr', ''),
                'description': detailed['description'],
                'structure': detailed['structure'],
                'segments': detailed['segments'],
                'sport': 'cycling'
            }
            remaining_hours -= dur_h
            remaining_km -= distance

        return workouts

    def _create_workout_schedule(self, phase: str, is_recovery: bool, week_num: int) -> dict:
        """Crea el horario semanal de entrenamientos de ciclismo."""
        schedule = {}
        if self.training_days_per_week == 3:
            day_slots = ['Tuesday', 'Thursday', 'Saturday']
        elif self.training_days_per_week == 4:
            day_slots = ['Tuesday', 'Thursday', 'Saturday', 'Sunday']
        elif self.training_days_per_week == 5:
            day_slots = ['Monday', 'Tuesday', 'Thursday', 'Saturday', 'Sunday']
        else:
            day_slots = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday']

        if is_recovery:
            for day in day_slots:
                if day == 'Saturday':
                    schedule[day] = 'Endurance Ride'
                else:
                    schedule[day] = 'Recovery Ride'
        else:
            quality = []
            if phase == 'base':
                patterns = [
                    ['Tempo Ride', 'Sweet Spot'], ['Sweet Spot', 'Endurance Ride'],
                    ['Tempo Ride', 'Hill Ride'], ['Sweet Spot', 'Tempo Ride'],
                ]
            elif phase == 'build':
                patterns = [
                    ['Threshold Intervals', 'Sweet Spot'], ['VO2max Intervals', 'Tempo Ride'],
                    ['Sweet Spot', 'Threshold Intervals'], ['Threshold Intervals', 'VO2max Intervals'],
                ]
            elif phase == 'peak':
                patterns = [
                    ['VO2max Intervals', 'Threshold Intervals'], ['Threshold Intervals', 'VO2max Intervals'],
                    ['VO2max Intervals', 'Sweet Spot'],
                ]
            else:
                patterns = [['Tempo Ride'], ['Sweet Spot']]

            pattern = patterns[(week_num - 1) % len(patterns)]
            quality = [w for w in pattern if self.preferred_workouts.get(
                w.lower().replace(' ', '_').replace('ride', '').replace('intervals', '').strip('_'), True
            )]
            if not quality:
                quality = pattern[:1]

            qi = 0
            for i, day in enumerate(day_slots):
                if day == 'Saturday':
                    schedule[day] = 'Long Ride'
                elif day == 'Sunday':
                    schedule[day] = 'Endurance Ride'
                elif i == 1 and quality:
                    schedule[day] = quality[0]
                    qi = 1
                elif i == 3 and len(quality) > 1:
                    schedule[day] = quality[1]
                else:
                    schedule[day] = 'Endurance Ride'

        return schedule

    def _calculate_workout_duration(self, workout_type: str, remaining: float, total: float, phase: str) -> float:
        """Calcula duración en horas para un entrenamiento."""
        pcts = {
            'Long Ride': 0.35, 'Endurance Ride': 0.18, 'Recovery Ride': 0.10,
            'Tempo Ride': 0.15, 'Sweet Spot': 0.15, 'Threshold Intervals': 0.12,
            'VO2max Intervals': 0.10, 'Hill Ride': 0.15
        }
        dur = total * pcts.get(workout_type, 0.15)
        if workout_type == 'Long Ride':
            dur = min(dur, self.long_ride_max_hours, remaining * 0.5)
        mins = {
            'Long Ride': 1.5, 'Endurance Ride': 1.0, 'Recovery Ride': 0.5,
            'Tempo Ride': 1.0, 'Sweet Spot': 1.0, 'Threshold Intervals': 0.75,
            'VO2max Intervals': 0.75, 'Hill Ride': 1.0
        }
        return max(mins.get(workout_type, 0.75), min(remaining, dur))

    def _build_workout(self, workout_type: str, duration_min: float, phase: str, week_num: int) -> dict:
        """Construye un entrenamiento detallado con segmentos."""
        builders = {
            'Endurance Ride': self._build_endurance_ride,
            'Recovery Ride': self._build_recovery_ride,
            'Long Ride': self._build_long_ride,
            'Tempo Ride': self._build_tempo_ride,
            'Sweet Spot': self._build_sweet_spot,
            'Threshold Intervals': self._build_threshold_intervals,
            'VO2max Intervals': self._build_vo2max_intervals,
            'Hill Ride': self._build_hill_ride,
        }
        builder = builders.get(workout_type, self._build_endurance_ride)
        return builder(duration_min, phase, week_num)

    def _fmt_power(self, watts: int) -> str:
        return f"{watts}W"

    def _build_endurance_ride(self, duration_min: float, phase: str, week_num: int) -> dict:
        z2 = self.power_zones.get('zone_2', {})
        return {
            'type_es': 'Rodaje Resistencia',
            'description': f'Rodaje de resistencia a Z2. Mantén cadencia 85-95 rpm.',
            'structure': f'{duration_min:.0f}min en Z2',
            'segments': [{
                'name': 'Rodaje principal',
                'distance_km': None,
                'duration_min': duration_min,
                'pace': f"{z2.get('min_watts', self.ftp*0.56):.0f}-{z2.get('max_watts', self.ftp*0.75):.0f}W",
                'pace_range': f"Z2: {z2.get('min_pct', 56)}-{z2.get('max_pct', 75)}% FTP",
                'hr_min': '', 'hr_max': '',
                'zone': 2,
                'notes': 'Cadencia 85-95 rpm, esfuerzo conversacional'
            }]
        }

    def _build_recovery_ride(self, duration_min: float, phase: str, week_num: int) -> dict:
        z1 = self.power_zones.get('zone_1', {})
        return {
            'type_es': 'Recuperación Activa',
            'description': 'Pedaleo suave de recuperación. No forzar.',
            'structure': f'{duration_min:.0f}min muy suave Z1',
            'segments': [{
                'name': 'Recuperación activa',
                'distance_km': None,
                'duration_min': duration_min,
                'pace': f"<{z1.get('max_watts', self.ftp*0.55):.0f}W",
                'pace_range': f"Z1: <{z1.get('max_pct', 55)}% FTP",
                'hr_min': '', 'hr_max': '',
                'zone': 1,
                'notes': 'Cadencia alta >90 rpm, desarrollo suave'
            }]
        }

    def _build_long_ride(self, duration_min: float, phase: str, week_num: int) -> dict:
        z2 = self.power_zones.get('zone_2', {})
        z3 = self.power_zones.get('zone_3', {})
        warmup = min(20, duration_min * 0.1)
        cooldown = min(15, duration_min * 0.08)
        main_time = duration_min - warmup - cooldown

        segments = [
            {
                'name': 'Calentamiento',
                'distance_km': None, 'duration_min': warmup,
                'pace': f"<{z2.get('min_watts', self.ftp*0.56):.0f}W",
                'pace_range': 'Z1-Z2',
                'hr_min': '', 'hr_max': '', 'zone': 1,
                'notes': 'Empezar suave, ir subiendo progresivamente'
            }
        ]

        if phase in ('build', 'peak'):
            tempo_block = min(30, main_time * 0.2)
            easy_block = main_time - tempo_block
            segments.append({
                'name': 'Bloque aeróbico',
                'distance_km': None, 'duration_min': easy_block,
                'pace': f"{z2.get('min_watts', self.ftp*0.56):.0f}-{z2.get('max_watts', self.ftp*0.75):.0f}W",
                'pace_range': f"Z2: {z2.get('min_pct', 56)}-{z2.get('max_pct', 75)}% FTP",
                'hr_min': '', 'hr_max': '', 'zone': 2,
                'notes': 'Ritmo constante y sostenible'
            })
            segments.append({
                'name': 'Progresión tempo',
                'distance_km': None, 'duration_min': tempo_block,
                'pace': f"{z3.get('min_watts', self.ftp*0.76):.0f}-{z3.get('max_watts', self.ftp*0.90):.0f}W",
                'pace_range': f"Z3: {z3.get('min_pct', 76)}-{z3.get('max_pct', 90)}% FTP",
                'hr_min': '', 'hr_max': '', 'zone': 3,
                'notes': 'Subir ritmo progresivamente'
            })
        else:
            segments.append({
                'name': 'Bloque principal',
                'distance_km': None, 'duration_min': main_time,
                'pace': f"{z2.get('min_watts', self.ftp*0.56):.0f}-{z2.get('max_watts', self.ftp*0.75):.0f}W",
                'pace_range': f"Z2: {z2.get('min_pct', 56)}-{z2.get('max_pct', 75)}% FTP",
                'hr_min': '', 'hr_max': '', 'zone': 2,
                'notes': 'Constante, practicar nutrición si >2h'
            })

        segments.append({
            'name': 'Vuelta a la calma',
            'distance_km': None, 'duration_min': cooldown,
            'pace': f"<{z2.get('min_watts', self.ftp*0.56):.0f}W",
            'pace_range': 'Z1', 'hr_min': '', 'hr_max': '', 'zone': 1,
            'notes': 'Pedaleo suave, estiramientos'
        })

        # Estimar distancia y desnivel para la descripción
        est_distance = round(duration_min / 60 * self.avg_speed_kmh, 1)
        long_ride_elev_share = self._ELEVATION_DISTRIBUTION.get('Long Ride', 0.35)
        est_elevation = round(self.weekly_elevation_gain * long_ride_elev_share)
        elev_note = f' Incluye ~{est_elevation}m de desnivel.' if est_elevation > 100 else ''

        return {
            'type_es': 'Fondo Largo',
            'description': f'Rodaje largo de {duration_min:.0f}min (~{est_distance}km) para desarrollar resistencia.{elev_note}',
            'structure': f'Calentamiento + {main_time:.0f}min Z2 + Vuelta calma',
            'segments': segments
        }

    def _build_tempo_ride(self, duration_min: float, phase: str, week_num: int) -> dict:
        z3 = self.power_zones.get('zone_3', {})
        warmup, cooldown = 15, 10
        tempo_time = duration_min - warmup - cooldown
        return {
            'type_es': 'Rodaje Tempo',
            'description': f'Entrenamiento de tempo a Z3 durante {tempo_time:.0f}min.',
            'structure': f'15min calent. + {tempo_time:.0f}min Z3 + 10min vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': warmup,
                 'pace': 'Z1-Z2', 'pace_range': 'Progresivo', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Incluir 3-4 aceleraciones de 10s'},
                {'name': 'Bloque Tempo', 'distance_km': None, 'duration_min': tempo_time,
                 'pace': f"{z3.get('min_watts', self.ftp*0.76):.0f}-{z3.get('max_watts', self.ftp*0.90):.0f}W",
                 'pace_range': f"Z3: {z3.get('min_pct', 76)}-{z3.get('max_pct', 90)}% FTP",
                 'hr_min': '', 'hr_max': '', 'zone': 3,
                 'notes': 'Cadencia 85-95 rpm, esfuerzo sostenido'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': cooldown,
                 'pace': 'Z1', 'pace_range': '<55% FTP', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Pedaleo muy suave'}
            ]
        }

    def _build_sweet_spot(self, duration_min: float, phase: str, week_num: int) -> dict:
        ss_min = int(self.ftp * 0.88)
        ss_max = int(self.ftp * 0.93)
        warmup, cooldown = 15, 10
        main_time = duration_min - warmup - cooldown

        if phase == 'base':
            reps = max(2, int(main_time / 12))
            on_time = int(main_time * 0.7 / reps)
            off_time = int(main_time * 0.3 / reps)
        else:
            reps = max(2, int(main_time / 15))
            on_time = int(main_time * 0.8 / reps)
            off_time = int(main_time * 0.2 / reps)

        return {
            'type_es': 'Sweet Spot',
            'description': f'{reps}x{on_time}min a 88-93% FTP con {off_time}min recuperación.',
            'structure': f'Calent. + {reps}x{on_time}min SS + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': warmup,
                 'pace': 'Z1-Z2', 'pace_range': 'Progresivo', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Incluir 2-3 aceleraciones cortas'},
                {'name': f'{reps}x{on_time}min Sweet Spot', 'distance_km': None,
                 'duration_min': reps * (on_time + off_time),
                 'pace': f"{ss_min}-{ss_max}W",
                 'pace_range': '88-93% FTP',
                 'hr_min': '', 'hr_max': '', 'zone': 3,
                 'reps': reps, 'rep_duration': on_time * 60,
                 'rest_after': f'{off_time}min Z1',
                 'notes': 'Cadencia 85-95 rpm, constante'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': cooldown,
                 'pace': 'Z1', 'pace_range': '<55% FTP', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Pedaleo suave'}
            ]
        }

    def _build_threshold_intervals(self, duration_min: float, phase: str, week_num: int) -> dict:
        z4 = self.power_zones.get('zone_4', {})
        warmup, cooldown = 15, 10
        main_time = duration_min - warmup - cooldown

        if phase in ('base', 'build'):
            reps = max(2, int(main_time / 12))
            on_time = 8
            off_time = 4
        else:
            reps = max(2, int(main_time / 15))
            on_time = 10
            off_time = 5

        return {
            'type_es': 'Intervalos Umbral',
            'description': f'{reps}x{on_time}min a 95-105% FTP.',
            'structure': f'Calent. + {reps}x{on_time}min Z4 + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': warmup,
                 'pace': 'Z1-Z2', 'pace_range': 'Progresivo', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Incluir 3 aceleraciones de 15s'},
                {'name': f'{reps}x{on_time}min Umbral', 'distance_km': None,
                 'duration_min': reps * (on_time + off_time),
                 'pace': f"{z4.get('min_watts', self.ftp*0.91):.0f}-{z4.get('max_watts', self.ftp*1.05):.0f}W",
                 'pace_range': f"Z4: {z4.get('min_pct', 91)}-{z4.get('max_pct', 105)}% FTP",
                 'hr_min': '', 'hr_max': '', 'zone': 4,
                 'reps': reps, 'rep_duration': on_time * 60,
                 'rest_after': f'{off_time}min Z1',
                 'notes': 'Cadencia 85-95 rpm, constante'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': cooldown,
                 'pace': 'Z1', 'pace_range': '<55% FTP', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Pedaleo suave'}
            ]
        }

    def _build_vo2max_intervals(self, duration_min: float, phase: str, week_num: int) -> dict:
        z5 = self.power_zones.get('zone_5', {})
        warmup, cooldown = 15, 10
        main_time = duration_min - warmup - cooldown
        on_time = 4  # min
        off_time = 4  # min
        reps = max(3, int(main_time / (on_time + off_time)))

        return {
            'type_es': 'Intervalos VO2max',
            'description': f'{reps}x{on_time}min a 106-120% FTP con {off_time}min recuperación.',
            'structure': f'Calent. + {reps}x{on_time}min Z5 + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': warmup,
                 'pace': 'Z1-Z2', 'pace_range': 'Progresivo', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Incluir 3 sprints cortos de 10s'},
                {'name': f'{reps}x{on_time}min VO2max', 'distance_km': None,
                 'duration_min': reps * (on_time + off_time),
                 'pace': f"{z5.get('min_watts', self.ftp*1.06):.0f}-{z5.get('max_watts', self.ftp*1.20):.0f}W",
                 'pace_range': f"Z5: {z5.get('min_pct', 106)}-{z5.get('max_pct', 120)}% FTP",
                 'hr_min': '', 'hr_max': '', 'zone': 5,
                 'reps': reps, 'rep_duration': on_time * 60,
                 'rest_after': f'{off_time}min Z1 pedaleo suave',
                 'notes': 'Cadencia alta >95 rpm, máximo esfuerzo sostenible'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': cooldown,
                 'pace': 'Z1', 'pace_range': '<55% FTP', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Pedaleo suave, hidratación'}
            ]
        }

    def _build_hill_ride(self, duration_min: float, phase: str, week_num: int) -> dict:
        z4 = self.power_zones.get('zone_4', {})
        warmup, cooldown = 15, 10
        main_time = duration_min - warmup - cooldown
        reps = max(4, int(main_time / 8))
        climb_time = 3  # min
        descent_time = 3  # min

        # Estimar desnivel de esta sesión
        hill_elev_share = self._ELEVATION_DISTRIBUTION.get('Hill Ride', 0.40)
        est_elevation = round(self.weekly_elevation_gain * hill_elev_share)
        elev_per_rep = round(est_elevation / max(1, reps))
        elev_note = f' (~{est_elevation}m desnivel total, ~{elev_per_rep}m/rep)' if est_elevation > 50 else ''

        return {
            'type_es': 'Rodaje Subidas',
            'description': f'{reps} repeticiones de subida a Z4-Z5.{elev_note}',
            'structure': f'Calent. + {reps}x{climb_time}min subida + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': warmup,
                 'pace': 'Z1-Z2', 'pace_range': 'Progresivo', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Llegar a la subida caliente'},
                {'name': f'{reps}x{climb_time}min subida', 'distance_km': None,
                 'duration_min': reps * (climb_time + descent_time),
                 'pace': f"{z4.get('min_watts', self.ftp*0.91):.0f}-{self.ftp*1.10:.0f}W",
                 'pace_range': 'Z4-Z5: 91-110% FTP',
                 'hr_min': '', 'hr_max': '', 'zone': 4,
                 'reps': reps, 'rep_duration': climb_time * 60,
                 'rest_after': f'{descent_time}min descenso Z1',
                 'notes': 'Cadencia 70-80 rpm sentado, 60-70 rpm de pie alternando'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': cooldown,
                 'pace': 'Z1', 'pace_range': '<55% FTP', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Pedaleo suave en llano'}
            ]
        }

    def _get_zone_for_cycling_workout(self, workout_type: str) -> dict:
        """Devuelve la zona principal para un tipo de entrenamiento de ciclismo."""
        zone_map = {
            'Endurance Ride': 2, 'Recovery Ride': 1, 'Long Ride': 2,
            'Tempo Ride': 3, 'Sweet Spot': 3, 'Threshold Intervals': 4,
            'VO2max Intervals': 5, 'Hill Ride': 4
        }
        zone_num = zone_map.get(workout_type, 2)
        zone_key = f'zone_{zone_num}'
        hr_zone = self.training_zones.get(zone_key, {})
        return {
            'zone_num': zone_num,
            'min_hr': hr_zone.get('min_hr', ''),
            'max_hr': hr_zone.get('max_hr', '')
        }
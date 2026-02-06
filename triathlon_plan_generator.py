"""
Triathlon Plan Generator - Domin Cideam.es
Genera planes de entrenamiento de triatlón periodizados con natación, ciclismo y carrera.
Incluye entrenamientos brick, transiciones y zonas específicas por disciplina.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math


class TriathlonPlanGenerator:
    """Genera planes de entrenamiento de triatlón personalizados con detalle completo."""

    # Distancias por modalidad de triatlón (swim km, bike km, run km)
    TRIATHLON_DISTANCES = {
        'sprint': {'swim': 0.75, 'bike': 20, 'run': 5},
        'olimpico': {'swim': 1.5, 'bike': 40, 'run': 10},
        'medio_ironman': {'swim': 1.9, 'bike': 90, 'run': 21.1},
        'ironman': {'swim': 3.8, 'bike': 180, 'run': 42.2},
    }

    def __init__(
        self,
        target_event: str,
        target_date: datetime,
        current_fitness: dict,
        training_zones: dict,
        ftp: int = 200,
        power_zones: dict = None,
        swim_zones: dict = None,
        css_pace_100m: float = 105,
        run_paces: dict = None,
        intensity_preference: str = "moderate",
        preferred_workouts: dict = None,
        training_days_per_week: int = 5
    ):
        """
        Inicializa el generador de planes de triatlón.

        Args:
            target_event: Tipo de triatlón ('sprint', 'olimpico', 'medio_ironman', 'ironman')
            target_date: Fecha objetivo del evento
            current_fitness: Diccionario con métricas de forma física actual
            training_zones: Diccionario con zonas de FC
            ftp: FTP actual en watts
            power_zones: Diccionario con zonas de potencia ciclismo
            swim_zones: Diccionario con zonas de natación
            css_pace_100m: CSS pace en segundos por 100m
            run_paces: Diccionario con ritmos de carrera (min/km)
            intensity_preference: "easy", "moderate", o "high"
            preferred_workouts: Diccionario de preferencias
            training_days_per_week: Días de entrenamiento (4-7)
        """
        self.target_event = target_event
        self.target_date = target_date
        self.current_fitness = current_fitness
        self.training_zones = training_zones
        self.ftp = ftp
        self.power_zones = power_zones or {}
        self.swim_zones = swim_zones or {}
        self.css_pace_100m = css_pace_100m
        self.run_paces = run_paces or {
            'recuperacion': 7.0, 'suave': 6.5, 'aerobico': 6.0,
            'umbral': 5.3, '10k': 5.0, '5k': 4.8
        }
        self.intensity_preference = intensity_preference
        self.preferred_workouts = preferred_workouts or {}
        self.training_days_per_week = min(7, max(4, training_days_per_week))

        today = datetime.now().date()
        td = target_date.date() if isinstance(target_date, datetime) else target_date
        self.weeks_to_event = max(4, (td - today).days // 7)

        self.event_distances = self.TRIATHLON_DISTANCES.get(target_event, self.TRIATHLON_DISTANCES['olimpico'])
        self._set_training_parameters()

    def _set_training_parameters(self):
        """Establece parámetros de entrenamiento según evento."""
        event_params = {
            'sprint': {'peak_hours': 8, 'swim_pct': 0.20, 'bike_pct': 0.45, 'run_pct': 0.35},
            'olimpico': {'peak_hours': 10, 'swim_pct': 0.20, 'bike_pct': 0.40, 'run_pct': 0.40},
            'medio_ironman': {'peak_hours': 14, 'swim_pct': 0.15, 'bike_pct': 0.45, 'run_pct': 0.40},
            'ironman': {'peak_hours': 20, 'swim_pct': 0.15, 'bike_pct': 0.50, 'run_pct': 0.35},
        }
        params = event_params.get(self.target_event, event_params['olimpico'])
        self.peak_weekly_hours = params['peak_hours']
        self.swim_pct = params['swim_pct']
        self.bike_pct = params['bike_pct']
        self.run_pct = params['run_pct']

        base_hours = self.current_fitness.get('avg_weekly_hours', 6)
        self.base_weekly_hours = base_hours

        if self.intensity_preference == "easy":
            self.easy_pct = 0.80
        elif self.intensity_preference == "high":
            self.easy_pct = 0.60
        else:
            self.easy_pct = 0.70

    def _calculate_phases(self) -> list:
        """Calcula las fases de periodización para triatlón."""
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
        if phase == 'taper':
            return False
        return week_num % 4 == 0

    def _format_pace(self, pace_min_per_km: float) -> str:
        """Formatea el ritmo en min:seg/km."""
        minutes = int(pace_min_per_km)
        seconds = int((pace_min_per_km - minutes) * 60)
        return f"{minutes}:{seconds:02d}/km"

    def _format_swim_pace(self, sec_per_100m: float) -> str:
        """Formatea el ritmo de natación en min:seg/100m."""
        minutes = int(sec_per_100m) // 60
        seconds = int(sec_per_100m) % 60
        return f"{minutes}:{seconds:02d}/100m"

    def generate_plan(self) -> dict:
        """Genera el plan de entrenamiento completo de triatlón."""
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

                week_workouts = self._generate_week_workouts(week_num, target_hours, phase, is_recovery)

                # Calcular distancia total estimada
                total_dist = sum(
                    wo.get('distance', 0) for wo in week_workouts.values() if wo.get('type') != 'Rest'
                )

                weeks.append({
                    'week_number': week_num,
                    'phase': phase,
                    'is_recovery': is_recovery,
                    'target_hours': target_hours,
                    'total_distance': round(total_dist, 1),
                    'start_date': week_start_date,
                    'workouts': week_workouts
                })
                week_start_date += timedelta(days=7)

        return {
            'weeks': weeks,
            'total_weeks': week_num,
            'target_event': self.target_event,
            'target_date': self.target_date,
            'event_distances': self.event_distances,
            'peak_week_distance': round(self.peak_weekly_hours * 20, 1),
            'sport_type': 'triathlon',
            'phases': phases
        }

    def _generate_week_workouts(self, week_num: int, target_hours: float, phase: str, is_recovery: bool) -> dict:
        """Genera entrenamientos de triatlón para una semana."""
        workouts = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        schedule = self._create_workout_schedule(phase, is_recovery, week_num)

        swim_hours = target_hours * self.swim_pct
        bike_hours = target_hours * self.bike_pct
        run_hours = target_hours * self.run_pct

        for day in days:
            workout_type = schedule.get(day, 'Rest')
            if workout_type == 'Rest':
                workouts[day] = {
                    'type': 'Rest', 'type_es': 'Descanso', 'distance': 0,
                    'zone': '', 'hr_min': '', 'hr_max': '',
                    'description': 'Día de descanso completo',
                    'structure': None, 'segments': [], 'sport': 'rest'
                }
                continue

            sport = self._get_sport_for_workout(workout_type)
            if sport == 'swim':
                dur_min = max(30, swim_hours * 60 / max(1, self._count_sport_sessions(schedule, 'swim')))
            elif sport == 'bike':
                dur_min = max(45, bike_hours * 60 / max(1, self._count_sport_sessions(schedule, 'bike')))
            elif sport == 'run':
                dur_min = max(30, run_hours * 60 / max(1, self._count_sport_sessions(schedule, 'run')))
            else:  # brick
                dur_min = max(75, (bike_hours * 0.3 + run_hours * 0.2) * 60)

            detailed = self._build_workout(workout_type, dur_min, phase, week_num)
            distance = self._estimate_distance(workout_type, dur_min, sport)

            workouts[day] = {
                'type': workout_type,
                'type_es': detailed.get('type_es', workout_type),
                'distance': round(distance, 1),
                'zone': detailed.get('zone', 2),
                'hr_min': '', 'hr_max': '',
                'description': detailed['description'],
                'structure': detailed['structure'],
                'segments': detailed['segments'],
                'sport': sport
            }

        return workouts

    def _get_sport_for_workout(self, workout_type: str) -> str:
        """Devuelve el deporte principal del tipo de entrenamiento."""
        sport_map = {
            'Easy Swim': 'swim', 'Threshold Swim': 'swim', 'Interval Swim': 'swim',
            'Open Water Swim': 'swim',
            'Endurance Ride': 'bike', 'Tempo Ride': 'bike', 'Long Ride': 'bike',
            'Sweet Spot Ride': 'bike',
            'Easy Run': 'run', 'Tempo Run': 'run', 'Long Run': 'run',
            'Intervals Run': 'run',
            'Brick Workout': 'brick', 'Transition Practice': 'brick',
        }
        return sport_map.get(workout_type, 'run')

    def _count_sport_sessions(self, schedule: dict, sport: str) -> int:
        """Cuenta sesiones de un deporte en la semana."""
        count = 0
        for wtype in schedule.values():
            s = self._get_sport_for_workout(wtype)
            if s == sport or (sport == 'bike' and s == 'brick') or (sport == 'run' and s == 'brick'):
                count += 1
        return count

    def _estimate_distance(self, workout_type: str, dur_min: float, sport: str) -> float:
        """Estima distancia según deporte y duración."""
        if sport == 'swim':
            return dur_min * 0.025  # ~2.5km/100min
        elif sport == 'bike':
            return dur_min * 0.47  # ~28 km/h
        elif sport == 'run':
            return dur_min / 6.0  # ~6 min/km promedio
        else:  # brick
            return dur_min * 0.30  # promedio mixto


    def _create_workout_schedule(self, phase: str, is_recovery: bool, week_num: int) -> dict:
        """Crea el horario semanal de entrenamientos multi-deporte."""
        schedule = {}

        if is_recovery:
            if self.training_days_per_week >= 6:
                schedule = {
                    'Monday': 'Easy Swim', 'Tuesday': 'Easy Run',
                    'Thursday': 'Endurance Ride', 'Saturday': 'Easy Swim'
                }
            elif self.training_days_per_week >= 5:
                schedule = {
                    'Monday': 'Easy Swim', 'Tuesday': 'Easy Run',
                    'Thursday': 'Endurance Ride', 'Saturday': 'Easy Swim'
                }
            else:
                schedule = {
                    'Tuesday': 'Easy Swim', 'Thursday': 'Easy Run',
                    'Saturday': 'Endurance Ride'
                }
            return schedule

        # Patrones normales por fase
        if phase == 'base':
            patterns = [
                {'Monday': 'Easy Swim', 'Tuesday': 'Easy Run', 'Wednesday': 'Endurance Ride',
                 'Thursday': 'Threshold Swim', 'Saturday': 'Long Ride', 'Sunday': 'Long Run'},
                {'Monday': 'Threshold Swim', 'Tuesday': 'Tempo Run', 'Wednesday': 'Endurance Ride',
                 'Thursday': 'Easy Swim', 'Saturday': 'Long Ride', 'Sunday': 'Easy Run'},
            ]
        elif phase == 'build':
            patterns = [
                {'Monday': 'Interval Swim', 'Tuesday': 'Intervals Run', 'Wednesday': 'Tempo Ride',
                 'Thursday': 'Easy Swim', 'Friday': 'Easy Run', 'Saturday': 'Brick Workout', 'Sunday': 'Long Run'},
                {'Monday': 'Threshold Swim', 'Tuesday': 'Tempo Run', 'Wednesday': 'Sweet Spot Ride',
                 'Thursday': 'Interval Swim', 'Saturday': 'Long Ride', 'Sunday': 'Brick Workout'},
            ]
        elif phase == 'peak':
            patterns = [
                {'Monday': 'Interval Swim', 'Tuesday': 'Intervals Run', 'Wednesday': 'Tempo Ride',
                 'Thursday': 'Open Water Swim', 'Saturday': 'Brick Workout', 'Sunday': 'Long Run'},
                {'Monday': 'Open Water Swim', 'Tuesday': 'Tempo Run', 'Wednesday': 'Sweet Spot Ride',
                 'Thursday': 'Interval Swim', 'Saturday': 'Brick Workout', 'Sunday': 'Easy Run'},
            ]
        else:  # taper
            patterns = [
                {'Monday': 'Easy Swim', 'Wednesday': 'Easy Run', 'Friday': 'Endurance Ride',
                 'Saturday': 'Transition Practice'},
                {'Tuesday': 'Easy Swim', 'Thursday': 'Easy Run', 'Saturday': 'Endurance Ride'},
            ]

        pattern = patterns[(week_num - 1) % len(patterns)]

        # Filtrar según días disponibles
        if self.training_days_per_week < 6:
            # Priorizar: mantener al menos 1 swim, 1-2 bike, 1-2 run, brick si es build/peak
            priority_days = ['Saturday', 'Sunday', 'Tuesday', 'Thursday', 'Monday', 'Wednesday', 'Friday']
            selected = {}
            count = 0
            for day in priority_days:
                if day in pattern and count < self.training_days_per_week:
                    selected[day] = pattern[day]
                    count += 1
            return selected

        return pattern

    def _build_workout(self, workout_type: str, duration_min: float, phase: str, week_num: int) -> dict:
        """Construye un entrenamiento detallado con segmentos."""
        builders = {
            'Easy Swim': self._build_easy_swim,
            'Threshold Swim': self._build_threshold_swim,
            'Interval Swim': self._build_interval_swim,
            'Open Water Swim': self._build_open_water_swim,
            'Endurance Ride': self._build_endurance_ride,
            'Tempo Ride': self._build_tempo_ride,
            'Long Ride': self._build_long_ride,
            'Sweet Spot Ride': self._build_sweet_spot_ride,
            'Easy Run': self._build_easy_run,
            'Tempo Run': self._build_tempo_run,
            'Long Run': self._build_long_run,
            'Intervals Run': self._build_intervals_run,
            'Brick Workout': self._build_brick_workout,
            'Transition Practice': self._build_transition_practice,
        }
        builder = builders.get(workout_type, self._build_easy_run)
        return builder(duration_min, phase, week_num)


    # ===================== SWIM WORKOUTS =====================

    def _build_easy_swim(self, duration_min: float, phase: str, week_num: int) -> dict:
        css = self.css_pace_100m
        return {
            'type_es': 'Natación Suave',
            'zone': 2,
            'description': f'Nado continuo a ritmo suave enfocado en técnica.',
            'structure': f'{duration_min:.0f}min nado Z2',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': 10,
                 'pace': self._format_swim_pace(css * 1.25), 'pace_range': 'Z1-Z2',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': '200m suave + 4x50m técnica'},
                {'name': 'Bloque principal', 'distance_km': None, 'duration_min': duration_min - 15,
                 'pace': self._format_swim_pace(css * 1.15), 'pace_range': 'Z2',
                 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Nado continuo, enfocarse en brazada larga y eficiente'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': 5,
                 'pace': self._format_swim_pace(css * 1.30), 'pace_range': 'Z1',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': '200m suave + estiramientos'}
            ]
        }

    def _build_threshold_swim(self, duration_min: float, phase: str, week_num: int) -> dict:
        css = self.css_pace_100m
        main_time = duration_min - 15
        reps = max(3, int(main_time / 5))
        on_time = int(main_time * 0.7 / reps)
        off_time = int(main_time * 0.3 / reps)
        return {
            'type_es': 'Natación Umbral',
            'zone': 4,
            'description': f'{reps}x{on_time*60//60}min a ritmo CSS con {off_time*60//60}min recuperación.',
            'structure': f'Calent. + {reps}x a CSS + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': 10,
                 'pace': self._format_swim_pace(css * 1.20), 'pace_range': 'Z1-Z2',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': '300m suave + 4x50m progresivos'},
                {'name': f'{reps}x rep. umbral', 'distance_km': None,
                 'duration_min': reps * (on_time + off_time),
                 'pace': self._format_swim_pace(css), 'pace_range': 'Z4: CSS',
                 'hr_min': '', 'hr_max': '', 'zone': 4,
                 'reps': reps, 'rep_duration': on_time * 60,
                 'rest_after': f'{off_time}min suave',
                 'notes': 'Mantener ritmo CSS constante'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': 5,
                 'pace': self._format_swim_pace(css * 1.30), 'pace_range': 'Z1',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Nado suave, técnica'}
            ]
        }

    def _build_interval_swim(self, duration_min: float, phase: str, week_num: int) -> dict:
        css = self.css_pace_100m
        main_time = duration_min - 15
        reps = max(4, int(main_time / 4))
        on_time = 2  # min
        off_time = 1  # min
        return {
            'type_es': 'Natación Intervalos',
            'zone': 5,
            'description': f'{reps}x{on_time}min rápido a ritmo VO2max con {off_time}min rec.',
            'structure': f'Calent. + {reps}x{on_time}min Z5 + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': 10,
                 'pace': self._format_swim_pace(css * 1.20), 'pace_range': 'Z1-Z2',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': '300m suave + 4x50m progresivos + 100m patada'},
                {'name': f'{reps}x{on_time}min VO2max', 'distance_km': None,
                 'duration_min': reps * (on_time + off_time),
                 'pace': self._format_swim_pace(css * 0.92), 'pace_range': 'Z5: VO2max',
                 'hr_min': '', 'hr_max': '', 'zone': 5,
                 'reps': reps, 'rep_duration': on_time * 60,
                 'rest_after': f'{off_time}min suave',
                 'notes': 'Esfuerzo alto, mantener técnica'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': 5,
                 'pace': self._format_swim_pace(css * 1.30), 'pace_range': 'Z1',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Nado suave regenerativo'}
            ]
        }

    def _build_open_water_swim(self, duration_min: float, phase: str, week_num: int) -> dict:
        css = self.css_pace_100m
        return {
            'type_es': 'Natación Aguas Abiertas',
            'zone': 2,
            'description': f'Nado en aguas abiertas para practicar orientación y neopreno.',
            'structure': f'{duration_min:.0f}min nado aguas abiertas',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': 5,
                 'pace': self._format_swim_pace(css * 1.20), 'pace_range': 'Z1-Z2',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Adaptación al agua, respiración bilateral'},
                {'name': 'Nado principal', 'distance_km': None, 'duration_min': duration_min - 10,
                 'pace': self._format_swim_pace(css * 1.10), 'pace_range': 'Z2-Z3',
                 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Practicar sighting cada 6-8 brazadas, drafting si hay compañeros'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': 5,
                 'pace': self._format_swim_pace(css * 1.25), 'pace_range': 'Z1',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Nado relajado'}
            ]
        }

    # ===================== BIKE WORKOUTS =====================

    def _build_endurance_ride(self, duration_min: float, phase: str, week_num: int) -> dict:
        z2 = self.power_zones.get('zone_2', {})
        return {
            'type_es': 'Rodaje Resistencia',
            'zone': 2,
            'description': f'Rodaje de resistencia a Z2 en bici.',
            'structure': f'{duration_min:.0f}min en Z2',
            'segments': [{'name': 'Rodaje principal', 'distance_km': None,
                 'duration_min': duration_min,
                 'pace': f"{z2.get('min_watts', int(self.ftp*0.56))}-{z2.get('max_watts', int(self.ftp*0.75))}W",
                 'pace_range': 'Z2: 56-75% FTP', 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Cadencia 85-95 rpm, esfuerzo conversacional'}]
        }

    def _build_tempo_ride(self, duration_min: float, phase: str, week_num: int) -> dict:
        z3 = self.power_zones.get('zone_3', {})
        warmup, cooldown = 15, 10
        tempo_time = duration_min - warmup - cooldown
        return {
            'type_es': 'Rodaje Tempo',
            'zone': 3,
            'description': f'Tempo a Z3 durante {tempo_time:.0f}min en bici.',
            'structure': f'15min calent. + {tempo_time:.0f}min Z3 + 10min vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': warmup,
                 'pace': 'Z1-Z2', 'pace_range': 'Progresivo', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Incluir 3 aceleraciones de 10s'},
                {'name': 'Bloque Tempo', 'distance_km': None, 'duration_min': tempo_time,
                 'pace': f"{z3.get('min_watts', int(self.ftp*0.76))}-{z3.get('max_watts', int(self.ftp*0.90))}W",
                 'pace_range': 'Z3: 76-90% FTP', 'hr_min': '', 'hr_max': '', 'zone': 3,
                 'notes': 'Cadencia 85-95 rpm, esfuerzo sostenido'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': cooldown,
                 'pace': 'Z1', 'pace_range': '<55% FTP', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Pedaleo suave'}
            ]
        }

    def _build_long_ride(self, duration_min: float, phase: str, week_num: int) -> dict:
        z2 = self.power_zones.get('zone_2', {})
        warmup = min(15, duration_min * 0.08)
        cooldown = min(10, duration_min * 0.06)
        main_time = duration_min - warmup - cooldown
        return {
            'type_es': 'Fondo Largo Bici',
            'zone': 2,
            'description': f'Fondo largo de {duration_min:.0f}min en bici.',
            'structure': f'Calent. + {main_time:.0f}min Z2 + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': warmup,
                 'pace': 'Z1-Z2', 'pace_range': 'Progresivo', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Empezar suave'},
                {'name': 'Bloque principal', 'distance_km': None, 'duration_min': main_time,
                 'pace': f"{z2.get('min_watts', int(self.ftp*0.56))}-{z2.get('max_watts', int(self.ftp*0.75))}W",
                 'pace_range': 'Z2: 56-75% FTP', 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Practicar nutrición si >2h'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': cooldown,
                 'pace': 'Z1', 'pace_range': '<55% FTP', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Pedaleo suave'}
            ]
        }

    def _build_sweet_spot_ride(self, duration_min: float, phase: str, week_num: int) -> dict:
        ss_min = int(self.ftp * 0.88)
        ss_max = int(self.ftp * 0.93)
        warmup, cooldown = 15, 10
        main_time = duration_min - warmup - cooldown
        reps = max(2, int(main_time / 12))
        on_time = int(main_time * 0.75 / reps)
        off_time = int(main_time * 0.25 / reps)
        return {
            'type_es': 'Sweet Spot Bici',
            'zone': 3,
            'description': f'{reps}x{on_time}min a 88-93% FTP.',
            'structure': f'Calent. + {reps}x{on_time}min SS + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': None, 'duration_min': warmup,
                 'pace': 'Z1-Z2', 'pace_range': 'Progresivo', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Incluir 2-3 aceleraciones'},
                {'name': f'{reps}x{on_time}min Sweet Spot', 'distance_km': None,
                 'duration_min': reps * (on_time + off_time),
                 'pace': f'{ss_min}-{ss_max}W', 'pace_range': '88-93% FTP',
                 'hr_min': '', 'hr_max': '', 'zone': 3,
                 'reps': reps, 'rep_duration': on_time * 60,
                 'rest_after': f'{off_time}min Z1',
                 'notes': 'Cadencia 85-95 rpm'},
                {'name': 'Vuelta a la calma', 'distance_km': None, 'duration_min': cooldown,
                 'pace': 'Z1', 'pace_range': '<55% FTP', 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Pedaleo suave'}
            ]
        }

    # ===================== RUN WORKOUTS =====================

    def _build_easy_run(self, duration_min: float, phase: str, week_num: int) -> dict:
        pace = self.run_paces.get('suave', 6.5)
        distance = duration_min / pace
        return {
            'type_es': 'Rodaje Suave',
            'zone': 2,
            'description': f'Rodaje suave de {distance:.1f}km a ritmo conversacional.',
            'structure': f'{distance:.1f}km a Z2',
            'segments': [{'name': 'Rodaje principal', 'distance_km': round(distance, 1),
                 'duration_min': duration_min,
                 'pace': self._format_pace(pace), 'pace_range': f'{self._format_pace(pace)} - {self._format_pace(pace*1.05)}',
                 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Esfuerzo conversacional'}]
        }

    def _build_tempo_run(self, duration_min: float, phase: str, week_num: int) -> dict:
        pace_umbral = self.run_paces.get('umbral', 5.3)
        pace_suave = self.run_paces.get('suave', 6.5)
        warmup, cooldown = 12, 8
        tempo_time = duration_min - warmup - cooldown
        tempo_dist = tempo_time / pace_umbral
        return {
            'type_es': 'Tempo Carrera',
            'zone': 3,
            'description': f'Entrenamiento de umbral con {tempo_dist:.1f}km a ritmo tempo.',
            'structure': f'Calent. + {tempo_time:.0f}min tempo + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': round(warmup / pace_suave, 1),
                 'duration_min': warmup,
                 'pace': self._format_pace(pace_suave), 'pace_range': 'Z2',
                 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Incluir 4 progresiones de 100m'},
                {'name': 'Bloque Tempo', 'distance_km': round(tempo_dist, 1),
                 'duration_min': tempo_time,
                 'pace': self._format_pace(pace_umbral), 'pace_range': f'Z3-Z4: {self._format_pace(pace_umbral)}',
                 'hr_min': '', 'hr_max': '', 'zone': 3,
                 'notes': 'Ritmo cómodamente duro'},
                {'name': 'Vuelta a la calma', 'distance_km': round(cooldown / (pace_suave * 1.1), 1),
                 'duration_min': cooldown,
                 'pace': self._format_pace(pace_suave * 1.1), 'pace_range': 'Z1',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Trote suave + estiramientos'}
            ]
        }

    def _build_long_run(self, duration_min: float, phase: str, week_num: int) -> dict:
        pace = self.run_paces.get('aerobico', 6.0)
        pace_suave = self.run_paces.get('suave', 6.5)
        warmup = 10
        cooldown = 8
        main_time = duration_min - warmup - cooldown
        main_dist = main_time / pace
        return {
            'type_es': 'Tirada Larga',
            'zone': 2,
            'description': f'Tirada larga de {(duration_min/pace):.1f}km para resistencia.',
            'structure': f'Calent. + {main_time:.0f}min Z2 + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': round(warmup / pace_suave, 1),
                 'duration_min': warmup,
                 'pace': self._format_pace(pace_suave), 'pace_range': 'Z1-Z2',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Empezar muy suave'},
                {'name': 'Bloque principal', 'distance_km': round(main_dist, 1),
                 'duration_min': main_time,
                 'pace': self._format_pace(pace), 'pace_range': f'Z2: {self._format_pace(pace)}',
                 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Ritmo constante, practicar nutrición si >60min'},
                {'name': 'Vuelta a la calma', 'distance_km': round(cooldown / (pace_suave * 1.1), 1),
                 'duration_min': cooldown,
                 'pace': self._format_pace(pace_suave * 1.1), 'pace_range': 'Z1',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Trote suave'}
            ]
        }

    def _build_intervals_run(self, duration_min: float, phase: str, week_num: int) -> dict:
        pace_5k = self.run_paces.get('5k', 4.8)
        pace_suave = self.run_paces.get('suave', 6.5)
        warmup, cooldown = 12, 8
        main_time = duration_min - warmup - cooldown

        if phase == 'base':
            reps, dist_m = 6, 400
        elif phase == 'build':
            reps, dist_m = 5, 800
        else:
            reps, dist_m = 4, 1000

        est_time_per_rep = (dist_m / 1000) * pace_5k
        rest_time = max(60, int(est_time_per_rep * 60 * 0.7))

        return {
            'type_es': 'Series Carrera',
            'zone': 5,
            'description': f'{reps}x{dist_m}m a ritmo 5K.',
            'structure': f'Calent. + {reps}x{dist_m}m + Vuelta calma',
            'segments': [
                {'name': 'Calentamiento', 'distance_km': round(warmup / pace_suave, 1),
                 'duration_min': warmup,
                 'pace': self._format_pace(pace_suave), 'pace_range': 'Z2',
                 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Trote suave + 4x80m progresivos'},
                {'name': f'{reps}x{dist_m}m', 'distance_km': round(reps * dist_m / 1000, 1),
                 'duration_min': reps * (est_time_per_rep + rest_time / 60),
                 'pace': self._format_pace(pace_5k), 'pace_range': f'Z4-Z5: {self._format_pace(pace_5k)}',
                 'hr_min': '', 'hr_max': '', 'zone': 5,
                 'reps': reps, 'rep_distance': dist_m,
                 'rest_after': f'{rest_time}s trote suave',
                 'notes': f'Ritmo 5K, recuperación {rest_time}s'},
                {'name': 'Vuelta a la calma', 'distance_km': round(cooldown / (pace_suave * 1.1), 1),
                 'duration_min': cooldown,
                 'pace': self._format_pace(pace_suave * 1.1), 'pace_range': 'Z1',
                 'hr_min': '', 'hr_max': '', 'zone': 1,
                 'notes': 'Trote suave + estiramientos'}
            ]
        }

    # ===================== BRICK / TRANSITION WORKOUTS =====================

    def _build_brick_workout(self, duration_min: float, phase: str, week_num: int) -> dict:
        """Entrenamiento brick: bici + carrera inmediata."""
        z2_bike = self.power_zones.get('zone_2', {})
        pace_suave = self.run_paces.get('suave', 6.5)
        pace_aerobico = self.run_paces.get('aerobico', 6.0)

        # Distribución: ~70% bici, ~30% carrera
        bike_time = duration_min * 0.70
        run_time = duration_min * 0.30
        transition_time = 3  # min
        run_time -= transition_time

        bike_warmup = min(10, bike_time * 0.1)
        bike_main = bike_time - bike_warmup
        run_dist = run_time / pace_suave

        segments = [
            {'name': '🚲 Calentamiento bici', 'distance_km': None, 'duration_min': bike_warmup,
             'pace': 'Z1-Z2', 'pace_range': 'Progresivo',
             'hr_min': '', 'hr_max': '', 'zone': 1,
             'notes': 'Empezar suave, subir gradualmente'},
            {'name': '🚲 Bloque principal bici', 'distance_km': None, 'duration_min': bike_main,
             'pace': f"{z2_bike.get('min_watts', int(self.ftp*0.56))}-{z2_bike.get('max_watts', int(self.ftp*0.75))}W",
             'pace_range': 'Z2-Z3: 56-90% FTP',
             'hr_min': '', 'hr_max': '', 'zone': 2,
             'notes': 'Últimos 10min subir a Z3 para simular esfuerzo pre-transición'},
        ]

        if phase in ('build', 'peak'):
            tempo_block = min(15, bike_main * 0.2)
            segments[-1]['duration_min'] = bike_main - tempo_block
            z3 = self.power_zones.get('zone_3', {})
            segments.append({
                'name': '🚲 Progresión tempo bici', 'distance_km': None, 'duration_min': tempo_block,
                'pace': f"{z3.get('min_watts', int(self.ftp*0.76))}-{z3.get('max_watts', int(self.ftp*0.90))}W",
                'pace_range': 'Z3: 76-90% FTP',
                'hr_min': '', 'hr_max': '', 'zone': 3,
                'notes': 'Subir ritmo para preparar transición'
            })

        segments.extend([
            {'name': '🔄 T2 Transición', 'distance_km': None, 'duration_min': transition_time,
             'pace': 'Máxima velocidad', 'pace_range': 'Transición',
             'hr_min': '', 'hr_max': '', 'zone': 0,
             'notes': 'Practicar: bajar bici, cambiar zapatillas, salir corriendo'},
            {'name': '🏃 Carrera post-bici', 'distance_km': round(run_dist, 1), 'duration_min': run_time,
             'pace': self._format_pace(pace_suave), 'pace_range': f'Z2: {self._format_pace(pace_suave)}',
             'hr_min': '', 'hr_max': '', 'zone': 2,
             'notes': 'Empezar suave, piernas pesadas es normal. Cadencia alta.'},
        ])

        return {
            'type_es': 'Entrenamiento Brick',
            'zone': 2,
            'description': f'Brick: {bike_time:.0f}min bici + T2 + {run_time:.0f}min carrera.',
            'structure': f'Bici {bike_time:.0f}min + T2 + Carrera {run_time:.0f}min',
            'segments': segments
        }

    def _build_transition_practice(self, duration_min: float, phase: str, week_num: int) -> dict:
        """Práctica de transiciones con mini-brick."""
        z2_bike = self.power_zones.get('zone_2', {})
        pace_suave = self.run_paces.get('suave', 6.5)
        css = self.css_pace_100m

        # Mini-brick con las 3 disciplinas
        swim_time = min(15, duration_min * 0.2)
        bike_time = min(30, duration_min * 0.4)
        run_time = min(15, duration_min * 0.25)

        return {
            'type_es': 'Práctica Transiciones',
            'zone': 2,
            'description': 'Práctica de transiciones T1 y T2 con mini sesiones de cada disciplina.',
            'structure': f'Nado {swim_time:.0f}min + T1 + Bici {bike_time:.0f}min + T2 + Carrera {run_time:.0f}min',
            'segments': [
                {'name': '🏊 Mini nado', 'distance_km': None, 'duration_min': swim_time,
                 'pace': self._format_swim_pace(css * 1.10), 'pace_range': 'Z2-Z3',
                 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Nado ritmo competición, últimos 200m fuerte'},
                {'name': '🔄 T1 Transición', 'distance_km': None, 'duration_min': 3,
                 'pace': 'Máxima velocidad', 'pace_range': 'Transición',
                 'hr_min': '', 'hr_max': '', 'zone': 0,
                 'notes': 'Practicar: quitar neopreno, casco, montar bici'},
                {'name': '🚲 Mini bici', 'distance_km': None, 'duration_min': bike_time,
                 'pace': f"{z2_bike.get('min_watts', int(self.ftp*0.56))}-{z2_bike.get('max_watts', int(self.ftp*0.75))}W",
                 'pace_range': 'Z2: 56-75% FTP',
                 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Ritmo moderado, últimos 5min subir intensidad'},
                {'name': '🔄 T2 Transición', 'distance_km': None, 'duration_min': 3,
                 'pace': 'Máxima velocidad', 'pace_range': 'Transición',
                 'hr_min': '', 'hr_max': '', 'zone': 0,
                 'notes': 'Practicar: bajar bici, cambiar zapatillas'},
                {'name': '🏃 Mini carrera', 'distance_km': round(run_time / pace_suave, 1),
                 'duration_min': run_time,
                 'pace': self._format_pace(pace_suave), 'pace_range': f'Z2: {self._format_pace(pace_suave)}',
                 'hr_min': '', 'hr_max': '', 'zone': 2,
                 'notes': 'Empezar controlado, ir subiendo ritmo'}
            ]
        }
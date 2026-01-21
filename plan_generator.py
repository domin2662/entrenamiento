"""
Training Plan Generator - Domin Cideam.es
Genera planes de entrenamiento periodizados basados en objetivos y nivel de forma física.
Incluye entrenamientos detallados con series, descansos y ritmos específicos.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math


class TrainingPlanGenerator:
    """Genera planes de entrenamiento de carrera personalizados con detalle completo."""

    def __init__(
        self,
        target_distance: float,
        target_date: datetime,
        current_fitness: dict,
        training_zones: dict,
        intensity_preference: str = "moderate",
        preferred_workouts: dict = None,
        training_days_per_week: int = 4,
        best_5k_time: str = None,
        best_10k_time: str = None
    ):
        """
        Inicializa el generador de planes de entrenamiento.

        Args:
            target_distance: Distancia objetivo en km (10, 15, 21.1, 42.2)
            target_date: Fecha objetivo de la carrera
            current_fitness: Diccionario con métricas de forma física actual
            training_zones: Diccionario con zonas de FC
            intensity_preference: "easy", "moderate", o "high"
            preferred_workouts: Diccionario de tipos de entrenamiento preferidos
            training_days_per_week: Días de entrenamiento (3-7)
            best_5k_time: Mejor tiempo 5K (MM:SS) - últimos 4-5 meses
            best_10k_time: Mejor tiempo 10K (MM:SS) - últimos 4-5 meses
        """
        self.target_distance = target_distance
        self.target_date = target_date
        self.current_fitness = current_fitness
        self.training_zones = training_zones
        self.intensity_preference = intensity_preference
        self.preferred_workouts = preferred_workouts or {
            "easy": True, "tempo": True, "intervals": True,
            "long": True, "hills": False, "fartlek": False
        }
        self.training_days_per_week = training_days_per_week
        self.best_5k_time = best_5k_time
        self.best_10k_time = best_10k_time

        # Calcular ritmos de entrenamiento basados en tiempos personales
        self._calculate_training_paces()

        # Calcular semanas hasta la carrera
        today = datetime.now().date()
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        self.weeks_to_race = (target_date - today).days // 7

        # Distancia base semanal según forma física actual
        self.base_weekly_distance = current_fitness.get('avg_weekly_distance', 20)

        # Calcular distancia pico según distancia objetivo
        self._set_training_parameters()

    def _calculate_training_paces(self):
        """Calcula los ritmos de entrenamiento basados en tiempos personales."""
        # Ritmo base (si no hay tiempos, usar estimación conservadora)
        base_pace = 6.0  # min/km por defecto

        if self.best_5k_time:
            try:
                parts = self.best_5k_time.split(':')
                total_minutes = int(parts[0]) + int(parts[1]) / 60
                base_pace = total_minutes / 5.0  # Ritmo 5K
            except:
                pass
        elif self.best_10k_time:
            try:
                parts = self.best_10k_time.split(':')
                total_minutes = int(parts[0]) + int(parts[1]) / 60
                base_pace = (total_minutes / 10.0) * 0.95  # Estimar ritmo 5K
            except:
                pass

        # Calcular todos los ritmos de entrenamiento (en min/km)
        self.paces = {
            'recuperacion': base_pace * 1.25,      # Recuperación: +25%
            'suave': base_pace * 1.15,             # Rodaje suave: +15%
            'aerobico': base_pace * 1.10,          # Aeróbico: +10%
            'maraton': base_pace * 1.05,           # Ritmo maratón: +5%
            'medio_maraton': base_pace * 1.02,     # Ritmo medio maratón: +2%
            'umbral': base_pace * 0.98,            # Umbral: -2%
            '10k': base_pace * 0.95,               # Ritmo 10K: -5%
            '5k': base_pace,                       # Ritmo 5K: base
            '3k': base_pace * 0.95,                # Ritmo 3K: -5%
            '1500': base_pace * 0.90,              # Ritmo 1500: -10%
            'repeticiones': base_pace * 0.88,      # Repeticiones: -12%
        }

    def _format_pace(self, pace_min_per_km: float) -> str:
        """Formatea el ritmo en min:seg/km."""
        minutes = int(pace_min_per_km)
        seconds = int((pace_min_per_km - minutes) * 60)
        return f"{minutes}:{seconds:02d}/km"
    
    def _set_training_parameters(self):
        """Set training parameters based on race distance and intensity preference."""
        # Peak weekly distance as percentage of race distance
        distance_multipliers = {
            10: 4.0,    # 40km peak for 10K
            15: 3.5,    # 52.5km peak for 15K
            21.1: 3.0,  # 63km peak for half marathon
            42.2: 2.5   # 105km peak for marathon
        }
        
        self.peak_weekly_distance = self.target_distance * distance_multipliers.get(
            self.target_distance, 3.0
        )
        
        # Adjust based on current fitness
        min_peak = self.base_weekly_distance * 1.5
        self.peak_weekly_distance = max(min_peak, self.peak_weekly_distance)
        
        # Long run max distance
        long_run_percentages = {
            10: 0.8,     # 8km for 10K
            15: 0.85,    # 12.75km for 15K
            21.1: 0.9,   # 19km for half marathon
            42.2: 0.75   # 32km for marathon
        }
        
        self.max_long_run = self.target_distance * long_run_percentages.get(
            self.target_distance, 0.85
        )
        
        # Intensity distribution based on preference
        if self.intensity_preference == "easy":
            self.easy_percentage = 0.80
            self.tempo_percentage = 0.15
            self.interval_percentage = 0.05
        elif self.intensity_preference == "high":
            self.easy_percentage = 0.60
            self.tempo_percentage = 0.20
            self.interval_percentage = 0.20
        else:  # moderate
            self.easy_percentage = 0.70
            self.tempo_percentage = 0.20
            self.interval_percentage = 0.10
    
    def generate_plan(self) -> dict:
        """
        Generate the complete training plan.
        
        Returns:
            Dictionary containing the full training plan
        """
        weeks = []
        
        # Determine training phases
        phases = self._calculate_phases()
        
        current_week_distance = self.base_weekly_distance
        week_start_date = datetime.now().date()
        
        for week_num in range(1, self.weeks_to_race + 1):
            # Determine phase and if it's a recovery week
            phase = self._get_phase(week_num, phases)
            is_recovery = self._is_recovery_week(week_num)
            
            # Calculate target distance for this week
            if is_recovery:
                target_distance = current_week_distance * 0.7
            else:
                # Progressive increase
                progress = min(1.0, week_num / (self.weeks_to_race * 0.75))
                target_distance = self.base_weekly_distance + (
                    (self.peak_weekly_distance - self.base_weekly_distance) * progress
                )
                
                # Taper in final weeks
                if week_num > self.weeks_to_race - 3:
                    taper_factor = 1 - ((week_num - (self.weeks_to_race - 3)) * 0.2)
                    target_distance *= taper_factor
            
            # Generate workouts for the week
            week_workouts = self._generate_week_workouts(
                week_num, target_distance, phase, is_recovery
            )
            
            actual_distance = sum(w.get('distance', 0) for w in week_workouts.values())
            
            weeks.append({
                'week_number': week_num,
                'phase': phase,
                'is_recovery': is_recovery,
                'target_distance': target_distance,
                'total_distance': actual_distance,
                'start_date': week_start_date,
                'workouts': week_workouts
            })
            
            week_start_date += timedelta(days=7)
            current_week_distance = target_distance
        
        return {
            'weeks': weeks,
            'total_weeks': self.weeks_to_race,
            'target_distance': self.target_distance,
            'target_date': self.target_date,
            'peak_week_distance': self.peak_weekly_distance,
            'phases': phases
        }

    def _calculate_phases(self) -> dict:
        """Calculate training phases based on weeks available."""
        total_weeks = self.weeks_to_race

        if total_weeks >= 16:
            return {
                'base': (1, int(total_weeks * 0.25)),
                'build': (int(total_weeks * 0.25) + 1, int(total_weeks * 0.6)),
                'peak': (int(total_weeks * 0.6) + 1, total_weeks - 2),
                'taper': (total_weeks - 1, total_weeks)
            }
        elif total_weeks >= 8:
            return {
                'base': (1, int(total_weeks * 0.2)),
                'build': (int(total_weeks * 0.2) + 1, int(total_weeks * 0.65)),
                'peak': (int(total_weeks * 0.65) + 1, total_weeks - 1),
                'taper': (total_weeks, total_weeks)
            }
        else:
            return {
                'base': (1, 1),
                'build': (2, total_weeks - 1),
                'peak': (total_weeks - 1, total_weeks - 1),
                'taper': (total_weeks, total_weeks)
            }

    def _get_phase(self, week_num: int, phases: dict) -> str:
        """Get the training phase for a specific week."""
        for phase_name, (start, end) in phases.items():
            if start <= week_num <= end:
                return phase_name
        return 'build'

    def _is_recovery_week(self, week_num: int) -> bool:
        """Determine if a week should be a recovery week."""
        # Recovery every 4th week, except during taper
        if week_num > self.weeks_to_race - 2:
            return False
        return week_num % 4 == 0

    def _generate_week_workouts(
        self,
        week_num: int,
        target_distance: float,
        phase: str,
        is_recovery: bool
    ) -> Dict[str, dict]:
        """Genera entrenamientos específicos para una semana con detalle completo."""
        workouts = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_es = {
            'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
            'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        }

        # Determinar tipos de entrenamiento para la semana
        workout_schedule = self._create_workout_schedule(phase, is_recovery, week_num)

        # Distribuir distancia entre entrenamientos
        remaining_distance = target_distance

        for day in days:
            if day in workout_schedule:
                workout_type = workout_schedule[day]
                distance = self._calculate_workout_distance(
                    workout_type, remaining_distance, target_distance, phase
                )
                remaining_distance -= distance

                zone_info = self._get_zone_for_workout(workout_type)

                # Generar estructura detallada del entrenamiento
                detailed_workout = self._generate_detailed_workout(
                    workout_type, distance, phase, week_num
                )

                workout_data = {
                    'type': workout_type,
                    'type_es': detailed_workout['type_es'],
                    'day_es': day_names_es[day],
                    'distance': round(distance, 1),
                    'zone': zone_info.get('zone_num', 2),
                    'hr_min': zone_info.get('min_hr', 120),
                    'hr_max': zone_info.get('max_hr', 150),
                    'description': detailed_workout['description'],
                    'structure': detailed_workout['structure'],
                    'segments': detailed_workout['segments']
                }

                # Añadir datos específicos de series
                if workout_type == 'Intervals':
                    if 'reps' in detailed_workout:
                        workout_data['reps'] = detailed_workout['reps']
                    if 'rep_distance' in detailed_workout:
                        workout_data['rep_distance'] = detailed_workout['rep_distance']
                    if 'recovery_time' in detailed_workout:
                        workout_data['recovery_time'] = detailed_workout['recovery_time']
                    if 'recovery_distance' in detailed_workout:
                        workout_data['recovery_distance'] = detailed_workout['recovery_distance']
                    if 'recovery_type' in detailed_workout:
                        workout_data['recovery_type'] = detailed_workout['recovery_type']
                    if 'pace_type' in detailed_workout:
                        workout_data['pace_type'] = detailed_workout['pace_type']

                # Añadir datos específicos de fartlek
                elif workout_type == 'Fartlek':
                    if 'num_cambios' in detailed_workout:
                        workout_data['num_cambios'] = detailed_workout['num_cambios']
                    if 'duracion_rapido' in detailed_workout:
                        workout_data['duracion_rapido'] = detailed_workout['duracion_rapido']
                    if 'duracion_suave' in detailed_workout:
                        workout_data['duracion_suave'] = detailed_workout['duracion_suave']

                # Añadir datos específicos de cuestas
                elif workout_type == 'Hill Repeats':
                    if 'reps' in detailed_workout:
                        workout_data['reps'] = detailed_workout['reps']
                    if 'rep_duration' in detailed_workout:
                        workout_data['rep_duration'] = detailed_workout['rep_duration']
                    if 'recovery_duration' in detailed_workout:
                        workout_data['recovery_duration'] = detailed_workout['recovery_duration']
                    if 'incline' in detailed_workout:
                        workout_data['incline'] = detailed_workout['incline']

                workouts[day] = workout_data
            else:
                workouts[day] = {
                    'type': 'Rest',
                    'type_es': 'Descanso',
                    'day_es': day_names_es[day],
                    'distance': 0,
                    'zone': '-',
                    'hr_min': '',
                    'hr_max': '',
                    'description': 'Día de descanso - recuperación activa o descanso completo',
                    'structure': None,
                    'segments': []
                }

        return workouts

    def _generate_detailed_workout(
        self,
        workout_type: str,
        distance: float,
        phase: str,
        week_num: int
    ) -> dict:
        """Genera la estructura detallada de un entrenamiento con series, descansos y ritmos."""

        type_translations = {
            'Easy Run': 'Rodaje Suave',
            'Recovery Run': 'Recuperación',
            'Long Run': 'Tirada Larga',
            'Tempo Run': 'Tempo',
            'Intervals': 'Series',
            'Hill Repeats': 'Cuestas',
            'Fartlek': 'Fartlek'
        }

        result = {
            'type_es': type_translations.get(workout_type, workout_type),
            'description': '',
            'structure': '',
            'segments': []
        }

        if workout_type == 'Easy Run':
            result = self._build_easy_run(distance)
        elif workout_type == 'Recovery Run':
            result = self._build_recovery_run(distance)
        elif workout_type == 'Long Run':
            result = self._build_long_run(distance, phase)
        elif workout_type == 'Tempo Run':
            result = self._build_tempo_run(distance, phase)
        elif workout_type == 'Intervals':
            result = self._build_intervals(distance, phase, week_num)
        elif workout_type == 'Hill Repeats':
            result = self._build_hill_repeats(distance, phase, week_num)
        elif workout_type == 'Fartlek':
            result = self._build_fartlek(distance, phase, week_num)

        return result

    def _build_easy_run(self, distance: float) -> dict:
        """Construye entrenamiento de rodaje suave."""
        pace = self.paces['suave']
        hr_zone = self.training_zones.get('zone_2', {})

        return {
            'type_es': 'Rodaje Suave',
            'description': f'Rodaje continuo a ritmo suave. Mantén conversación fácil.',
            'structure': f'{distance:.1f}km a ritmo suave',
            'segments': [{
                'name': 'Rodaje principal',
                'distance_km': distance,
                'duration_min': None,
                'pace': self._format_pace(pace),
                'pace_range': f"{self._format_pace(pace)} - {self._format_pace(pace * 1.05)}",
                'hr_min': hr_zone.get('min_hr', 130),
                'hr_max': hr_zone.get('max_hr', 150),
                'zone': 2,
                'rest_after': None,
                'notes': 'Esfuerzo conversacional'
            }]
        }

    def _build_recovery_run(self, distance: float) -> dict:
        """Construye entrenamiento de recuperación."""
        pace = self.paces['recuperacion']
        hr_zone = self.training_zones.get('zone_1', {})

        return {
            'type_es': 'Recuperación',
            'description': 'Trote muy suave para recuperación activa.',
            'structure': f'{distance:.1f}km muy suave',
            'segments': [{
                'name': 'Recuperación',
                'distance_km': distance,
                'duration_min': None,
                'pace': self._format_pace(pace),
                'pace_range': f"{self._format_pace(pace)} o más lento",
                'hr_min': hr_zone.get('min_hr', 110),
                'hr_max': hr_zone.get('max_hr', 130),
                'zone': 1,
                'rest_after': None,
                'notes': 'Debe sentirse muy fácil'
            }]
        }

    def _build_long_run(self, distance: float, phase: str) -> dict:
        """Construye entrenamiento de tirada larga."""
        pace = self.paces['aerobico']
        hr_zone_2 = self.training_zones.get('zone_2', {})

        segments = []

        # Calentamiento
        warmup_dist = min(2.0, distance * 0.15)
        segments.append({
            'name': 'Calentamiento',
            'distance_km': warmup_dist,
            'duration_min': None,
            'pace': self._format_pace(self.paces['suave']),
            'pace_range': f"{self._format_pace(self.paces['suave'])} - {self._format_pace(self.paces['recuperacion'])}",
            'hr_min': hr_zone_2.get('min_hr', 130),
            'hr_max': hr_zone_2.get('max_hr', 145),
            'zone': 2,
            'rest_after': None,
            'notes': 'Empezar muy suave'
        })

        # Bloque principal
        main_dist = distance - warmup_dist - 1.0
        if phase == 'peak' and main_dist > 10:
            # En fase pico, incluir ritmo objetivo
            progression_dist = main_dist * 0.3
            easy_dist = main_dist - progression_dist

            segments.append({
                'name': 'Bloque aeróbico',
                'distance_km': easy_dist,
                'duration_min': None,
                'pace': self._format_pace(pace),
                'pace_range': f"{self._format_pace(pace * 0.98)} - {self._format_pace(pace * 1.05)}",
                'hr_min': hr_zone_2.get('min_hr', 130),
                'hr_max': hr_zone_2.get('max_hr', 150),
                'zone': 2,
                'rest_after': None,
                'notes': 'Ritmo constante y cómodo'
            })

            segments.append({
                'name': 'Progresión a ritmo objetivo',
                'distance_km': progression_dist,
                'duration_min': None,
                'pace': self._format_pace(self.paces['maraton']),
                'pace_range': f"{self._format_pace(self.paces['maraton'])} - {self._format_pace(self.paces['medio_maraton'])}",
                'hr_min': 150,
                'hr_max': 165,
                'zone': 3,
                'rest_after': None,
                'notes': 'Aumentar ritmo gradualmente'
            })
        else:
            segments.append({
                'name': 'Bloque principal',
                'distance_km': main_dist,
                'duration_min': None,
                'pace': self._format_pace(pace),
                'pace_range': f"{self._format_pace(pace * 0.98)} - {self._format_pace(pace * 1.05)}",
                'hr_min': hr_zone_2.get('min_hr', 130),
                'hr_max': hr_zone_2.get('max_hr', 150),
                'zone': 2,
                'rest_after': None,
                'notes': 'Mantener ritmo constante'
            })

        # Vuelta a la calma
        segments.append({
            'name': 'Vuelta a la calma',
            'distance_km': 1.0,
            'duration_min': None,
            'pace': self._format_pace(self.paces['recuperacion']),
            'pace_range': f"{self._format_pace(self.paces['recuperacion'])} o más lento",
            'hr_min': 110,
            'hr_max': 130,
            'zone': 1,
            'rest_after': None,
            'notes': 'Trote muy suave para recuperar'
        })

        return {
            'type_es': 'Tirada Larga',
            'description': f'Tirada larga de {distance:.1f}km para desarrollar resistencia aeróbica.',
            'structure': f'Calentamiento + {main_dist:.1f}km aeróbico + Vuelta calma',
            'segments': segments
        }

    def _build_tempo_run(self, distance: float, phase: str) -> dict:
        """Construye entrenamiento de tempo/umbral."""
        hr_zone_3 = self.training_zones.get('zone_3', {})

        warmup_dist = 2.0
        cooldown_dist = 1.5
        tempo_dist = distance - warmup_dist - cooldown_dist

        segments = [
            {
                'name': 'Calentamiento',
                'distance_km': warmup_dist,
                'duration_min': None,
                'pace': self._format_pace(self.paces['suave']),
                'pace_range': f"{self._format_pace(self.paces['suave'])}",
                'hr_min': 120,
                'hr_max': 140,
                'zone': 2,
                'rest_after': None,
                'notes': 'Incluir 4-6 progresiones de 100m'
            },
            {
                'name': 'Bloque Tempo',
                'distance_km': tempo_dist,
                'duration_min': None,
                'pace': self._format_pace(self.paces['umbral']),
                'pace_range': f"{self._format_pace(self.paces['umbral'])} - {self._format_pace(self.paces['10k'])}",
                'hr_min': hr_zone_3.get('min_hr', 155),
                'hr_max': hr_zone_3.get('max_hr', 170),
                'zone': 3,
                'rest_after': None,
                'notes': 'Ritmo "cómodamente duro" - poder hablar frases cortas'
            },
            {
                'name': 'Vuelta a la calma',
                'distance_km': cooldown_dist,
                'duration_min': None,
                'pace': self._format_pace(self.paces['recuperacion']),
                'pace_range': f"{self._format_pace(self.paces['recuperacion'])}",
                'hr_min': 110,
                'hr_max': 130,
                'zone': 1,
                'rest_after': None,
                'notes': 'Trote muy suave'
            }
        ]

        return {
            'type_es': 'Tempo',
            'description': f'Entrenamiento de umbral con {tempo_dist:.1f}km a ritmo tempo.',
            'structure': f'2km calentamiento + {tempo_dist:.1f}km tempo + 1.5km vuelta calma',
            'segments': segments
        }

    def _build_intervals(self, distance: float, phase: str, week_num: int) -> dict:
        """Construye entrenamiento de series/intervalos con detalle completo."""
        hr_zone_4 = self.training_zones.get('zone_4', {})
        hr_zone_5 = self.training_zones.get('zone_5', {})

        # Definir tipo de series según fase y semana con más detalles
        if phase == 'base':
            interval_options = [
                {'reps': 8, 'distance': 400, 'pace_type': '5k', 'rest_dist': 200, 'rest_time': 90, 'rest_type': 'trote'},
                {'reps': 10, 'distance': 300, 'pace_type': '3k', 'rest_dist': 100, 'rest_time': 60, 'rest_type': 'caminar'},
                {'reps': 6, 'distance': 500, 'pace_type': '5k', 'rest_dist': 200, 'rest_time': 90, 'rest_type': 'trote'},
            ]
        elif phase == 'build':
            interval_options = [
                {'reps': 6, 'distance': 800, 'pace_type': '5k', 'rest_dist': 400, 'rest_time': 120, 'rest_type': 'trote'},
                {'reps': 5, 'distance': 1000, 'pace_type': '10k', 'rest_dist': 400, 'rest_time': 150, 'rest_type': 'trote'},
                {'reps': 4, 'distance': 1200, 'pace_type': '10k', 'rest_dist': 400, 'rest_time': 180, 'rest_type': 'trote'},
            ]
        elif phase == 'peak':
            interval_options = [
                {'reps': 4, 'distance': 1600, 'pace_type': '10k', 'rest_dist': 400, 'rest_time': 180, 'rest_type': 'trote'},
                {'reps': 3, 'distance': 2000, 'pace_type': 'umbral', 'rest_dist': 600, 'rest_time': 240, 'rest_type': 'trote'},
                {'reps': 5, 'distance': 1000, 'pace_type': '5k', 'rest_dist': 400, 'rest_time': 150, 'rest_type': 'trote'},
            ]
        else:  # taper
            interval_options = [
                {'reps': 4, 'distance': 400, 'pace_type': '5k', 'rest_dist': 400, 'rest_time': 120, 'rest_type': 'trote'},
                {'reps': 3, 'distance': 600, 'pace_type': '10k', 'rest_dist': 300, 'rest_time': 90, 'rest_type': 'trote'},
            ]

        interval_config = interval_options[week_num % len(interval_options)]

        reps = interval_config['reps']
        interval_dist = interval_config['distance']
        pace_type = interval_config['pace_type']
        rest_dist = interval_config['rest_dist']
        rest_time = interval_config['rest_time']
        rest_type = interval_config['rest_type']

        # Formatear tiempo de recuperación
        if rest_time >= 60:
            rest_time_str = f"{rest_time // 60}:{rest_time % 60:02d}" if rest_time % 60 else f"{rest_time // 60} min"
        else:
            rest_time_str = f"{rest_time}s"

        rest_desc = f'{rest_dist}m {rest_type} ({rest_time_str})'
        interval_pace = self.paces.get(pace_type, self.paces['5k'])

        # Estimar tiempo por intervalo
        est_time_per_rep = interval_dist / 1000 * interval_pace  # minutos

        segments = [
            {
                'name': 'Calentamiento',
                'distance_km': 2.0,
                'duration_min': 15,
                'pace': self._format_pace(self.paces['suave']),
                'pace_range': f"{self._format_pace(self.paces['suave'])}",
                'hr_min': 120,
                'hr_max': 140,
                'zone': 2,
                'rest_after': None,
                'notes': 'Trote suave + 4x80m progresivos + ejercicios técnica'
            },
            {
                'name': f'{reps}x {interval_dist}m',
                'distance_km': (reps * interval_dist) / 1000,
                'duration_min': reps * (est_time_per_rep + rest_time / 60),
                'pace': self._format_pace(interval_pace),
                'pace_range': f"{self._format_pace(interval_pace * 0.97)} - {self._format_pace(interval_pace * 1.03)}",
                'hr_min': hr_zone_4.get('min_hr', 165),
                'hr_max': hr_zone_5.get('max_hr', 185),
                'zone': '4-5',
                'rest_after': rest_desc,
                'reps': reps,
                'rep_distance': interval_dist,
                'recovery_distance': rest_dist,
                'recovery_time': rest_time,
                'recovery_type': rest_type,
                'target_pace': self._format_pace(interval_pace),
                'estimated_time_per_rep': f'{int(est_time_per_rep)}:{int((est_time_per_rep % 1) * 60):02d}',
                'notes': f'{reps}x {interval_dist}m a ritmo {pace_type.upper()}. Rec: {rest_desc} entre series.'
            },
            {
                'name': 'Vuelta a la calma',
                'distance_km': 1.5,
                'duration_min': 10,
                'pace': self._format_pace(self.paces['recuperacion']),
                'pace_range': f"{self._format_pace(self.paces['recuperacion'])}",
                'hr_min': 100,
                'hr_max': 120,
                'zone': 1,
                'rest_after': None,
                'notes': 'Trote muy suave + estiramientos'
            }
        ]

        return {
            'type_es': 'Series',
            'description': f'{reps}x {interval_dist}m a ritmo {pace_type.upper()}. Rec: {rest_desc}.',
            'structure': f'2km calent. + {reps}x{interval_dist}m ({rest_desc}) + 1.5km vuelta calma',
            'reps': reps,
            'rep_distance': interval_dist,
            'recovery_distance': rest_dist,
            'recovery_time': rest_time,
            'recovery_type': rest_type,
            'pace_type': pace_type,
            'segments': segments
        }

    def _build_hill_repeats(self, distance: float, phase: str, week_num: int = 1) -> dict:
        """Construye entrenamiento de cuestas con detalles específicos."""
        hr_zone_4 = self.training_zones.get('zone_4', {})
        hr_zone_5 = self.training_zones.get('zone_5', {})

        # Configuraciones de cuestas según fase
        if phase == 'base':
            hill_configs = [
                {'reps': 6, 'duration': 45, 'incline': '4-6%', 'recovery_sec': 90, 'intensity': 'moderada'},
                {'reps': 8, 'duration': 30, 'incline': '6-8%', 'recovery_sec': 60, 'intensity': 'fuerte'},
                {'reps': 5, 'duration': 60, 'incline': '4-5%', 'recovery_sec': 120, 'intensity': 'moderada'},
            ]
        elif phase == 'build':
            hill_configs = [
                {'reps': 8, 'duration': 60, 'incline': '5-7%', 'recovery_sec': 90, 'intensity': 'fuerte'},
                {'reps': 6, 'duration': 90, 'incline': '4-6%', 'recovery_sec': 120, 'intensity': 'fuerte'},
                {'reps': 10, 'duration': 45, 'incline': '6-8%', 'recovery_sec': 75, 'intensity': 'muy fuerte'},
            ]
        elif phase == 'peak':
            hill_configs = [
                {'reps': 6, 'duration': 90, 'incline': '5-7%', 'recovery_sec': 90, 'intensity': 'muy fuerte'},
                {'reps': 5, 'duration': 120, 'incline': '4-6%', 'recovery_sec': 150, 'intensity': 'fuerte'},
            ]
        else:  # taper
            hill_configs = [
                {'reps': 4, 'duration': 45, 'incline': '4-5%', 'recovery_sec': 120, 'intensity': 'moderada'},
                {'reps': 5, 'duration': 30, 'incline': '5-6%', 'recovery_sec': 90, 'intensity': 'moderada'},
            ]

        config = hill_configs[week_num % len(hill_configs)]
        reps = config['reps']
        duration = config['duration']
        incline = config['incline']
        recovery_sec = config['recovery_sec']
        intensity = config['intensity']

        # Formatear tiempo de recuperación
        if recovery_sec >= 60:
            recovery_str = f"{recovery_sec // 60}:{recovery_sec % 60:02d}" if recovery_sec % 60 else f"{recovery_sec // 60} min"
        else:
            recovery_str = f"{recovery_sec}s"

        segments = [
            {
                'name': 'Calentamiento',
                'distance_km': 2.0,
                'duration_min': 15,
                'pace': self._format_pace(self.paces['suave']),
                'pace_range': f"{self._format_pace(self.paces['suave'])}",
                'hr_min': 120,
                'hr_max': 140,
                'zone': 2,
                'rest_after': None,
                'notes': 'Trote suave + ejercicios técnica + 4x80m progresivos'
            },
            {
                'name': f'{reps}x cuesta {duration}s',
                'distance_km': None,
                'duration_min': reps * (duration / 60 + recovery_sec / 60),
                'pace': 'Esfuerzo ' + intensity,
                'pace_range': f'RPE {8 if intensity == "moderada" else 9}/10',
                'hr_min': hr_zone_4.get('min_hr', 165),
                'hr_max': hr_zone_5.get('max_hr', 185),
                'zone': 4,
                'rest_after': f'{recovery_str} bajando trotando',
                'reps': reps,
                'rep_duration': duration,
                'recovery_duration': recovery_sec,
                'incline': incline,
                'notes': f'Subir {duration}s en pendiente {incline}. Bajar trotando en {recovery_str}.'
            },
            {
                'name': 'Vuelta a la calma',
                'distance_km': 1.5,
                'duration_min': 10,
                'pace': self._format_pace(self.paces['recuperacion']),
                'pace_range': f"{self._format_pace(self.paces['recuperacion'])}",
                'hr_min': 100,
                'hr_max': 120,
                'zone': 1,
                'rest_after': None,
                'notes': 'Trote muy suave + estiramientos'
            }
        ]

        return {
            'type_es': 'Cuestas',
            'description': f'{reps}x cuesta de {duration}s (pendiente {incline}), rec: {recovery_str} trotando.',
            'structure': f'2km calent. + {reps}x{duration}s cuesta + {recovery_str} rec + 1.5km vuelta calma',
            'reps': reps,
            'rep_duration': duration,
            'recovery_duration': recovery_sec,
            'incline': incline,
            'segments': segments
        }

    def _build_fartlek(self, distance: float, phase: str, week_num: int = 1) -> dict:
        """Construye entrenamiento de fartlek con detalles específicos de cambios."""
        hr_zone_3 = self.training_zones.get('zone_3', {})
        hr_zone_4 = self.training_zones.get('zone_4', {})

        # Configuraciones de fartlek según fase
        if phase == 'base':
            fartlek_configs = [
                {'num_cambios': 8, 'dur_rapido': 45, 'dur_suave': 90, 'tipo': 'corto'},
                {'num_cambios': 10, 'dur_rapido': 30, 'dur_suave': 60, 'tipo': 'muy corto'},
                {'num_cambios': 6, 'dur_rapido': 60, 'dur_suave': 120, 'tipo': 'medio'},
            ]
        elif phase == 'build':
            fartlek_configs = [
                {'num_cambios': 6, 'dur_rapido': 90, 'dur_suave': 90, 'tipo': 'medio'},
                {'num_cambios': 8, 'dur_rapido': 60, 'dur_suave': 60, 'tipo': '1:1'},
                {'num_cambios': 5, 'dur_rapido': 120, 'dur_suave': 90, 'tipo': 'largo'},
            ]
        elif phase == 'peak':
            fartlek_configs = [
                {'num_cambios': 5, 'dur_rapido': 180, 'dur_suave': 90, 'tipo': 'largo'},
                {'num_cambios': 4, 'dur_rapido': 240, 'dur_suave': 120, 'tipo': 'muy largo'},
                {'num_cambios': 6, 'dur_rapido': 120, 'dur_suave': 60, 'tipo': 'intenso'},
            ]
        else:  # taper
            fartlek_configs = [
                {'num_cambios': 5, 'dur_rapido': 45, 'dur_suave': 90, 'tipo': 'suave'},
                {'num_cambios': 4, 'dur_rapido': 60, 'dur_suave': 120, 'tipo': 'suave'},
            ]

        config = fartlek_configs[week_num % len(fartlek_configs)]
        num_cambios = config['num_cambios']
        dur_rapido = config['dur_rapido']
        dur_suave = config['dur_suave']

        # Formatear tiempos
        def format_duration(seconds):
            if seconds >= 60:
                mins = seconds // 60
                secs = seconds % 60
                return f"{mins}:{secs:02d}" if secs else f"{mins} min"
            return f"{seconds}s"

        segments = [
            {
                'name': 'Calentamiento',
                'distance_km': 2.0,
                'duration_min': 12,
                'pace': self._format_pace(self.paces['suave']),
                'pace_range': f"{self._format_pace(self.paces['suave'])}",
                'hr_min': 120,
                'hr_max': 140,
                'zone': 2,
                'rest_after': None,
                'notes': 'Trote suave + 4x80m progresivos'
            },
            {
                'name': f'{num_cambios} cambios de ritmo',
                'distance_km': distance - 3.5,
                'duration_min': num_cambios * (dur_rapido + dur_suave) / 60,
                'pace': 'Variable',
                'pace_range': f"{self._format_pace(self.paces['5k'])} - {self._format_pace(self.paces['suave'])}",
                'hr_min': hr_zone_3.get('min_hr', 150),
                'hr_max': hr_zone_4.get('max_hr', 175),
                'zone': '2-4',
                'rest_after': f'{format_duration(dur_suave)} trote suave',
                'reps': num_cambios,
                'rep_duration': dur_rapido,
                'recovery_duration': dur_suave,
                'fast_pace': self._format_pace(self.paces.get('10k', self.paces['5k'])),
                'slow_pace': self._format_pace(self.paces['suave']),
                'notes': f'{num_cambios}x {format_duration(dur_rapido)} rápido + {format_duration(dur_suave)} suave'
            },
            {
                'name': 'Vuelta a la calma',
                'distance_km': 1.5,
                'duration_min': 10,
                'pace': self._format_pace(self.paces['recuperacion']),
                'pace_range': f"{self._format_pace(self.paces['recuperacion'])}",
                'hr_min': 100,
                'hr_max': 120,
                'zone': 1,
                'rest_after': None,
                'notes': 'Trote muy suave + estiramientos'
            }
        ]

        return {
            'type_es': 'Fartlek',
            'description': f'Fartlek: {num_cambios}x ({format_duration(dur_rapido)} rápido / {format_duration(dur_suave)} suave).',
            'structure': f'2km calent. + {num_cambios}x({format_duration(dur_rapido)}/{format_duration(dur_suave)}) + 1.5km vuelta calma',
            'num_cambios': num_cambios,
            'duracion_rapido': dur_rapido,
            'duracion_suave': dur_suave,
            'segments': segments
        }

    def _create_workout_schedule(self, phase: str, is_recovery: bool, week_num: int = 1) -> Dict[str, str]:
        """Create a weekly workout schedule based on training days, preferences and week number."""
        schedule = {}

        # Available workout types based on preferences
        available_workouts = []
        if self.preferred_workouts.get('easy', True):
            available_workouts.append('Easy Run')
        if self.preferred_workouts.get('tempo', True):
            available_workouts.append('Tempo Run')
        if self.preferred_workouts.get('intervals', True):
            available_workouts.append('Intervals')
        if self.preferred_workouts.get('long', True):
            available_workouts.append('Long Run')
        if self.preferred_workouts.get('hills', False):
            available_workouts.append('Hill Repeats')
        if self.preferred_workouts.get('fartlek', False):
            available_workouts.append('Fartlek')

        # Standard workout days based on training frequency
        if self.training_days_per_week == 3:
            day_slots = ['Tuesday', 'Thursday', 'Sunday']
        elif self.training_days_per_week == 4:
            day_slots = ['Tuesday', 'Thursday', 'Saturday', 'Sunday']
        elif self.training_days_per_week == 5:
            day_slots = ['Monday', 'Tuesday', 'Thursday', 'Saturday', 'Sunday']
        elif self.training_days_per_week == 6:
            day_slots = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday']
        else:  # 7 days
            day_slots = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Assign workouts to days
        if is_recovery:
            # Recovery week - all easy runs
            for day in day_slots:
                if day == 'Sunday':
                    schedule[day] = 'Long Run' if 'Long Run' in available_workouts else 'Easy Run'
                else:
                    schedule[day] = 'Recovery Run'
        else:
            # Normal week - mix of workouts based on phase and week number
            # Construir lista de entrenamientos de calidad disponibles
            quality_workouts = []
            if 'Intervals' in available_workouts:
                quality_workouts.append('Intervals')
            if 'Tempo Run' in available_workouts:
                quality_workouts.append('Tempo Run')
            if 'Hill Repeats' in available_workouts:
                quality_workouts.append('Hill Repeats')
            if 'Fartlek' in available_workouts:
                quality_workouts.append('Fartlek')

            # Determinar qué entrenamientos de calidad usar esta semana
            # Rotar usando week_num para variedad
            week_quality = []

            if quality_workouts:
                # Definir patrones de rotación según fase
                if phase == 'base':
                    # Fase base: más variedad, incluir cuestas y fartlek regularmente
                    rotation_patterns = [
                        ['Intervals', 'Fartlek'],
                        ['Hill Repeats', 'Tempo Run'],
                        ['Fartlek', 'Intervals'],
                        ['Intervals', 'Hill Repeats'],
                        ['Tempo Run', 'Fartlek'],
                        ['Hill Repeats', 'Intervals'],
                    ]
                elif phase == 'build':
                    # Fase build: intervalos y tempo principales, cuestas cada 3 semanas
                    rotation_patterns = [
                        ['Intervals', 'Tempo Run'],
                        ['Tempo Run', 'Intervals'],
                        ['Hill Repeats', 'Intervals'],  # Cuestas cada 3 semanas
                        ['Intervals', 'Tempo Run'],
                        ['Tempo Run', 'Fartlek'],
                        ['Intervals', 'Hill Repeats'],  # Cuestas cada 3 semanas
                    ]
                elif phase == 'peak':
                    # Fase peak: intervalos específicos y tempo
                    rotation_patterns = [
                        ['Intervals', 'Tempo Run'],
                        ['Tempo Run', 'Intervals'],
                        ['Intervals', 'Fartlek'],
                        ['Hill Repeats', 'Tempo Run'],  # Cuestas ocasionales
                        ['Intervals', 'Tempo Run'],
                    ]
                else:  # taper
                    # Taper: solo mantener agudeza
                    rotation_patterns = [
                        ['Intervals'],
                        ['Tempo Run'],
                        ['Fartlek'],
                    ]

                # Seleccionar patrón basado en semana
                pattern_idx = (week_num - 1) % len(rotation_patterns)
                pattern = rotation_patterns[pattern_idx]

                # Filtrar solo entrenamientos disponibles
                for workout in pattern:
                    if workout in quality_workouts:
                        week_quality.append(workout)
                        if len(week_quality) >= 2:
                            break

                # Si no hay suficientes, añadir lo que esté disponible
                if not week_quality and quality_workouts:
                    week_quality.append(quality_workouts[0])

            # Asignar entrenamientos a días
            quality_idx = 0
            for i, day in enumerate(day_slots):
                if day == 'Sunday':
                    # Domingo siempre tirada larga
                    if 'Long Run' in available_workouts:
                        schedule[day] = 'Long Run'
                    else:
                        schedule[day] = 'Easy Run'
                elif day == 'Saturday' and len(day_slots) > 3:
                    # Sábado: rodaje suave antes de la tirada larga
                    schedule[day] = 'Easy Run'
                elif i == 1 and week_quality:  # Primer día de calidad (martes)
                    schedule[day] = week_quality[0]
                    quality_idx = 1
                elif i == 3 and len(week_quality) > 1 and len(day_slots) >= 5:  # Segundo día calidad (jueves)
                    schedule[day] = week_quality[min(1, len(week_quality) - 1)]
                elif i == 2 and len(day_slots) >= 4 and week_quality and quality_idx == 0:
                    # Si no se asignó en i==1, asignar aquí
                    schedule[day] = week_quality[0]
                else:
                    schedule[day] = 'Easy Run'

        return schedule

    def _calculate_workout_distance(
        self,
        workout_type: str,
        remaining: float,
        total: float,
        phase: str
    ) -> float:
        """Calculate distance for a specific workout type."""
        # Base percentages of weekly distance
        distance_percentages = {
            'Long Run': 0.30,
            'Tempo Run': 0.18,
            'Intervals': 0.12,
            'Easy Run': 0.15,
            'Recovery Run': 0.10,
            'Hill Repeats': 0.12,
            'Fartlek': 0.15
        }

        percentage = distance_percentages.get(workout_type, 0.15)
        base_distance = total * percentage

        # Adjust long run for race distance
        if workout_type == 'Long Run':
            max_long = min(self.max_long_run, remaining * 0.6)
            base_distance = min(base_distance, max_long)

            # Progressive long run during build phase
            if phase == 'build':
                base_distance *= 1.1
            elif phase == 'peak':
                base_distance = max_long * 0.95
            elif phase == 'taper':
                base_distance *= 0.7

        # Minimum distances
        min_distances = {
            'Long Run': 8,
            'Tempo Run': 5,
            'Intervals': 4,
            'Easy Run': 4,
            'Recovery Run': 3,
            'Hill Repeats': 4,
            'Fartlek': 5
        }

        min_dist = min_distances.get(workout_type, 3)
        return max(min_dist, min(remaining, base_distance))

    def _get_zone_for_workout(self, workout_type: str) -> dict:
        """Get HR zone information for a workout type."""
        zone_mapping = {
            'Easy Run': ('zone_2', 2),
            'Recovery Run': ('zone_1', 1),
            'Long Run': ('zone_2', 2),
            'Tempo Run': ('zone_3', 3),
            'Intervals': ('zone_5', 5),
            'Hill Repeats': ('zone_4', 4),
            'Fartlek': ('zone_3', 3)
        }

        zone_key, zone_num = zone_mapping.get(workout_type, ('zone_2', 2))
        zone_data = self.training_zones.get(zone_key, {})

        return {
            'zone_num': zone_num,
            'min_hr': zone_data.get('min_hr', 120),
            'max_hr': zone_data.get('max_hr', 150)
        }

    def _get_workout_description(self, workout_type: str, distance: float, phase: str) -> str:
        """Generate a description for a workout."""
        descriptions = {
            'Easy Run': f"Easy-paced run at conversational effort. Stay in Zone 2. "
                       f"Focus on running form and relaxation.",

            'Recovery Run': f"Very easy run for active recovery. Keep heart rate in Zone 1. "
                           f"This run should feel effortless.",

            'Long Run': f"Build endurance with this long run. Start easy and maintain Zone 2. "
                       f"Practice race nutrition if over 90 minutes.",

            'Tempo Run': f"After 10-15 min warmup, run at comfortably hard pace (Zone 3) "
                        f"for {distance-2:.0f}km. Cool down for 10 min.",

            'Intervals': self._get_interval_description(distance, phase),

            'Hill Repeats': f"Warm up 15 min, then run 6-10 hill repeats of 60-90 seconds "
                           f"at hard effort. Jog down for recovery. Cool down 10 min.",

            'Fartlek': f"Start easy, then alternate between fast (30-90 sec) and recovery "
                      f"(1-2 min) throughout the run. Listen to your body.",

            'Rest': "Complete rest or very light cross-training (walking, swimming, yoga)."
        }

        return descriptions.get(workout_type, "Run at comfortable pace.")

    def _get_interval_description(self, distance: float, phase: str) -> str:
        """Generate specific interval workout description based on phase."""
        if phase == 'base':
            return (f"Warm up 15 min. Run 6-8 x 400m at 5K pace with 200m jog recovery. "
                   f"Cool down 10 min.")
        elif phase == 'build':
            return (f"Warm up 15 min. Run 5-6 x 800m at slightly faster than 10K pace "
                   f"with 400m jog recovery. Cool down 10 min.")
        elif phase == 'peak':
            return (f"Warm up 15 min. Run 3-4 x 1200m at 10K pace with 400m jog recovery. "
                   f"Cool down 10 min.")
        else:  # taper
            return (f"Warm up 15 min. Run 4 x 400m at 5K pace with 400m easy recovery. "
                   f"Focus on staying sharp, not building fitness.")

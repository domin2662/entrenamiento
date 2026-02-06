"""
Training Zones Calculator
Calculates heart rate and pace zones based on user metrics.
"""

class TrainingZones:
    """Calculate personalized training zones based on physiological parameters."""
    
    def __init__(self, max_hr: int, vo2_max: float, age: int, weight: float):
        """
        Initialize training zones calculator.
        
        Args:
            max_hr: Maximum heart rate in bpm
            vo2_max: VO2 max in ml/kg/min
            age: Age in years
            weight: Weight in kg
        """
        self.max_hr = max_hr
        self.vo2_max = vo2_max
        self.age = age
        self.weight = weight
        
        # Calculate resting heart rate estimation based on VO2 max and age
        # Higher VO2 max generally correlates with lower resting HR
        self.resting_hr = self._estimate_resting_hr()
        
        # Heart rate reserve (Karvonen method)
        self.hr_reserve = self.max_hr - self.resting_hr
    
    def _estimate_resting_hr(self) -> int:
        """Estimate resting heart rate based on VO2 max and age."""
        # Base resting HR decreases with better fitness (higher VO2 max)
        # Average person: ~70 bpm, Elite athlete: ~40-50 bpm
        base_hr = 80  # Starting point for sedentary person
        
        # VO2 max adjustment: better fitness = lower resting HR
        # VO2 max 30 = sedentary, 60+ = elite
        vo2_factor = (self.vo2_max - 30) * 0.5
        
        # Age adjustment: resting HR slightly increases with age
        age_factor = (self.age - 25) * 0.1 if self.age > 25 else 0
        
        estimated_hr = base_hr - vo2_factor + age_factor
        return max(40, min(80, int(estimated_hr)))
    
    def calculate_zones(self) -> dict:
        """
        Calculate 5-zone heart rate training zones using Karvonen method.
        
        Returns:
            Dictionary with zone names and HR ranges
        """
        zones = {}
        
        # Zone percentages (of HR reserve) - based on common training methodologies
        zone_definitions = {
            'zone_1': {'min_pct': 0.50, 'max_pct': 0.60, 'name': 'Recovery'},
            'zone_2': {'min_pct': 0.60, 'max_pct': 0.70, 'name': 'Aerobic Base'},
            'zone_3': {'min_pct': 0.70, 'max_pct': 0.80, 'name': 'Tempo'},
            'zone_4': {'min_pct': 0.80, 'max_pct': 0.90, 'name': 'Threshold'},
            'zone_5': {'min_pct': 0.90, 'max_pct': 1.00, 'name': 'VO2 Max'}
        }
        
        for zone_key, zone_def in zone_definitions.items():
            min_hr = int(self.resting_hr + (self.hr_reserve * zone_def['min_pct']))
            max_hr = int(self.resting_hr + (self.hr_reserve * zone_def['max_pct']))
            
            zones[zone_key] = {
                'name': zone_def['name'],
                'min_hr': min_hr,
                'max_hr': max_hr,
                'min_pct': int(zone_def['min_pct'] * 100),
                'max_pct': int(zone_def['max_pct'] * 100)
            }
        
        return zones
    
    def calculate_pace_zones(self, best_5k_time: str = None, best_10k_time: str = None) -> dict:
        """
        Calculate pace training zones based on personal records using Jack Daniels VDOT.

        Args:
            best_5k_time: Best 5K time in format "MM:SS"
            best_10k_time: Best 10K time in format "MM:SS"

        Returns:
            Dictionary with pace zone recommendations
        """
        # Calculate VDOT from race times or VO2max
        vdot = self._calculate_vdot(best_5k_time, best_10k_time)

        # Calculate all training paces based on VDOT (Jack Daniels tables approximation)
        paces = self._calculate_training_paces(vdot)

        # Return structured pace zones with ranges
        pace_zones = {
            'Recuperación': {
                'pace': self._format_pace(paces['recovery']),
                'pace_min': self._format_pace(paces['recovery']),
                'pace_max': self._format_pace(paces['recovery'] * 1.10),
                'description': 'Muy fácil, conversación fluida',
                'zone': 1
            },
            'Easy Pace': {
                'pace': self._format_pace(paces['easy']),
                'pace_min': self._format_pace(paces['easy'] * 0.95),
                'pace_max': self._format_pace(paces['easy'] * 1.05),
                'description': 'Rodaje suave, puedes hablar',
                'zone': 2
            },
            'Marathon Pace': {
                'pace': self._format_pace(paces['marathon']),
                'pace_min': self._format_pace(paces['marathon'] * 0.98),
                'pace_max': self._format_pace(paces['marathon'] * 1.02),
                'description': 'Ritmo de maratón objetivo',
                'zone': 3
            },
            'Threshold Pace': {
                'pace': self._format_pace(paces['threshold']),
                'pace_min': self._format_pace(paces['threshold'] * 0.97),
                'pace_max': self._format_pace(paces['threshold'] * 1.03),
                'description': 'Umbral, 1 hora a este ritmo',
                'zone': 4
            },
            'Interval Pace': {
                'pace': self._format_pace(paces['interval']),
                'pace_min': self._format_pace(paces['interval'] * 0.97),
                'pace_max': self._format_pace(paces['interval'] * 1.03),
                'description': 'VO2max, series de 3-5 min',
                'zone': 5
            },
            'Repetition Pace': {
                'pace': self._format_pace(paces['repetition']),
                'pace_min': self._format_pace(paces['repetition'] * 0.95),
                'pace_max': self._format_pace(paces['repetition'] * 1.02),
                'description': 'Velocidad, series cortas',
                'zone': 5
            }
        }

        return pace_zones

    def _calculate_vdot(self, best_5k_time: str = None, best_10k_time: str = None) -> float:
        """
        Calculate VDOT using interpolation from Jack Daniels tables.
        Referencia: Daniels' Running Formula (3rd Edition)

        Tabla de referencia VDOT:
        - VDOT 30: 5K=30:40, 10K=63:46
        - VDOT 35: 5K=27:00, 10K=56:03
        - VDOT 40: 5K=24:08, 10K=50:03
        - VDOT 45: 5K=21:50, 10K=45:16
        - VDOT 50: 5K=19:57, 10K=41:21
        - VDOT 55: 5K=18:22, 10K=38:06
        - VDOT 60: 5K=17:03, 10K=35:22
        """
        vdot = None

        # Tabla de referencia: (tiempo_5k_min, vdot)
        vdot_table_5k = [
            (30.67, 30), (27.0, 35), (24.13, 40), (21.83, 45),
            (19.95, 50), (18.37, 55), (17.05, 60), (15.92, 65), (14.93, 70)
        ]

        # Tabla de referencia: (tiempo_10k_min, vdot)
        vdot_table_10k = [
            (63.77, 30), (56.05, 35), (50.05, 40), (45.27, 45),
            (41.35, 50), (38.10, 55), (35.37, 60), (33.0, 65), (30.95, 70)
        ]

        if best_5k_time:
            try:
                parts = str(best_5k_time).split(':')
                total_minutes = int(parts[0]) + int(parts[1]) / 60
                # Interpolar VDOT desde tabla 5K
                vdot = self._interpolate_vdot(total_minutes, vdot_table_5k)
            except:
                pass

        if best_10k_time and vdot is None:
            try:
                parts = str(best_10k_time).split(':')
                total_minutes = int(parts[0]) + int(parts[1]) / 60
                # Interpolar VDOT desde tabla 10K
                vdot = self._interpolate_vdot(total_minutes, vdot_table_10k)
            except:
                pass

        if vdot is None:
            # Estimar desde VO2max (relación aproximada)
            vdot = self.vo2_max * 0.90
            vdot = max(30, min(70, vdot))

        return vdot

    def _interpolate_vdot(self, time_minutes: float, table: list) -> float:
        """Interpola VDOT desde una tabla de tiempos."""
        # Si el tiempo es mayor que el más lento, retornar VDOT mínimo
        if time_minutes >= table[0][0]:
            return table[0][1]
        # Si el tiempo es menor que el más rápido, retornar VDOT máximo
        if time_minutes <= table[-1][0]:
            return table[-1][1]

        # Interpolar entre dos valores
        for i in range(len(table) - 1):
            t1, v1 = table[i]
            t2, v2 = table[i + 1]
            if t2 <= time_minutes <= t1:
                # Interpolación lineal
                ratio = (t1 - time_minutes) / (t1 - t2)
                return v1 + ratio * (v2 - v1)

        return 40  # Valor por defecto

    def _calculate_training_paces(self, vdot: float) -> dict:
        """
        Calculate training paces based on VDOT using Jack Daniels tables.
        Returns paces in min/km.

        Tabla de ritmos por VDOT (min/km):
        VDOT 35: Easy=6:40-7:18, Marathon=6:02, Threshold=5:41, Interval=5:14, Rep=4:52
        VDOT 40: Easy=5:54-6:26, Marathon=5:20, Threshold=5:01, Interval=4:38, Rep=4:18
        VDOT 45: Easy=5:18-5:47, Marathon=4:48, Threshold=4:30, Interval=4:09, Rep=3:52
        VDOT 50: Easy=4:49-5:16, Marathon=4:21, Threshold=4:05, Interval=3:46, Rep=3:30
        """
        # Tablas de ritmo (vdot, pace en min/km decimal)
        easy_table = [(30, 7.8), (35, 6.95), (40, 6.17), (45, 5.52), (50, 5.02), (55, 4.60), (60, 4.25), (65, 3.95)]
        marathon_table = [(30, 7.02), (35, 6.03), (40, 5.33), (45, 4.80), (50, 4.35), (55, 3.98), (60, 3.67), (65, 3.40)]
        threshold_table = [(30, 6.53), (35, 5.68), (40, 5.02), (45, 4.50), (50, 4.08), (55, 3.73), (60, 3.43), (65, 3.18)]
        interval_table = [(30, 6.00), (35, 5.23), (40, 4.63), (45, 4.15), (50, 3.77), (55, 3.45), (60, 3.18), (65, 2.95)]
        rep_table = [(30, 5.55), (35, 4.87), (40, 4.30), (45, 3.87), (50, 3.50), (55, 3.20), (60, 2.95), (65, 2.73)]

        def interpolate_pace(table, vdot_val):
            if vdot_val <= table[0][0]:
                return table[0][1]
            if vdot_val >= table[-1][0]:
                return table[-1][1]
            for i in range(len(table) - 1):
                v1, p1 = table[i]
                v2, p2 = table[i + 1]
                if v1 <= vdot_val <= v2:
                    ratio = (vdot_val - v1) / (v2 - v1)
                    return p1 + ratio * (p2 - p1)
            return 5.5

        easy_pace = interpolate_pace(easy_table, vdot)
        recovery_pace = easy_pace * 1.10  # 10% más lento que easy

        paces = {
            'recovery': max(5.5, min(9.0, recovery_pace)),
            'easy': max(4.5, min(8.0, easy_pace)),
            'marathon': max(3.8, min(7.5, interpolate_pace(marathon_table, vdot))),
            'threshold': max(3.5, min(7.0, interpolate_pace(threshold_table, vdot))),
            'interval': max(3.2, min(6.5, interpolate_pace(interval_table, vdot))),
            'repetition': max(3.0, min(6.0, interpolate_pace(rep_table, vdot))),
        }

        return paces

    def _format_pace(self, pace_decimal: float) -> str:
        """Convert decimal pace (min/km) to MM:SS format."""
        pace_decimal = max(2.5, min(12.0, pace_decimal))
        minutes = int(pace_decimal)
        seconds = int((pace_decimal - minutes) * 60)
        return f"{minutes}:{seconds:02d}"

    def get_all_training_paces(self, best_5k_time: str = None, best_10k_time: str = None) -> dict:
        """
        Get all training paces as simple min/km values for use in workout generation.
        """
        vdot = self._calculate_vdot(best_5k_time, best_10k_time)
        return self._calculate_training_paces(vdot)
    
    def get_zone_for_workout(self, workout_type: str) -> dict:
        """Get recommended zone for a specific workout type."""
        zones = self.calculate_zones()

        workout_zones = {
            'Easy Run': zones['zone_2'],
            'Recovery Run': zones['zone_1'],
            'Long Run': zones['zone_2'],
            'Tempo Run': zones['zone_3'],
            'Threshold Run': zones['zone_4'],
            'Intervals': zones['zone_5'],
            'Hill Repeats': zones['zone_4'],
            'Fartlek': zones['zone_3'],
            'Rest': None
        }

        return workout_zones.get(workout_type, zones['zone_2'])

    def calculate_power_zones(self, ftp: int) -> dict:
        """
        Calculate 7-zone cycling power zones based on FTP (Coggan/Allen model).

        Args:
            ftp: Functional Threshold Power in watts

        Returns:
            Dictionary with power zone ranges
        """
        zones = {
            'zone_1': {
                'name': 'Recuperación Activa',
                'name_en': 'Active Recovery',
                'min_watts': 0,
                'max_watts': int(ftp * 0.55),
                'min_pct': 0,
                'max_pct': 55,
                'description': 'Recuperación, pedaleo suave'
            },
            'zone_2': {
                'name': 'Resistencia',
                'name_en': 'Endurance',
                'min_watts': int(ftp * 0.56),
                'max_watts': int(ftp * 0.75),
                'min_pct': 56,
                'max_pct': 75,
                'description': 'Base aeróbica, fondos'
            },
            'zone_3': {
                'name': 'Tempo',
                'name_en': 'Tempo',
                'min_watts': int(ftp * 0.76),
                'max_watts': int(ftp * 0.90),
                'min_pct': 76,
                'max_pct': 90,
                'description': 'Ritmo sostenible, esfuerzo moderado'
            },
            'zone_4': {
                'name': 'Umbral',
                'name_en': 'Threshold',
                'min_watts': int(ftp * 0.91),
                'max_watts': int(ftp * 1.05),
                'min_pct': 91,
                'max_pct': 105,
                'description': 'Umbral funcional, esfuerzo intenso sostenible ~1h'
            },
            'zone_5': {
                'name': 'VO2max',
                'name_en': 'VO2max',
                'min_watts': int(ftp * 1.06),
                'max_watts': int(ftp * 1.20),
                'min_pct': 106,
                'max_pct': 120,
                'description': 'Intervalos 3-8 min, máxima capacidad aeróbica'
            },
            'zone_6': {
                'name': 'Capacidad Anaeróbica',
                'name_en': 'Anaerobic Capacity',
                'min_watts': int(ftp * 1.21),
                'max_watts': int(ftp * 1.50),
                'min_pct': 121,
                'max_pct': 150,
                'description': 'Esfuerzos cortos 30s-3min'
            },
            'zone_7': {
                'name': 'Potencia Neuromuscular',
                'name_en': 'Neuromuscular Power',
                'min_watts': int(ftp * 1.51),
                'max_watts': int(ftp * 2.00),
                'min_pct': 151,
                'max_pct': 200,
                'description': 'Sprints máximos <30s'
            }
        }
        return zones

    def calculate_swim_zones(self, css_pace_100m: float) -> dict:
        """
        Calculate swimming training zones based on CSS (Critical Swim Speed).

        Args:
            css_pace_100m: CSS pace in seconds per 100m

        Returns:
            Dictionary with swim zone ranges in sec/100m
        """
        zones = {
            'zone_1': {
                'name': 'Recuperación',
                'min_pace': int(css_pace_100m * 1.20),
                'max_pace': int(css_pace_100m * 1.30),
                'description': 'Nado muy suave, técnica'
            },
            'zone_2': {
                'name': 'Resistencia Aeróbica',
                'min_pace': int(css_pace_100m * 1.10),
                'max_pace': int(css_pace_100m * 1.19),
                'description': 'Base aeróbica, fondos'
            },
            'zone_3': {
                'name': 'Tempo',
                'min_pace': int(css_pace_100m * 1.02),
                'max_pace': int(css_pace_100m * 1.09),
                'description': 'Ritmo sostenido, umbral aeróbico'
            },
            'zone_4': {
                'name': 'Umbral',
                'min_pace': int(css_pace_100m * 0.96),
                'max_pace': int(css_pace_100m * 1.01),
                'description': 'CSS, esfuerzo intenso sostenible'
            },
            'zone_5': {
                'name': 'VO2max',
                'min_pace': int(css_pace_100m * 0.88),
                'max_pace': int(css_pace_100m * 0.95),
                'description': 'Series rápidas, capacidad aeróbica máxima'
            }
        }
        return zones


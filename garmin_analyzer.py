"""
Garmin Data Analyzer
Parses and analyzes Garmin training history data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
from typing import Union


class GarminDataAnalyzer:
    """Analyze Garmin training history to determine current fitness status."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.required_columns = ['date', 'distance']
        self.optional_columns = ['duration', 'average_heart_rate', 'calories', 'activity_type']
    
    def load_csv(self, file_input: Union[str, bytes, StringIO], filter_running_only: bool = False) -> pd.DataFrame:
        """
        Load and parse Garmin CSV file. Supports Spanish and English formats.

        Args:
            file_input: File path, bytes, or StringIO object
            filter_running_only: If True, only keep running activities. Default False (all activities)

        Returns:
            Parsed DataFrame with standardized columns
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            last_error = None

            for encoding in encodings:
                try:
                    if hasattr(file_input, 'read'):
                        file_input.seek(0)
                        df = pd.read_csv(file_input, encoding=encoding)
                    else:
                        df = pd.read_csv(file_input, encoding=encoding)
                    break
                except UnicodeDecodeError as e:
                    last_error = e
                    continue
                except Exception as e:
                    last_error = e
                    continue

            if df is None:
                raise ValueError(f"No se pudo decodificar el archivo CSV: {last_error}")

            if df.empty:
                raise ValueError("El archivo CSV está vacío")

            # First, standardize column names (lowercase, remove accents, replace spaces)
            df.columns = df.columns.str.lower().str.strip()
            # Remove accents for easier matching
            accent_map = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'}
            for accent, letter in accent_map.items():
                df.columns = df.columns.str.replace(accent, letter, regex=False)
            df.columns = df.columns.str.replace(' ', '_', regex=False)

            # Now handle duplicate column names AFTER standardization
            # This catches cases where "Duration" and "Duración" both become "duration"
            seen_cols = {}
            new_cols = []
            for i, col in enumerate(df.columns):
                if col in seen_cols:
                    # Mark duplicate columns for removal (add suffix)
                    new_cols.append(f"{col}_dup_{seen_cols[col]}")
                    seen_cols[col] += 1
                else:
                    new_cols.append(col)
                    seen_cols[col] = 1
            df.columns = new_cols

            # Remove duplicate columns (keep only original ones)
            cols_to_keep = [c for c in df.columns if '_dup_' not in c]
            df = df[cols_to_keep]

            # Map common Garmin column variations (Spanish and English)
            column_mappings = {
                # Date
                'fecha': 'date',
                # Distance
                'distancia': 'distance',
                # Duration
                'duracion': 'duration',
                'tiempo': 'duration',
                'time': 'duration',
                'elapsed_time': 'duration',
                'tiempo_transcurrido': 'duration',
                # Heart rate
                'fc_media': 'average_heart_rate',
                'frecuencia_cardiaca_media': 'average_heart_rate',
                'avg_hr': 'average_heart_rate',
                'avg_heart_rate': 'average_heart_rate',
                # Max heart rate
                'fc_maxima': 'max_heart_rate',
                'fc_max': 'max_heart_rate',
                'max_hr': 'max_heart_rate',
                # Calories
                'calorias': 'calories',
                # Activity type
                'tipo_de_actividad': 'activity_type',
                'tipo': 'activity_type',
                'actividad': 'activity_type',
                'activity': 'activity_type',
                # Title
                'titulo': 'title',
                # Cadence
                'cadencia_media_de_carrera': 'avg_running_cadence',
                'avg_run_cadence': 'avg_running_cadence',
                # Speed/Pace
                'velocidad_media': 'avg_speed',
                'avg_speed': 'avg_speed',
                'ritmo_medio': 'avg_pace',
                'avg_pace': 'avg_pace',
                # Power
                'potencia_media': 'avg_power',
                'potencia_maxima': 'max_power',
                'avg_power': 'avg_power',
                'max_power': 'max_power',
                'normalized_power': 'normalized_power',
                'potencia_normalizada': 'normalized_power',
                'potencia_media_de_pedaleo': 'avg_power',
                'power': 'avg_power'
            }

            df.rename(columns=column_mappings, inplace=True)

            # Handle duplicates AGAIN after renaming (e.g., tiempo and tiempo_transcurrido both become duration)
            seen_cols = {}
            new_cols = []
            for col in df.columns:
                if col in seen_cols:
                    new_cols.append(f"{col}_dup_{seen_cols[col]}")
                    seen_cols[col] += 1
                else:
                    new_cols.append(col)
                    seen_cols[col] = 1
            df.columns = new_cols

            # Remove duplicate columns after renaming
            cols_to_keep = [c for c in df.columns if '_dup_' not in c]
            df = df[cols_to_keep]

            # Parse date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            else:
                raise ValueError("Columna de fecha no encontrada en el CSV. Columnas disponibles: " +
                               ", ".join(df.columns.tolist()))

            # Parse distance - handle Spanish decimal separator (comma)
            if 'distance' in df.columns:
                try:
                    # Convert comma decimal separator to dot
                    df['distance'] = df['distance'].astype(str).str.replace(',', '.', regex=False).str.replace('"', '', regex=False)
                    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
                    # Fill NaN with 0 for indoor activities without distance
                    df['distance'] = df['distance'].fillna(0)
                    # If values seem to be in meters (>1000), convert to km
                    mean_distance = float(df['distance'].mean())
                    if pd.notna(mean_distance) and mean_distance > 1000:
                        df['distance'] = df['distance'] / 1000
                except Exception as e:
                    df['distance'] = 0
            else:
                raise ValueError("Columna de distancia no encontrada en el CSV")

            # Parse duration
            if 'duration' in df.columns:
                try:
                    df['duration_minutes'] = df['duration'].apply(self._parse_duration)
                except Exception:
                    df['duration_minutes'] = 30  # Default 30 min

            # Parse heart rate - handle Spanish format and missing values (--)
            if 'average_heart_rate' in df.columns:
                try:
                    df['average_heart_rate'] = df['average_heart_rate'].astype(str).str.replace(',', '.', regex=False).str.replace('--', '', regex=False)
                    df['average_heart_rate'] = pd.to_numeric(df['average_heart_rate'], errors='coerce')
                except Exception:
                    df['average_heart_rate'] = np.nan

            # Parse calories - handle Spanish format (e.g., "1.265" means 1265)
            if 'calories' in df.columns:
                try:
                    df['calories'] = df['calories'].apply(self._parse_calories)
                except Exception:
                    df['calories'] = 0

            # Parse power columns
            for power_col in ['avg_power', 'max_power', 'normalized_power']:
                if power_col in df.columns:
                    try:
                        df[power_col] = df[power_col].astype(str).str.replace(',', '.', regex=False).str.replace('--', '', regex=False)
                        df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
                    except Exception:
                        df[power_col] = np.nan

            # Filter running activities only if requested
            if filter_running_only and 'activity_type' in df.columns:
                try:
                    running_keywords = ['running', 'run', 'carrera', 'correr', 'trail']
                    mask = df['activity_type'].astype(str).str.lower().str.contains('|'.join(running_keywords), na=False)
                    df = df[mask].copy()
                except Exception:
                    pass  # Keep all activities if filter fails

            # Remove rows with invalid dates
            df = df.dropna(subset=['date'])

            if df.empty:
                raise ValueError("No se encontraron actividades válidas con fechas correctas")

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            return df

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error inesperado al procesar el archivo CSV: {str(e)}")

    def _parse_calories(self, val) -> float:
        """Parse calories value handling Spanish format."""
        try:
            if pd.isna(val):
                return 0.0
            val_str = str(val).strip().replace('"', '')
            if val_str == '' or val_str == '--':
                return 0.0
            # Spanish format uses . as thousands separator
            if '.' in val_str and ',' not in val_str:
                # Check if it looks like thousands separator (e.g., "1.265")
                parts = val_str.split('.')
                if len(parts) == 2 and len(parts[1]) == 3:
                    val_str = val_str.replace('.', '')  # Remove thousands separator
            val_str = val_str.replace(',', '.')  # Convert decimal comma to dot
            return float(val_str)
        except Exception:
            return 0.0
    
    def _parse_duration(self, duration_str) -> float:
        """Parse duration string to minutes. Handles Spanish format with comma decimals."""
        if pd.isna(duration_str):
            return np.nan

        duration_str = str(duration_str).strip()
        # Handle Spanish decimal separator (comma) in seconds - e.g., "00:06:14,1"
        duration_str = duration_str.replace(',', '.')

        try:
            # Try HH:MM:SS or HH:MM:SS.f format
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 3:
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 60 + minutes + seconds / 60
                elif len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes + seconds / 60
            else:
                # Assume it's already in minutes
                return float(duration_str)
        except:
            return np.nan
    
    def analyze_fitness(self, df: pd.DataFrame, max_hr: int = 185,
                        resting_hr: int = 60, gender: str = 'male') -> dict:
        """
        Analyze training data to determine current fitness status.
        Analyzes ALL activities (running, cycling, etc.) for general fitness estimation.

        Args:
            df: DataFrame with training data (all activities)
            max_hr: User's maximum heart rate
            resting_hr: Resting heart rate (default 60)
            gender: Gender for TRIMP calculation ('male' or 'female')

        Returns:
            Dictionary with fitness metrics
        """
        try:
            # Default return values
            default_result = {
                'avg_weekly_distance': 0,
                'avg_weekly_duration': 0,
                'avg_weekly_calories': 0,
                'avg_pace': 0,
                'avg_heart_rate': 0,
                'fitness_level': 'Principiante',
                'training_load': 0,
                'recommended_increase': 10,
                'total_activities': 0,
                'activity_breakdown': {},
                'longest_run': 0,
                'weeks_analyzed': 0
            }

            if df is None or len(df) == 0:
                return default_result

            # Get last 8 weeks of data
            end_date = df['date'].max()
            start_date = end_date - timedelta(weeks=8)
            recent_df = df[df['date'] >= start_date].copy()

            if len(recent_df) == 0:
                return default_result

            # Separate running activities for pace calculation
            running_df = None
            if 'activity_type' in recent_df.columns:
                try:
                    running_keywords = ['running', 'run', 'carrera', 'correr', 'trail']
                    running_mask = recent_df['activity_type'].astype(str).str.lower().str.contains('|'.join(running_keywords), na=False)
                    running_df = recent_df[running_mask].copy()
                except Exception:
                    running_df = None

            # Weekly statistics for ALL activities
            agg_dict = {'distance': 'sum'}
            if 'duration_minutes' in recent_df.columns:
                agg_dict['duration_minutes'] = 'sum'
            if 'calories' in recent_df.columns:
                agg_dict['calories'] = 'sum'

            weekly_stats = recent_df.groupby(pd.Grouper(key='date', freq='W')).agg(agg_dict).reset_index()

            avg_weekly_distance = float(weekly_stats['distance'].mean()) if len(weekly_stats) > 0 else 0
            avg_weekly_duration = float(weekly_stats['duration_minutes'].mean()) if 'duration_minutes' in weekly_stats.columns and len(weekly_stats) > 0 else 0
            avg_weekly_calories = float(weekly_stats['calories'].mean()) if 'calories' in weekly_stats.columns and len(weekly_stats) > 0 else 0

            # Handle NaN values
            avg_weekly_distance = 0 if pd.isna(avg_weekly_distance) else avg_weekly_distance
            avg_weekly_duration = 0 if pd.isna(avg_weekly_duration) else avg_weekly_duration
            avg_weekly_calories = 0 if pd.isna(avg_weekly_calories) else avg_weekly_calories

            # Calculate average pace (only for running activities with distance > 0)
            avg_pace = 0
            if running_df is not None and len(running_df) > 0 and 'duration_minutes' in running_df.columns:
                try:
                    running_with_dist = running_df[running_df['distance'] > 0]
                    if len(running_with_dist) > 0:
                        total_distance = float(running_with_dist['distance'].sum())
                        total_duration = float(running_with_dist['duration_minutes'].sum())
                        if total_distance > 0:
                            avg_pace = total_duration / total_distance
                except Exception:
                    avg_pace = 0

            # Count activities by type
            activity_counts = {}
            if 'activity_type' in recent_df.columns:
                try:
                    activity_counts = recent_df['activity_type'].value_counts().to_dict()
                except Exception:
                    activity_counts = {}

            # Determine fitness level based on weekly training volume (duration + intensity)
            # Use a combination of duration and heart rate based training load
            training_load = self._calculate_training_load(recent_df, max_hr, resting_hr, gender)

            # Fitness level based on weekly training load and duration
            if avg_weekly_duration >= 420:  # 7+ hours/week
                fitness_level = 'Avanzado'
            elif avg_weekly_duration >= 300:  # 5+ hours/week
                fitness_level = 'Intermedio-Avanzado'
            elif avg_weekly_duration >= 180:  # 3+ hours/week
                fitness_level = 'Intermedio'
            elif avg_weekly_duration >= 90:  # 1.5+ hours/week
                fitness_level = 'Principiante-Intermedio'
            else:
                fitness_level = 'Principiante'

            # Recommended weekly increase percentage (based on current fitness)
            if fitness_level in ['Principiante', 'Principiante-Intermedio']:
                recommended_increase = 10
            elif fitness_level == 'Intermedio':
                recommended_increase = 8
            else:
                recommended_increase = 5

            # Calculate average heart rate across all activities
            avg_hr = 0
            if 'average_heart_rate' in recent_df.columns:
                try:
                    valid_hr = recent_df['average_heart_rate'].dropna()
                    avg_hr = float(valid_hr.mean()) if len(valid_hr) > 0 else 0
                    avg_hr = 0 if pd.isna(avg_hr) else avg_hr
                except Exception:
                    avg_hr = 0

            # Calculate longest run
            longest_run = 0
            if running_df is not None and len(running_df) > 0:
                try:
                    longest_run = float(running_df['distance'].max())
                    longest_run = 0 if pd.isna(longest_run) else longest_run
                except Exception:
                    longest_run = 0

            return {
                'avg_weekly_distance': avg_weekly_distance,
                'avg_weekly_duration': avg_weekly_duration,
                'avg_weekly_calories': avg_weekly_calories,
                'avg_pace': avg_pace,
                'avg_heart_rate': avg_hr,
                'fitness_level': fitness_level,
                'training_load': training_load,
                'recommended_increase': recommended_increase,
                'total_activities': len(recent_df),
                'activity_breakdown': activity_counts,
                'longest_run': longest_run,
                'weeks_analyzed': len(weekly_stats)
            }

        except Exception as e:
            # Return safe defaults on any error
            return {
                'avg_weekly_distance': 0,
                'avg_weekly_duration': 0,
                'avg_weekly_calories': 0,
                'avg_pace': 0,
                'avg_heart_rate': 0,
                'fitness_level': 'Principiante',
                'training_load': 0,
                'recommended_increase': 10,
                'total_activities': 0,
                'activity_breakdown': {},
                'longest_run': 0,
                'weeks_analyzed': 0
            }

    def _calculate_training_load(self, df: pd.DataFrame, max_hr: int,
                                   resting_hr: int = 60, gender: str = 'male') -> float:
        """
        Calculate simplified training load (TRIMP-like metric).
        Uses Banister's TRIMP formula with gender-specific coefficients.

        Args:
            df: DataFrame with training data
            max_hr: Maximum heart rate
            resting_hr: Resting heart rate (default 60)
            gender: Gender for coefficient selection ('male' or 'female')

        Returns:
            Weekly training load in arbitrary units
        """
        if len(df) == 0:
            return 0

        # Validar que max_hr > resting_hr para evitar división por cero
        hr_reserve_total = max_hr - resting_hr
        if hr_reserve_total <= 0:
            hr_reserve_total = 125  # Valor por defecto seguro (185 - 60)

        total_load = 0

        # Coeficientes de Banister según género
        if gender.lower() == 'female':
            k_coefficient = 0.86  # Coeficiente multiplicador para mujeres
            y_factor = 1.67      # Exponente para mujeres
        else:
            k_coefficient = 0.64  # Coeficiente multiplicador para hombres
            y_factor = 1.92      # Exponente para hombres

        for _, row in df.iterrows():
            duration = row.get('duration_minutes', 30)  # Default 30 min if not available
            if pd.isna(duration) or duration <= 0:
                duration = 30

            if pd.notna(row.get('average_heart_rate')):
                # Calculate intensity factor based on HR using Banister formula
                hr = row['average_heart_rate']
                hr_reserve_pct = (hr - resting_hr) / hr_reserve_total
                hr_reserve_pct = max(0.0, min(1.0, hr_reserve_pct))

                # Fórmula TRIMP de Banister completa
                intensity_factor = hr_reserve_pct * k_coefficient * np.exp(y_factor * hr_reserve_pct)
            else:
                # Estimación mejorada sin HR basada en duración y distancia
                distance = row.get('distance', 0)
                if pd.isna(distance):
                    distance = 0

                # Estimar intensidad basada en velocidad si hay distancia
                if distance > 0 and duration > 0:
                    speed_kmh = distance / (duration / 60)  # km/h
                    # Asumir intensidad moderada (~65% HRR) para velocidades típicas
                    # Velocidad 8-12 km/h = intensidad moderada
                    estimated_hr_pct = min(0.85, max(0.5, 0.4 + speed_kmh * 0.035))
                    intensity_factor = estimated_hr_pct * k_coefficient * np.exp(y_factor * estimated_hr_pct)
                else:
                    # Sin datos suficientes, asumir intensidad baja-moderada
                    intensity_factor = 0.5 * k_coefficient * np.exp(y_factor * 0.5)

            total_load += duration * intensity_factor

        # Calculate weeks in data
        if len(df) > 0:
            date_range = (df['date'].max() - df['date'].min()).days
            weeks = max(1, date_range / 7)
        else:
            weeks = 1

        return total_load / weeks

    def generate_sample_data(self, weeks: int = 8) -> pd.DataFrame:
        """
        Generate sample training data for testing.

        Args:
            weeks: Number of weeks of data to generate

        Returns:
            DataFrame with sample training data
        """
        data = []
        end_date = datetime.now()

        for week in range(weeks):
            week_start = end_date - timedelta(weeks=week)

            # Generate 3-5 runs per week
            num_runs = np.random.randint(3, 6)

            for run in range(num_runs):
                run_date = week_start - timedelta(days=np.random.randint(0, 7))

                # Vary distance based on week progression
                base_distance = 5 + week * 0.5  # Progressive increase
                distance = base_distance + np.random.uniform(-2, 3)
                distance = max(3, distance)

                # One long run per week
                if run == 0:
                    distance = base_distance * 1.5 + np.random.uniform(0, 2)

                # Duration based on ~6 min/km pace with variation
                pace = 5.5 + np.random.uniform(-0.5, 1)
                duration_minutes = distance * pace

                # Heart rate based on pace and randomness
                avg_hr = 140 + np.random.randint(-15, 25)

                # Calories estimate
                calories = int(distance * 70 + np.random.randint(-50, 100))

                data.append({
                    'date': run_date,
                    'distance': round(distance, 2),
                    'duration': f"{int(duration_minutes//60)}:{int(duration_minutes%60):02d}:00",
                    'average_heart_rate': avg_hr,
                    'calories': calories,
                    'activity_type': 'Running'
                })

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Add duration_minutes column
        df['duration_minutes'] = df['duration'].apply(self._parse_duration)

        return df

    def calculate_fitness_score(self, df: pd.DataFrame, max_hr: int = 185,
                                 resting_hr: int = 60, age: int = 35,
                                 gender: str = 'male') -> dict:
        """
        Calcula un Fitness Score completo basado en datos de entrenamiento.
        Utiliza una versión mejorada del TRIMP (Training Impulse) junto con
        métricas de carga aguda/crónica y comparación con percentiles poblacionales.

        Args:
            df: DataFrame con datos de entrenamiento
            max_hr: Frecuencia cardíaca máxima
            resting_hr: Frecuencia cardíaca en reposo
            age: Edad del usuario
            gender: Género ('male' o 'female')

        Returns:
            Diccionario con métricas de fitness score
        """
        try:
            if df is None or len(df) == 0:
                return self._empty_fitness_score()

            # Ordenar por fecha
            df = df.sort_values('date').copy()

            # Asegurarse de que duration_minutes existe
            if 'duration_minutes' not in df.columns:
                df['duration_minutes'] = 30  # Default

            # Calcular TRIMP para cada actividad
            df['trimp'] = df.apply(lambda row: self._calculate_trimp_row(
                row, max_hr, resting_hr, gender
            ), axis=1)

            # Calcular evolución semanal del fitness
            agg_dict = {'trimp': 'sum', 'distance': 'sum'}
            if 'duration_minutes' in df.columns:
                agg_dict['duration_minutes'] = 'sum'

            weekly_trimp = df.groupby(pd.Grouper(key='date', freq='W')).agg(agg_dict).reset_index()

            # Renombrar columnas dinámicamente
            new_cols = ['week', 'weekly_trimp']
            if 'distance' in agg_dict:
                new_cols.append('weekly_distance')
            if 'duration_minutes' in agg_dict:
                new_cols.append('weekly_duration')

            # Ajustar número de columnas
            if len(weekly_trimp.columns) == 4:
                weekly_trimp.columns = ['week', 'weekly_trimp', 'weekly_distance', 'weekly_duration']
            elif len(weekly_trimp.columns) == 3:
                weekly_trimp.columns = ['week', 'weekly_trimp', 'weekly_distance']
            else:
                weekly_trimp.columns = ['week', 'weekly_trimp']

            # Calcular Fitness (CTL - Chronic Training Load) y Fatigue (ATL - Acute Training Load)
            # Usando promedios móviles exponenciales
            ctl_days = 42  # 6 semanas
            atl_days = 7   # 1 semana

            daily_trimp = df.groupby(pd.Grouper(key='date', freq='D'))['trimp'].sum().reset_index()
            daily_trimp.columns = ['date', 'daily_trimp']

            # Rellenar días sin entrenamiento con 0
            if len(daily_trimp) > 1:
                try:
                    date_range = pd.date_range(start=daily_trimp['date'].min(),
                                               end=daily_trimp['date'].max(), freq='D')
                    daily_trimp = daily_trimp.set_index('date').reindex(date_range).fillna(0).reset_index()
                    daily_trimp.columns = ['date', 'daily_trimp']
                except Exception:
                    pass  # Keep original if reindex fails

            # Calcular CTL (Fitness) y ATL (Fatigue)
            daily_trimp['ctl'] = daily_trimp['daily_trimp'].ewm(span=ctl_days, adjust=False).mean()
            daily_trimp['atl'] = daily_trimp['daily_trimp'].ewm(span=atl_days, adjust=False).mean()
            daily_trimp['tsb'] = daily_trimp['ctl'] - daily_trimp['atl']  # Training Stress Balance

            # Valores actuales (con conversión a float para evitar Series ambiguity)
            current_ctl = float(daily_trimp['ctl'].iloc[-1]) if len(daily_trimp) > 0 else 0
            current_atl = float(daily_trimp['atl'].iloc[-1]) if len(daily_trimp) > 0 else 0
            current_tsb = float(daily_trimp['tsb'].iloc[-1]) if len(daily_trimp) > 0 else 0

            # Handle NaN
            current_ctl = 0 if pd.isna(current_ctl) else current_ctl
            current_atl = 0 if pd.isna(current_atl) else current_atl
            current_tsb = 0 if pd.isna(current_tsb) else current_tsb

            # Calcular Fitness Score (0-100) basado en CTL normalizado
            fitness_score = self._normalize_fitness_score(current_ctl, age, gender)

            # Determinar percentil poblacional
            percentile = self._get_population_percentile(fitness_score, age, gender)

            # Evolución temporal para gráfico
            evolution_data = daily_trimp[['date', 'ctl', 'atl', 'tsb']].copy()
            evolution_data['fitness_score'] = evolution_data['ctl'].apply(
                lambda x: self._normalize_fitness_score(float(x) if pd.notna(x) else 0, age, gender)
            )

            # Estadísticas de progreso
            weeks_data = len(weekly_trimp)
            progress_pct = 0
            if weeks_data >= 4:
                try:
                    first_half = float(weekly_trimp['weekly_trimp'].iloc[:weeks_data//2].mean())
                    second_half = float(weekly_trimp['weekly_trimp'].iloc[weeks_data//2:].mean())
                    if first_half > 0 and not pd.isna(first_half) and not pd.isna(second_half):
                        progress_pct = ((second_half - first_half) / first_half * 100)
                except Exception:
                    progress_pct = 0

            # Calcular avg_weekly_trimp de forma segura
            avg_weekly_trimp = 0
            if len(weekly_trimp) > 0:
                try:
                    avg_weekly_trimp = float(weekly_trimp['weekly_trimp'].mean())
                    avg_weekly_trimp = 0 if pd.isna(avg_weekly_trimp) else avg_weekly_trimp
                except Exception:
                    avg_weekly_trimp = 0

            # Calcular total_trimp de forma segura
            total_trimp = 0
            try:
                total_trimp = float(df['trimp'].sum())
                total_trimp = 0 if pd.isna(total_trimp) else total_trimp
            except Exception:
                total_trimp = 0

            return {
                'fitness_score': round(fitness_score, 1),
                'ctl': round(current_ctl, 1),
                'atl': round(current_atl, 1),
                'tsb': round(current_tsb, 1),
                'form_status': self._get_form_status(current_tsb),
                'percentile': percentile,
                'percentile_label': self._get_percentile_label(percentile),
                'total_trimp': round(total_trimp, 0),
                'avg_weekly_trimp': round(avg_weekly_trimp, 1),
                'progress_pct': round(progress_pct, 1),
                'evolution_data': evolution_data.to_dict('records'),
                'weekly_data': weekly_trimp.to_dict('records'),
                'age': age,
                'gender': gender
            }

        except Exception:
            return self._empty_fitness_score()

    def _calculate_trimp_row(self, row, max_hr: int, resting_hr: int, gender: str) -> float:
        """
        Calcula TRIMP (Training Impulse) para una fila de datos.
        Usa la fórmula de Banister con coeficientes específicos por género.

        Referencia: Banister EW (1991). Modeling elite athletic performance.
        - Hombres: TRIMP = duración × HRreserve × 0.64 × e^(1.92 × HRreserve)
        - Mujeres: TRIMP = duración × HRreserve × 0.86 × e^(1.67 × HRreserve)
        """
        duration = row.get('duration_minutes', 30)
        if pd.isna(duration) or duration <= 0:
            duration = 30

        # Validar que max_hr > resting_hr para evitar división por cero
        hr_reserve_total = max_hr - resting_hr
        if hr_reserve_total <= 0:
            hr_reserve_total = 125  # Valor por defecto seguro (185 - 60)

        # Coeficientes de Banister según género (CORREGIDO)
        if gender.lower() == 'female':
            k_coefficient = 0.86  # Coeficiente multiplicador para mujeres
            y_factor = 1.67      # Exponente para mujeres
        else:
            k_coefficient = 0.64  # Coeficiente multiplicador para hombres
            y_factor = 1.92      # Exponente para hombres

        if pd.notna(row.get('average_heart_rate')):
            hr = row['average_heart_rate']
            # Calcular fracción de reserva de FC (método Karvonen)
            hr_reserve = (hr - resting_hr) / hr_reserve_total
            hr_reserve = max(0.0, min(1.0, hr_reserve))

            # Fórmula TRIMP de Banister con coeficientes correctos
            trimp = duration * hr_reserve * k_coefficient * np.exp(y_factor * hr_reserve)
        else:
            # Estimación mejorada sin HR basada en duración y distancia
            distance = row.get('distance', 0)
            if pd.isna(distance):
                distance = 0

            # Estimar intensidad basada en velocidad si hay distancia
            if distance > 0 and duration > 0:
                speed_kmh = distance / (duration / 60)  # km/h
                # Estimar %HRR basado en velocidad (modelo simplificado)
                # Velocidad 6 km/h ≈ 50% HRR, 12 km/h ≈ 85% HRR
                estimated_hr_pct = min(0.90, max(0.45, 0.35 + speed_kmh * 0.04))
                trimp = duration * estimated_hr_pct * k_coefficient * np.exp(y_factor * estimated_hr_pct)
            else:
                # Sin datos suficientes, asumir intensidad moderada (~55% HRR)
                estimated_hr_pct = 0.55
                trimp = duration * estimated_hr_pct * k_coefficient * np.exp(y_factor * estimated_hr_pct)

        return trimp

    def _normalize_fitness_score(self, ctl: float, age: int, gender: str) -> float:
        """
        Normaliza CTL a un score de 0-100.

        Basado en valores de referencia de CTL para diferentes niveles de atletas:
        - Principiante: CTL 10-25
        - Recreacional: CTL 25-50
        - Aficionado activo: CTL 50-80
        - Competidor amateur: CTL 80-120
        - Élite: CTL 120+

        La normalización usa una función logarítmica-lineal que es más realista
        que una sigmoide pura, especialmente para valores bajos de CTL.
        """
        # Validar CTL
        if pd.isna(ctl) or ctl < 0:
            ctl = 0

        # Factor de edad: declive fisiológico de ~0.7% por año después de 30
        # Basado en estudios de VO2max y capacidad aeróbica
        age_factor = 1.0 - (max(0, age - 30) * 0.007)
        age_factor = max(0.65, age_factor)  # Mínimo 65% a edad avanzada

        # NO penalizar por género - el CTL ya refleja la capacidad individual
        # Las diferencias fisiológicas ya están capturadas en los coeficientes TRIMP

        # Valores de referencia de CTL ajustados por edad
        # CTL de 40 = score 50 para un adulto de 30 años
        reference_ctl = 40 * age_factor

        # Usar función de normalización híbrida:
        # - Para CTL bajo (0-30): crecimiento más rápido
        # - Para CTL medio (30-80): crecimiento lineal
        # - Para CTL alto (80+): saturación gradual

        if ctl <= 0:
            normalized = 0
        elif ctl <= reference_ctl * 0.5:
            # Zona baja: crecimiento acelerado (0-25 score aprox)
            normalized = 25 * (ctl / (reference_ctl * 0.5))
        elif ctl <= reference_ctl * 1.5:
            # Zona media: crecimiento lineal (25-65 score aprox)
            normalized = 25 + 40 * ((ctl - reference_ctl * 0.5) / reference_ctl)
        else:
            # Zona alta: saturación logarítmica (65-100 score)
            excess = ctl - reference_ctl * 1.5
            # Usar logaritmo para saturación suave
            normalized = 65 + 35 * (1 - np.exp(-excess / (reference_ctl * 1.5)))

        return min(100, max(0, normalized))

    def _get_population_percentile(self, fitness_score: float, age: int, gender: str) -> int:
        """
        Calcula el percentil poblacional basado en fitness score.

        Usa la aproximación de Abramowitz y Stegun para la función error,
        que tiene precisión de ~1.5×10^-7 (mucho mejor que la aproximación tanh).

        Distribución basada en datos epidemiológicos de fitness poblacional:
        - Media poblacional: ~35 (la mayoría de la población es sedentaria)
        - Desviación estándar: ~18 (alta variabilidad)
        """
        # Validar fitness_score
        if pd.isna(fitness_score):
            fitness_score = 0

        # Ajustar por edad: personas mayores con mismo score están mejor posicionadas
        # ya que la población general también declina con edad
        age_adjustment = max(0, (age - 35) * 0.4)
        adjusted_score = fitness_score + age_adjustment

        # Parámetros de distribución poblacional
        mean_pop = 35  # Media más baja (población general es sedentaria)
        std_pop = 18   # Mayor variabilidad

        # Calcular z-score
        z = (adjusted_score - mean_pop) / std_pop

        # Aproximación de Abramowitz y Stegun para CDF normal
        # Φ(z) = 1 - φ(z)(b1*t + b2*t² + b3*t³ + b4*t⁴ + b5*t⁵) para z >= 0
        # donde t = 1/(1 + p*z), p = 0.2316419
        # Precisión: |error| < 1.5×10^-7

        def normal_cdf(x):
            """Aproximación precisa de la CDF normal estándar."""
            if x < -8:
                return 0.0
            if x > 8:
                return 1.0

            # Coeficientes de Abramowitz y Stegun
            p = 0.2316419
            b1 = 0.319381530
            b2 = -0.356563782
            b3 = 1.781477937
            b4 = -1.821255978
            b5 = 1.330274429

            # Calcular para |x|
            abs_x = abs(x)
            t = 1.0 / (1.0 + p * abs_x)

            # φ(x) = exp(-x²/2) / √(2π)
            phi = np.exp(-0.5 * abs_x * abs_x) / np.sqrt(2 * np.pi)

            # Aproximación polinomial
            cdf_complement = phi * (b1*t + b2*t**2 + b3*t**3 + b4*t**4 + b5*t**5)

            if x >= 0:
                return 1.0 - cdf_complement
            else:
                return cdf_complement

        percentile = normal_cdf(z) * 100

        return int(min(99, max(1, percentile)))

    def _get_percentile_label(self, percentile: int) -> str:
        """Devuelve una etiqueta descriptiva para el percentil."""
        if percentile >= 95:
            return "Élite"
        elif percentile >= 85:
            return "Excelente"
        elif percentile >= 70:
            return "Muy Bueno"
        elif percentile >= 55:
            return "Bueno"
        elif percentile >= 40:
            return "Promedio"
        elif percentile >= 25:
            return "Por debajo del promedio"
        else:
            return "Principiante"

    def _get_form_status(self, tsb: float) -> str:
        """Determina el estado de forma basado en TSB."""
        if tsb > 25:
            return "Muy descansado - Riesgo de pérdida de forma"
        elif tsb > 10:
            return "Fresco - Óptimo para competir"
        elif tsb > -10:
            return "Forma óptima - Buen balance"
        elif tsb > -25:
            return "Fatigado - Necesita recuperación"
        else:
            return "Muy fatigado - Riesgo de sobreentrenamiento"

    def _empty_fitness_score(self) -> dict:
        """Devuelve un diccionario vacío de fitness score."""
        return {
            'fitness_score': 0,
            'ctl': 0,
            'atl': 0,
            'tsb': 0,
            'form_status': 'Sin datos',
            'percentile': 0,
            'percentile_label': 'Sin datos',
            'total_trimp': 0,
            'avg_weekly_trimp': 0,
            'progress_pct': 0,
            'evolution_data': [],
            'weekly_data': [],
            'age': 35,
            'gender': 'male'
        }


class PowerProfileAnalyzer:
    """
    Analiza el perfil de potencia del atleta basado en watts y watts/kg.

    Utiliza las tablas de referencia de Coggan/Allen para clasificar el nivel
    del ciclista basado en FTP (Functional Threshold Power) y potencia relativa.

    Referencias:
    - Allen & Coggan: "Training and Racing with a Power Meter"
    - British Cycling Power Profile
    """

    # Tablas de referencia de potencia por categoría (watts/kg para FTP de 60 min)
    # Basadas en Allen & Coggan Power Profile
    POWER_CATEGORIES_MALE = {
        'World Class': {'min_wpkg': 5.80, 'description': 'Profesional World Tour'},
        'Exceptional': {'min_wpkg': 5.25, 'description': 'Profesional Continental'},
        'Excellent': {'min_wpkg': 4.70, 'description': 'Elite Amateur / Cat 1'},
        'Very Good': {'min_wpkg': 4.15, 'description': 'Cat 2 / Competidor Serio'},
        'Good': {'min_wpkg': 3.60, 'description': 'Cat 3 / Aficionado Activo'},
        'Moderate': {'min_wpkg': 3.05, 'description': 'Cat 4 / Recreacional Fit'},
        'Fair': {'min_wpkg': 2.50, 'description': 'Cat 5 / Principiante'},
        'Untrained': {'min_wpkg': 0.0, 'description': 'Sin entrenamiento específico'}
    }

    POWER_CATEGORIES_FEMALE = {
        'World Class': {'min_wpkg': 5.10, 'description': 'Profesional World Tour'},
        'Exceptional': {'min_wpkg': 4.60, 'description': 'Profesional Continental'},
        'Excellent': {'min_wpkg': 4.10, 'description': 'Elite Amateur / Cat 1'},
        'Very Good': {'min_wpkg': 3.60, 'description': 'Cat 2 / Competidor Serio'},
        'Good': {'min_wpkg': 3.15, 'description': 'Cat 3 / Aficionado Activo'},
        'Moderate': {'min_wpkg': 2.70, 'description': 'Cat 4 / Recreacional Fit'},
        'Fair': {'min_wpkg': 2.20, 'description': 'Cat 5 / Principiante'},
        'Untrained': {'min_wpkg': 0.0, 'description': 'Sin entrenamiento específico'}
    }

    # Tablas de potencia absoluta por duración (5s, 1min, 5min, FTP)
    # Percentiles para hombres (watts)
    POWER_PERCENTILES_MALE = {
        '5s': [(1800, 99), (1500, 95), (1200, 85), (1000, 70), (800, 50), (600, 30), (400, 10)],
        '1min': [(700, 99), (600, 95), (500, 85), (420, 70), (350, 50), (280, 30), (200, 10)],
        '5min': [(450, 99), (400, 95), (360, 85), (320, 70), (280, 50), (240, 30), (180, 10)],
        'ftp': [(400, 99), (350, 95), (310, 85), (275, 70), (240, 50), (200, 30), (150, 10)]
    }

    # Percentiles para mujeres (watts)
    POWER_PERCENTILES_FEMALE = {
        '5s': [(1200, 99), (1000, 95), (850, 85), (700, 70), (550, 50), (420, 30), (300, 10)],
        '1min': [(500, 99), (430, 95), (370, 85), (310, 70), (260, 50), (210, 30), (150, 10)],
        '5min': [(330, 99), (290, 95), (260, 85), (230, 70), (200, 50), (170, 30), (130, 10)],
        'ftp': [(300, 99), (265, 95), (235, 85), (205, 70), (180, 50), (150, 30), (110, 10)]
    }

    def __init__(self, weight: float, gender: str = 'male', age: int = 35):
        """
        Inicializa el analizador de perfil de potencia.

        Args:
            weight: Peso del atleta en kg
            gender: Género ('male' o 'female')
            age: Edad del atleta
        """
        self.weight = max(40, min(150, weight))  # Validar peso razonable
        self.gender = gender.lower()
        self.age = age

        # Seleccionar tablas según género
        if self.gender == 'female':
            self.power_categories = self.POWER_CATEGORIES_FEMALE
            self.power_percentiles = self.POWER_PERCENTILES_FEMALE
        else:
            self.power_categories = self.POWER_CATEGORIES_MALE
            self.power_percentiles = self.POWER_PERCENTILES_MALE

    def extract_power_from_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Extrae datos de potencia de un DataFrame de Garmin.

        Analiza las actividades de ciclismo para extraer:
        - FTP estimado (basado en potencia normalizada o promedio de esfuerzos largos)
        - Potencia máxima (sprint)
        - Historial de potencia por actividad

        Args:
            df: DataFrame con datos de Garmin

        Returns:
            Diccionario con datos de potencia extraídos
        """
        result = {
            'has_power_data': False,
            'total_activities_with_power': 0,
            'cycling_activities': 0,
            'estimated_ftp': 0,
            'max_power': 0,
            'avg_power_all': 0,
            'power_history': [],
            'best_efforts': {
                '5s': 0,
                '1min': 0,
                '5min': 0,
                'ftp': 0
            }
        }

        if df is None or df.empty:
            return result

        # Verificar si hay columnas de potencia
        power_cols = ['avg_power', 'max_power', 'normalized_power']
        available_power_cols = [col for col in power_cols if col in df.columns]

        if not available_power_cols:
            return result

        # Filtrar actividades de ciclismo
        cycling_keywords = ['cycling', 'biking', 'bike', 'ciclismo', 'bici', 'road', 'mtb',
                           'indoor_cycling', 'virtual_ride', 'spinning', 'indoor cycling']

        if 'activity_type' in df.columns:
            cycling_mask = df['activity_type'].astype(str).str.lower().str.contains(
                '|'.join(cycling_keywords), na=False
            )
            cycling_df = df[cycling_mask].copy()
        else:
            # Si no hay tipo de actividad, usar todas las que tengan potencia
            cycling_df = df.copy()

        # Filtrar solo actividades con datos de potencia válidos
        if 'avg_power' in cycling_df.columns:
            cycling_df = cycling_df[cycling_df['avg_power'].notna() & (cycling_df['avg_power'] > 0)]

        if cycling_df.empty:
            return result

        result['has_power_data'] = True
        result['total_activities_with_power'] = len(cycling_df)
        result['cycling_activities'] = len(cycling_df)

        # Extraer potencia máxima
        if 'max_power' in cycling_df.columns:
            max_power_values = cycling_df['max_power'].dropna()
            if not max_power_values.empty:
                result['max_power'] = float(max_power_values.max())
                # Estimar potencia de 5s como el máximo registrado
                result['best_efforts']['5s'] = result['max_power']

        # Potencia promedio general
        if 'avg_power' in cycling_df.columns:
            avg_power_values = cycling_df['avg_power'].dropna()
            if not avg_power_values.empty:
                result['avg_power_all'] = float(avg_power_values.mean())

        # Estimar FTP
        # Prioridad: normalized_power de actividades largas > avg_power de actividades largas
        if 'duration_minutes' in cycling_df.columns:
            # Actividades de más de 40 minutos
            long_activities = cycling_df[cycling_df['duration_minutes'] >= 40]

            if not long_activities.empty:
                if 'normalized_power' in long_activities.columns:
                    np_values = long_activities['normalized_power'].dropna()
                    if not np_values.empty and np_values.max() > 0:
                        # FTP ≈ 95% del mejor NP en actividades largas
                        result['estimated_ftp'] = float(np_values.max() * 0.95)
                        result['best_efforts']['ftp'] = result['estimated_ftp']

                if result['estimated_ftp'] == 0 and 'avg_power' in long_activities.columns:
                    # Usar el mejor promedio de potencia en actividades largas
                    avg_values = long_activities['avg_power'].dropna()
                    if not avg_values.empty:
                        # El mejor esfuerzo sostenido es aproximadamente el FTP
                        result['estimated_ftp'] = float(avg_values.max())
                        result['best_efforts']['ftp'] = result['estimated_ftp']

                # Estimar potencia de 5 minutos (110-120% del FTP típicamente)
                if result['estimated_ftp'] > 0:
                    result['best_efforts']['5min'] = result['estimated_ftp'] * 1.15
                    result['best_efforts']['1min'] = result['estimated_ftp'] * 1.50

        # Si no hay actividades largas, estimar FTP desde el promedio
        if result['estimated_ftp'] == 0 and result['avg_power_all'] > 0:
            # Estimación conservadora: FTP ≈ avg_power * 1.05 (asumiendo entrenamientos variados)
            result['estimated_ftp'] = result['avg_power_all'] * 1.05
            result['best_efforts']['ftp'] = result['estimated_ftp']

            if result['max_power'] > 0:
                # Estimar otros esfuerzos desde el máximo
                result['best_efforts']['5s'] = result['max_power']
                result['best_efforts']['1min'] = result['max_power'] * 0.55
                result['best_efforts']['5min'] = result['max_power'] * 0.40

        # Crear historial de potencia
        if 'date' in cycling_df.columns and 'avg_power' in cycling_df.columns:
            for _, row in cycling_df.iterrows():
                if pd.notna(row.get('avg_power')) and row['avg_power'] > 0:
                    entry = {
                        'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                        'avg_power': float(row['avg_power']),
                        'max_power': float(row.get('max_power', 0)) if pd.notna(row.get('max_power')) else 0,
                        'normalized_power': float(row.get('normalized_power', 0)) if pd.notna(row.get('normalized_power')) else 0,
                        'duration_minutes': float(row.get('duration_minutes', 0)) if pd.notna(row.get('duration_minutes')) else 0
                    }
                    if 'activity_type' in row:
                        entry['activity_type'] = str(row['activity_type'])
                    result['power_history'].append(entry)

        # Ordenar historial por fecha
        result['power_history'] = sorted(result['power_history'], key=lambda x: x['date'])

        return result

    def analyze_ftp(self, ftp_watts: float) -> dict:
        """
        Analiza el FTP (Functional Threshold Power) del atleta.

        Args:
            ftp_watts: FTP en watts (potencia sostenible durante ~1 hora)

        Returns:
            Diccionario con análisis completo del FTP
        """
        if ftp_watts <= 0:
            return self._empty_power_analysis()

        # Calcular watts/kg
        watts_per_kg = ftp_watts / self.weight

        # Determinar categoría
        category = self._get_power_category(watts_per_kg)

        # Calcular percentil
        percentile = self._calculate_power_percentile(ftp_watts, 'ftp')

        # Calcular zonas de potencia (basadas en FTP)
        power_zones = self._calculate_power_zones(ftp_watts)

        # Estimar VO2max desde FTP (aproximación)
        estimated_vo2max = self._estimate_vo2max_from_ftp(watts_per_kg)

        # Factor de edad (la potencia máxima declina ~1% por año después de 35)
        age_adjusted_percentile = self._adjust_percentile_for_age(percentile)

        return {
            'ftp_watts': round(ftp_watts, 0),
            'watts_per_kg': round(watts_per_kg, 2),
            'category': category['name'],
            'category_description': category['description'],
            'percentile': percentile,
            'age_adjusted_percentile': age_adjusted_percentile,
            'estimated_vo2max': round(estimated_vo2max, 1),
            'power_zones': power_zones,
            'next_category': self._get_next_category(watts_per_kg),
            'watts_to_next': self._watts_to_next_category(watts_per_kg),
            'training_recommendations': self._get_training_recommendations(category['name'])
        }

    def analyze_power_profile(self, power_5s: float = 0, power_1min: float = 0,
                               power_5min: float = 0, ftp: float = 0) -> dict:
        """
        Analiza el perfil de potencia completo del atleta.

        Args:
            power_5s: Potencia máxima de 5 segundos (sprint)
            power_1min: Potencia máxima de 1 minuto (anaeróbico)
            power_5min: Potencia máxima de 5 minutos (VO2max)
            ftp: Potencia umbral funcional (~1 hora)

        Returns:
            Diccionario con perfil de potencia completo
        """
        profile = {
            'weight': self.weight,
            'gender': self.gender,
            'age': self.age,
            'powers': {},
            'strengths': [],
            'weaknesses': [],
            'rider_type': 'Desconocido',
            'overall_score': 0
        }

        powers = {
            '5s': power_5s,
            '1min': power_1min,
            '5min': power_5min,
            'ftp': ftp
        }

        percentiles = []
        wpkg_values = {}

        for duration, watts in powers.items():
            if watts > 0:
                wpkg = watts / self.weight
                wpkg_values[duration] = wpkg
                pct = self._calculate_power_percentile(watts, duration)
                percentiles.append(pct)

                profile['powers'][duration] = {
                    'watts': round(watts, 0),
                    'watts_per_kg': round(wpkg, 2),
                    'percentile': pct
                }

        # Calcular score general
        if percentiles:
            profile['overall_score'] = round(sum(percentiles) / len(percentiles), 0)

        # Determinar tipo de ciclista basado en fortalezas relativas
        if len(wpkg_values) >= 3:
            profile['rider_type'] = self._determine_rider_type(wpkg_values)
            profile['strengths'], profile['weaknesses'] = self._analyze_strengths_weaknesses(
                profile['powers']
            )

        # Añadir análisis FTP si está disponible
        if ftp > 0:
            profile['ftp_analysis'] = self.analyze_ftp(ftp)

        return profile

    def _get_power_category(self, watts_per_kg: float) -> dict:
        """Determina la categoría de potencia basada en watts/kg."""
        for name, data in self.power_categories.items():
            if watts_per_kg >= data['min_wpkg']:
                return {'name': name, 'description': data['description']}
        return {'name': 'Untrained', 'description': 'Sin entrenamiento específico'}

    def _get_next_category(self, watts_per_kg: float) -> str:
        """Obtiene la siguiente categoría a alcanzar."""
        categories = list(self.power_categories.items())
        for i, (name, data) in enumerate(categories):
            if watts_per_kg >= data['min_wpkg']:
                if i > 0:
                    return categories[i-1][0]
                return "Ya estás en la categoría más alta"
        return categories[-2][0] if len(categories) > 1 else "Fair"

    def _watts_to_next_category(self, watts_per_kg: float) -> float:
        """Calcula los watts/kg necesarios para la siguiente categoría."""
        categories = list(self.power_categories.items())
        for i, (name, data) in enumerate(categories):
            if watts_per_kg >= data['min_wpkg']:
                if i > 0:
                    next_wpkg = categories[i-1][1]['min_wpkg']
                    return round((next_wpkg - watts_per_kg) * self.weight, 0)
                return 0
        return round((categories[-2][1]['min_wpkg'] - watts_per_kg) * self.weight, 0)

    def _calculate_power_percentile(self, watts: float, duration: str) -> int:
        """Calcula el percentil para una potencia dada."""
        if duration not in self.power_percentiles:
            return 50

        table = self.power_percentiles[duration]

        # Interpolar percentil
        for i, (power, pct) in enumerate(table):
            if watts >= power:
                if i == 0:
                    return min(99, pct + int((watts - power) / 50))
                # Interpolar entre este y el anterior
                prev_power, prev_pct = table[i-1]
                ratio = (watts - power) / (prev_power - power)
                return int(pct + ratio * (prev_pct - pct))

        # Por debajo del mínimo
        return max(1, table[-1][1] - 5)

    def _calculate_power_zones(self, ftp: float) -> dict:
        """
        Calcula las zonas de potencia basadas en FTP.
        Modelo de 7 zonas de Coggan.
        """
        return {
            'Z1 - Recuperación Activa': {
                'min': 0,
                'max': round(ftp * 0.55),
                'description': 'Recuperación, calentamiento'
            },
            'Z2 - Resistencia': {
                'min': round(ftp * 0.56),
                'max': round(ftp * 0.75),
                'description': 'Entrenamiento de base aeróbica'
            },
            'Z3 - Tempo': {
                'min': round(ftp * 0.76),
                'max': round(ftp * 0.90),
                'description': 'Ritmo sostenido, "sweetspot"'
            },
            'Z4 - Umbral': {
                'min': round(ftp * 0.91),
                'max': round(ftp * 1.05),
                'description': 'Esfuerzo de umbral, ~1 hora'
            },
            'Z5 - VO2max': {
                'min': round(ftp * 1.06),
                'max': round(ftp * 1.20),
                'description': 'Intervalos de 3-8 minutos'
            },
            'Z6 - Capacidad Anaeróbica': {
                'min': round(ftp * 1.21),
                'max': round(ftp * 1.50),
                'description': 'Intervalos de 30s-2min'
            },
            'Z7 - Potencia Neuromuscular': {
                'min': round(ftp * 1.51),
                'max': 9999,
                'description': 'Sprints máximos <30s'
            }
        }

    def _estimate_vo2max_from_ftp(self, watts_per_kg: float) -> float:
        """
        Estima VO2max desde watts/kg de FTP.
        Basado en la relación: VO2max ≈ (watts/kg × 10.8) + 7
        """
        # Fórmula aproximada basada en estudios de fisiología del ejercicio
        vo2max = (watts_per_kg * 10.8) + 7

        # Ajustar por género (las mujeres tienen ~10% menos VO2max para mismo w/kg)
        if self.gender == 'female':
            vo2max *= 0.90

        return max(20, min(90, vo2max))

    def _adjust_percentile_for_age(self, percentile: int) -> int:
        """Ajusta el percentil considerando la edad."""
        # La potencia máxima declina ~1% por año después de 35
        if self.age > 35:
            age_bonus = min(15, (self.age - 35) * 0.5)
            return min(99, int(percentile + age_bonus))
        return percentile

    def _determine_rider_type(self, wpkg_values: dict) -> str:
        """Determina el tipo de ciclista basado en el perfil de potencia."""
        if len(wpkg_values) < 3:
            return "Datos insuficientes"

        # Normalizar valores relativos al FTP
        ftp_wpkg = wpkg_values.get('ftp', 0)
        if ftp_wpkg <= 0:
            return "Necesita datos de FTP"

        ratios = {}
        if '5s' in wpkg_values:
            ratios['sprint'] = wpkg_values['5s'] / ftp_wpkg
        if '1min' in wpkg_values:
            ratios['anaerobic'] = wpkg_values['1min'] / ftp_wpkg
        if '5min' in wpkg_values:
            ratios['vo2max'] = wpkg_values['5min'] / ftp_wpkg

        # Clasificar tipo de ciclista
        sprint_ratio = ratios.get('sprint', 0)
        anaerobic_ratio = ratios.get('anaerobic', 0)
        vo2max_ratio = ratios.get('vo2max', 0)

        if sprint_ratio > 5.5:
            return "Sprinter"
        elif anaerobic_ratio > 2.0 and sprint_ratio > 4.5:
            return "Sprinter/Puncher"
        elif vo2max_ratio > 1.25 and anaerobic_ratio > 1.8:
            return "Puncher"
        elif vo2max_ratio > 1.20:
            return "Escalador/Atacante"
        elif ftp_wpkg > 4.0 and vo2max_ratio < 1.15:
            return "Rodador/Contrarrelojista"
        elif ftp_wpkg > 3.5:
            return "All-Rounder"
        else:
            return "En desarrollo"

    def _analyze_strengths_weaknesses(self, powers: dict) -> tuple:
        """Analiza fortalezas y debilidades del perfil."""
        strengths = []
        weaknesses = []

        if not powers:
            return strengths, weaknesses

        # Ordenar por percentil
        sorted_powers = sorted(powers.items(), key=lambda x: x[1].get('percentile', 0), reverse=True)

        for duration, data in sorted_powers[:2]:
            pct = data.get('percentile', 0)
            if pct >= 70:
                duration_names = {'5s': 'Sprint', '1min': 'Anaeróbico', '5min': 'VO2max', 'ftp': 'Resistencia'}
                strengths.append(duration_names.get(duration, duration))

        for duration, data in sorted_powers[-2:]:
            pct = data.get('percentile', 0)
            if pct < 50:
                duration_names = {'5s': 'Sprint', '1min': 'Anaeróbico', '5min': 'VO2max', 'ftp': 'Resistencia'}
                weaknesses.append(duration_names.get(duration, duration))

        return strengths, weaknesses

    def _get_training_recommendations(self, category: str) -> list:
        """Genera recomendaciones de entrenamiento basadas en la categoría."""
        recommendations = {
            'Untrained': [
                "Comenzar con 3-4 sesiones semanales de 30-60 minutos",
                "Enfocarse en construir base aeróbica (Z2)",
                "Incluir 1 sesión semanal de intervalos suaves"
            ],
            'Fair': [
                "Aumentar volumen gradualmente a 5-8 horas semanales",
                "Introducir intervalos de tempo (Z3) 1-2 veces por semana",
                "Trabajar en técnica de pedaleo y cadencia"
            ],
            'Moderate': [
                "Mantener 8-12 horas semanales de entrenamiento",
                "Incluir trabajo de umbral (Z4) 2 veces por semana",
                "Añadir intervalos de VO2max (Z5) semanalmente"
            ],
            'Good': [
                "Periodizar el entrenamiento con bloques específicos",
                "Incluir trabajo de fuerza en gimnasio",
                "Optimizar nutrición y recuperación"
            ],
            'Very Good': [
                "Considerar entrenamiento con coach certificado",
                "Análisis detallado de datos de potencia",
                "Trabajo específico en debilidades del perfil"
            ],
            'Excellent': [
                "Entrenamiento altamente estructurado y periodizado",
                "Campos de entrenamiento y competiciones regulares",
                "Optimización de todos los aspectos del rendimiento"
            ],
            'Exceptional': [
                "Entrenamiento profesional con equipo de soporte",
                "Análisis biomecánico y aerodinámico",
                "Gestión de carga y recuperación avanzada"
            ],
            'World Class': [
                "Mantener consistencia en el entrenamiento",
                "Gestión de calendario de competiciones",
                "Optimización marginal de todos los factores"
            ]
        }
        return recommendations.get(category, recommendations['Moderate'])

    def _empty_power_analysis(self) -> dict:
        """Devuelve un análisis vacío."""
        return {
            'ftp_watts': 0,
            'watts_per_kg': 0,
            'category': 'Sin datos',
            'category_description': 'Introduce tu FTP para analizar',
            'percentile': 0,
            'age_adjusted_percentile': 0,
            'estimated_vo2max': 0,
            'power_zones': {},
            'next_category': '',
            'watts_to_next': 0,
            'training_recommendations': []
        }

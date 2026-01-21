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
                'avg_pace': 'avg_pace'
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
    
    def analyze_fitness(self, df: pd.DataFrame, max_hr: int = 185) -> dict:
        """
        Analyze training data to determine current fitness status.
        Analyzes ALL activities (running, cycling, etc.) for general fitness estimation.

        Args:
            df: DataFrame with training data (all activities)
            max_hr: User's maximum heart rate

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
            training_load = self._calculate_training_load(recent_df, max_hr)

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

    def _calculate_training_load(self, df: pd.DataFrame, max_hr: int) -> float:
        """
        Calculate simplified training load (TRIMP-like metric).

        Args:
            df: DataFrame with training data
            max_hr: Maximum heart rate

        Returns:
            Weekly training load in arbitrary units
        """
        if len(df) == 0:
            return 0

        total_load = 0

        for _, row in df.iterrows():
            duration = row.get('duration_minutes', 30)  # Default 30 min if not available

            if pd.notna(row.get('average_heart_rate')):
                # Calculate intensity factor based on HR
                hr = row['average_heart_rate']
                hr_reserve_pct = (hr - 60) / (max_hr - 60)  # Assume 60 resting HR
                hr_reserve_pct = max(0, min(1, hr_reserve_pct))

                # Exponential weighting for higher intensities
                intensity_factor = hr_reserve_pct * np.exp(1.92 * hr_reserve_pct)
            else:
                # Use distance as proxy for load if HR not available
                distance = row.get('distance', 5)
                intensity_factor = 0.7 + (distance / 20) * 0.3  # Scale based on distance

            total_load += duration * intensity_factor

        # Calculate weeks in data
        weeks = (df['date'].max() - df['date'].min()).days / 7
        weeks = max(1, weeks)

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
        """Calcula TRIMP para una fila de datos."""
        duration = row.get('duration_minutes', 30)
        if pd.isna(duration):
            duration = 30

        if pd.notna(row.get('average_heart_rate')):
            hr = row['average_heart_rate']
            # Calcular fracción de reserva de FC
            hr_reserve = (hr - resting_hr) / (max_hr - resting_hr)
            hr_reserve = max(0.0, min(1.0, hr_reserve))

            # Factor de género (las mujeres tienen respuesta fisiológica ligeramente diferente)
            if gender.lower() == 'female':
                y_factor = 1.67
            else:
                y_factor = 1.92

            # Fórmula TRIMP de Banister
            trimp = duration * hr_reserve * 0.64 * np.exp(y_factor * hr_reserve)
        else:
            # Estimar TRIMP basado en distancia si no hay FC
            distance = row.get('distance', 5)
            if pd.isna(distance):
                distance = 5
            trimp = duration * 0.5 + distance * 5  # Aproximación conservadora

        return trimp

    def _normalize_fitness_score(self, ctl: float, age: int, gender: str) -> float:
        """Normaliza CTL a un score de 0-100."""
        # Valores de referencia por edad y género
        # Basados en estudios de atletas recreacionales
        age_factor = 1.0 - (max(0, age - 25) * 0.005)  # Declive de 0.5% por año después de 25
        gender_factor = 0.9 if gender.lower() == 'female' else 1.0

        # CTL típico para diferentes niveles (ajustado por factores)
        reference_ctl = 50 * age_factor * gender_factor  # CTL de referencia para score 50

        # Normalizar a escala 0-100 (usando función sigmoidea suave)
        normalized = 100 / (1 + np.exp(-0.05 * (ctl - reference_ctl)))
        return min(100, max(0, normalized))

    def _get_population_percentile(self, fitness_score: float, age: int, gender: str) -> int:
        """Calcula el percentil poblacional basado en fitness score."""
        # Distribución aproximada basada en datos epidemiológicos
        # La población general tiene fitness scores distribuidos normalmente
        # con media ~40 y desviación estándar ~15

        # Ajustar por edad (la población general también declina con edad)
        age_adjustment = max(0, (age - 35) * 0.3)
        adjusted_score = fitness_score + age_adjustment

        # Calcular percentil usando distribución normal aproximada (sin scipy)
        mean_pop = 40
        std_pop = 15

        # Aproximación de la CDF normal usando error function (disponible en numpy)
        z = (adjusted_score - mean_pop) / (std_pop * np.sqrt(2))
        # Usar aproximación de erf con numpy
        # erf(x) ≈ tanh(x * (1.202 + 0.166 * x^2)) para buena precisión
        erf_approx = np.tanh(z * (1.202 + 0.166 * z * z))
        percentile = (1 + erf_approx) / 2 * 100

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


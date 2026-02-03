"""Script para diagnosticar problemas en las métricas avanzadas."""
import sys
sys.path.insert(0, '.')
from garmin_analyzer import GarminDataAnalyzer
import numpy as np
import pandas as pd

# Cargar datos
a = GarminDataAnalyzer()
df = a.load_csv('Activities (15).csv')
print(f"Total actividades: {len(df)}")

# Revisar columnas de potencia
print("\n=== COLUMNAS DE POTENCIA ===")
power_cols = [c for c in df.columns if 'power' in c.lower() or 'potencia' in c.lower() or 'np' in c.lower()]
print(f"Columnas encontradas: {power_cols}")

cycling = df[df['activity_type'].str.lower().str.contains('cicl|bici', na=False)]
print(f"Actividades ciclismo: {len(cycling)}")
if len(cycling) > 0 and len(power_cols) > 0:
    print("\nMuestra datos ciclismo:")
    print(cycling[power_cols].head(5))

# Filtrar running con datos válidos
running = df[
    (df['activity_type'].str.lower().str.contains('carrera|run', na=False)) &
    (df['average_heart_rate'] > 0) &
    (df['distance'] > 0) &
    (df['duration_minutes'] > 0)
].copy()
print(f"Carreras válidas: {len(running)}")

max_hr = 185
resting_hr = 60

print("\n=== ANÁLISIS DE DATOS DE RUNNING ===")
print("-" * 80)

vo2_estimates = []
for i, (_, row) in enumerate(running.head(10).iterrows()):
    dist = row['distance']  # km
    dur = row['duration_minutes']  # min
    hr = row['average_heart_rate']  # bpm
    
    # Cálculos básicos
    speed_kmh = dist / (dur / 60)  # km/h
    speed_mmin = (dist * 1000) / dur  # m/min
    pace_min_km = dur / dist  # min/km
    hr_pct = hr / max_hr
    
    # EF correcto: velocidad en m/min / FC
    ef = speed_mmin / hr
    
    # VO2 fórmula actual (INCORRECTA): 15.3 * (speed_kmh / hr_pct)
    vo2_bad = 15.3 * (speed_kmh / hr_pct)
    
    # VO2 fórmula ACSM (running en llano): VO2 = 3.5 + (0.2 * velocity_m/min)
    vo2_acsm = 3.5 + (0.2 * speed_mmin)
    
    # VO2 desde ritmo (aproximación VDOT simplificada)
    # Para un ritmo de 6:00 min/km → VO2max ≈ 45
    # Para un ritmo de 5:00 min/km → VO2max ≈ 52
    # Fórmula aproximada: VO2max ≈ 210 / pace_min_km (para ritmos de carrera)
    vo2_pace = 210 / pace_min_km if pace_min_km > 0 else 0
    
    print(f"\nCarrera {i+1}:")
    print(f"  Distancia: {dist:.2f} km | Duración: {dur:.1f} min | HR: {hr:.0f} bpm")
    print(f"  Velocidad: {speed_kmh:.1f} km/h | Pace: {pace_min_km:.2f} min/km | HR%: {hr_pct:.1%}")
    print(f"  EF: {ef:.3f} (velocidad m/min / HR)")
    print(f"  VO2 fórmula actual (MAL): {vo2_bad:.1f} ml/kg/min")
    print(f"  VO2 ACSM (submáximo): {vo2_acsm:.1f} ml/kg/min")
    print(f"  VO2 desde pace (aproximado): {vo2_pace:.1f} ml/kg/min")
    
    # Guardar para estadísticas
    if 0.6 <= hr_pct <= 0.85:  # Zona aeróbica
        vo2_estimates.append({
            'bad': vo2_bad,
            'acsm': vo2_acsm,
            'pace': vo2_pace
        })

print("\n" + "=" * 80)
print("ESTADÍSTICAS DE VO2MAX ESTIMADO")
print("=" * 80)
if vo2_estimates:
    bad_vals = [v['bad'] for v in vo2_estimates]
    acsm_vals = [v['acsm'] for v in vo2_estimates]
    pace_vals = [v['pace'] for v in vo2_estimates]
    
    print(f"Fórmula actual (MAL):   P50={np.percentile(bad_vals, 50):.1f}  P75={np.percentile(bad_vals, 75):.1f}")
    print(f"Fórmula ACSM:           P50={np.percentile(acsm_vals, 50):.1f}  P75={np.percentile(acsm_vals, 75):.1f}")
    print(f"Fórmula desde pace:     P50={np.percentile(pace_vals, 50):.1f}  P75={np.percentile(pace_vals, 75):.1f}")

# Verificar EF actual
print("\n" + "=" * 80)
print("ANÁLISIS DE EFFICIENCY FACTOR")
print("=" * 80)
metrics = a.calculate_advanced_metrics(df, max_hr=185, resting_hr=60, age=35, gender='male')
print(f"EF reportado: {metrics.get('efficiency_factor', 0):.3f}")
print(f"Decoupling reportado: {metrics.get('decoupling', 0):.1f}%")
print(f"VO2max reportado: {metrics.get('vo2max_estimated', 0):.1f}")

# Mostrar todas las métricas
print("\n" + "=" * 80)
print("TODAS LAS MÉTRICAS AVANZADAS")
print("=" * 80)
for key, value in metrics.items():
    if key not in ['alerts', 'race_predictions']:
        print(f"{key}: {value}")

print("\n" + "=" * 80)
print("PREDICCIONES DE CARRERA")
print("=" * 80)
preds = metrics.get('race_predictions', {})
for key, value in preds.items():
    print(f"{key}: {value}")


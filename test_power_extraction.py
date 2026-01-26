"""Test power extraction from DataFrame."""
import pandas as pd
from garmin_analyzer import PowerProfileAnalyzer

# Crear datos de prueba con potencia
data = {
    'date': pd.date_range('2024-01-01', periods=10, freq='D'),
    'activity_type': ['cycling'] * 10,
    'avg_power': [180, 190, 185, 200, 195, 210, 205, 220, 215, 225],
    'max_power': [450, 480, 470, 500, 490, 520, 510, 550, 540, 560],
    'duration_minutes': [60, 55, 65, 70, 45, 80, 75, 90, 50, 85]
}
df = pd.DataFrame(data)

analyzer = PowerProfileAnalyzer(weight=70, gender='male', age=35)
result = analyzer.extract_power_from_dataframe(df)

print('Has power data:', result['has_power_data'])
print('Activities with power:', result['total_activities_with_power'])
print('Estimated FTP:', result['estimated_ftp'])
print('Max Power:', result['max_power'])
print('Best efforts:', result['best_efforts'])
print('History entries:', len(result['power_history']))


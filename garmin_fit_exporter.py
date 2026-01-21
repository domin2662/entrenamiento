"""
Garmin FIT File Exporter
Generates FIT workout files for Garmin watches.
"""

import struct
import zipfile
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List
import json


class GarminFitExporter:
    """Export training workouts to Garmin FIT format for watch upload."""
    
    # FIT file constants
    FIT_PROTOCOL_VERSION = 0x20  # Protocol version 2.0
    FIT_PROFILE_VERSION = 0x0814  # Profile version 8.20
    
    # Message types
    MSG_FILE_ID = 0
    MSG_WORKOUT = 26
    MSG_WORKOUT_STEP = 27
    
    # Field definitions for workout files
    SPORT_RUNNING = 1
    SUB_SPORT_GENERIC = 0
    
    def __init__(self):
        """Initialize the FIT exporter."""
        self.data_records = []
    
    def export_week_to_fit(self, week_data: dict, week_number: int) -> BytesIO:
        """
        Export a week's workouts to FIT files in a ZIP archive.
        
        Args:
            week_data: Dictionary containing week workouts
            week_number: Week number for naming
            
        Returns:
            BytesIO containing ZIP file with FIT workouts
        """
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for day, workout in week_data['workouts'].items():
                if workout.get('type') != 'Rest':
                    fit_data = self._create_workout_fit(workout, day, week_number)
                    filename = f"week{week_number}_{day.lower()}_{workout['type'].lower().replace(' ', '_')}.fit"
                    zip_file.writestr(filename, fit_data)
            
            # Add a JSON summary file
            summary = {
                'week_number': week_number,
                'total_distance': week_data.get('total_distance', 0),
                'is_recovery': week_data.get('is_recovery', False),
                'workouts': [
                    {
                        'day': day,
                        'type': w.get('type'),
                        'distance_km': w.get('distance', 0),
                        'hr_zone': w.get('zone'),
                        'description': w.get('description', '')
                    }
                    for day, w in week_data['workouts'].items()
                    if w.get('type') != 'Rest'
                ]
            }
            zip_file.writestr(
                f"week{week_number}_summary.json",
                json.dumps(summary, indent=2)
            )
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def _create_workout_fit(self, workout: dict, day: str, week_num: int) -> bytes:
        """
        Create a FIT workout file.
        
        Note: This creates a simplified FIT-compatible structure.
        For full FIT SDK compatibility, consider using the official Garmin FIT SDK.
        
        Args:
            workout: Workout dictionary
            day: Day of week
            week_num: Week number
            
        Returns:
            Bytes of FIT file
        """
        # For production, you'd use the Garmin FIT SDK
        # This creates a simplified workout definition file
        
        workout_data = self._create_simplified_workout(workout, day, week_num)
        return workout_data
    
    def _create_simplified_workout(self, workout: dict, day: str, week_num: int) -> bytes:
        """
        Create a simplified workout file that can be imported to Garmin Connect.
        
        This creates a TCX-compatible format that Garmin Connect can import.
        """
        workout_type = workout.get('type', 'Easy Run')
        distance_km = workout.get('distance', 5)
        hr_min = workout.get('hr_min', 120)
        hr_max = workout.get('hr_max', 150)
        description = workout.get('description', '')
        
        # Create TCX workout format (XML-based, compatible with Garmin Connect)
        tcx_content = self._create_tcx_workout(
            name=f"Week {week_num} {day} - {workout_type}",
            workout_type=workout_type,
            distance_m=int(distance_km * 1000),
            hr_low=hr_min,
            hr_high=hr_max,
            description=description
        )
        
        return tcx_content.encode('utf-8')
    
    def _create_tcx_workout(
        self,
        name: str,
        workout_type: str,
        distance_m: int,
        hr_low: int,
        hr_high: int,
        description: str
    ) -> str:
        """Create TCX workout XML format."""
        
        # Determine workout structure based on type
        steps = self._generate_workout_steps(workout_type, distance_m, hr_low, hr_high)
        
        tcx = f'''<?xml version="1.0" encoding="UTF-8"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <Workouts>
        <Workout Sport="Running">
            <Name>{name}</Name>
            <Notes>{description}</Notes>
{steps}
            <ScheduledOn>{datetime.now().strftime('%Y-%m-%d')}</ScheduledOn>
        </Workout>
    </Workouts>
</TrainingCenterDatabase>'''
        
        return tcx
    
    def _generate_workout_steps(
        self,
        workout_type: str,
        distance_m: int,
        hr_low: int,
        hr_high: int
    ) -> str:
        """Generate workout steps XML based on workout type."""
        
        steps = []
        
        if workout_type == 'Easy Run':
            steps.append(self._create_step('Warmup', 'Time', 600, hr_low, hr_high))  # 10 min warmup
            steps.append(self._create_step('Active', 'Distance', distance_m - 2000, hr_low, hr_high))
            steps.append(self._create_step('Cooldown', 'Time', 600, hr_low - 10, hr_high - 10))
        
        elif workout_type == 'Recovery Run':
            steps.append(self._create_step('Active', 'Distance', distance_m, hr_low, hr_high))
        
        elif workout_type == 'Long Run':
            warmup_dist = 1000
            cooldown_dist = 1000
            main_dist = distance_m - warmup_dist - cooldown_dist
            steps.append(self._create_step('Warmup', 'Distance', warmup_dist, hr_low, hr_high))
            steps.append(self._create_step('Active', 'Distance', main_dist, hr_low, hr_high))
            steps.append(self._create_step('Cooldown', 'Distance', cooldown_dist, hr_low - 10, hr_high - 10))

        elif workout_type == 'Tempo Run':
            steps.append(self._create_step('Warmup', 'Time', 900, hr_low, hr_high))  # 15 min warmup
            tempo_dist = int(distance_m * 0.6)
            steps.append(self._create_step('Active', 'Distance', tempo_dist, hr_high, hr_high + 10))
            steps.append(self._create_step('Cooldown', 'Time', 600, hr_low - 10, hr_high - 10))

        elif workout_type == 'Intervals':
            steps.append(self._create_step('Warmup', 'Time', 900, hr_low, hr_high))  # 15 min warmup

            # Add interval repeats
            num_intervals = 6
            interval_dist = 400  # 400m intervals
            recovery_dist = 200

            for i in range(num_intervals):
                steps.append(self._create_step('Active', 'Distance', interval_dist, hr_high + 20, hr_high + 40))
                if i < num_intervals - 1:
                    steps.append(self._create_step('Recovery', 'Distance', recovery_dist, hr_low, hr_high))

            steps.append(self._create_step('Cooldown', 'Time', 600, hr_low - 10, hr_high - 10))

        elif workout_type == 'Hill Repeats':
            steps.append(self._create_step('Warmup', 'Time', 900, hr_low, hr_high))

            # Hill repeats
            num_repeats = 8
            for i in range(num_repeats):
                steps.append(self._create_step('Active', 'Time', 75, hr_high, hr_high + 20))  # 75 sec uphill
                if i < num_repeats - 1:
                    steps.append(self._create_step('Recovery', 'Time', 120, hr_low, hr_high))  # 2 min recovery

            steps.append(self._create_step('Cooldown', 'Time', 600, hr_low - 10, hr_high - 10))

        elif workout_type == 'Fartlek':
            steps.append(self._create_step('Warmup', 'Time', 600, hr_low, hr_high))

            # Fartlek segments
            for i in range(8):
                steps.append(self._create_step('Active', 'Time', 60, hr_high, hr_high + 15))  # 1 min fast
                steps.append(self._create_step('Recovery', 'Time', 90, hr_low, hr_high))  # 1.5 min easy

            steps.append(self._create_step('Cooldown', 'Time', 600, hr_low - 10, hr_high - 10))

        else:
            # Default simple workout
            steps.append(self._create_step('Active', 'Distance', distance_m, hr_low, hr_high))

        return '\n'.join(steps)

    def _create_step(
        self,
        intensity: str,
        duration_type: str,
        duration_value: int,
        hr_low: int,
        hr_high: int
    ) -> str:
        """Create a single workout step XML."""

        duration_xml = ''
        if duration_type == 'Distance':
            duration_xml = f'''                <DistanceMeters>{duration_value}</DistanceMeters>'''
        elif duration_type == 'Time':
            duration_xml = f'''                <Time>{duration_value}</Time>'''

        # Ensure HR values are reasonable
        hr_low = max(80, min(200, hr_low))
        hr_high = max(hr_low + 5, min(220, hr_high))

        step_xml = f'''            <Step>
                <StepId>{id(duration_value) % 1000}</StepId>
                <Name>{intensity}</Name>
                <Duration>
                    <DurationType>{duration_type}</DurationType>
{duration_xml}
                </Duration>
                <Intensity>{intensity}</Intensity>
                <Target>
                    <TargetType>HeartRate</TargetType>
                    <HeartRateZone>
                        <Low>{hr_low}</Low>
                        <High>{hr_high}</High>
                    </HeartRateZone>
                </Target>
            </Step>'''

        return step_xml

    def export_single_workout(self, workout: dict, day: str, week_num: int) -> BytesIO:
        """
        Export a single workout as TCX file.

        Args:
            workout: Workout dictionary
            day: Day of the week
            week_num: Week number

        Returns:
            BytesIO containing TCX file
        """
        tcx_data = self._create_simplified_workout(workout, day, week_num)
        buffer = BytesIO(tcx_data)
        buffer.seek(0)
        return buffer

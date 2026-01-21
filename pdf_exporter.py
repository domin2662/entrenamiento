"""
PDF Exporter Module - Álvaro Domingo - Entrenamiento Cideam.es
Exportación de planes de entrenamiento a PDF con diseño profesional.
"""

from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import calendar


class TrainingPlanPDFExporter:
    """Exportador de planes de entrenamiento a PDF con diseño profesional."""

    # Branding
    BRAND_NAME = "Álvaro Domingo"
    BRAND_SUBTITLE = "Entrenamiento Cideam.es"
    BRAND_URL = "www.cideam.es"

    # Colores corporativos (RGB normalizado 0-1)
    COLORS = {
        'primary': (0.13, 0.59, 0.95),      # Azul principal #2196F3
        'primary_dark': (0.10, 0.46, 0.82),  # Azul oscuro #1976D2
        'secondary': (0.30, 0.69, 0.31),     # Verde #4CAF50
        'accent': (1.0, 0.60, 0.0),          # Naranja #FF9800
        'dark': (0.18, 0.20, 0.21),          # Gris oscuro #2E3336
        'text': (0.26, 0.26, 0.26),          # Texto principal
        'text_light': (0.46, 0.46, 0.46),    # Texto secundario
        'background': (0.98, 0.98, 0.98),    # Fondo claro
        'white': (1.0, 1.0, 1.0),
        # Colores pastel para tablas (versiones suaves de los colores de marca)
        'pastel_blue': (0.88, 0.94, 0.99),       # #E1F0FD - Azul pastel suave
        'pastel_green': (0.91, 0.96, 0.91),      # #E8F5E8 - Verde pastel suave
        'pastel_orange': (1.0, 0.95, 0.88),      # #FFF3E0 - Naranja pastel suave
        'pastel_purple': (0.95, 0.91, 0.97),     # #F3E8F8 - Morado pastel suave
        'pastel_gray': (0.96, 0.96, 0.97),       # #F5F5F7 - Gris pastel suave
        'table_header': (0.13, 0.59, 0.95),      # Azul para headers de tabla
        'table_border': (0.85, 0.87, 0.90),      # Gris suave para bordes
        'table_alt_row': (0.97, 0.98, 0.99),     # Gris muy claro para filas alternadas
    }

    # Colores para tipos de entrenamiento (RGB 0-255)
    WORKOUT_COLORS = {
        'Rest': (189, 189, 189),
        'Easy Run': (76, 175, 80),
        'Tempo Run': (255, 152, 0),
        'Intervals': (244, 67, 54),
        'Long Run': (33, 150, 243),
        'Recovery Run': (139, 195, 74),
        'Hill Repeats': (156, 39, 176),
        'Fartlek': (0, 188, 212)
    }

    # Colores para zonas de entrenamiento
    ZONE_COLORS = {
        1: (76, 175, 80),     # Verde - Recuperación
        2: (139, 195, 74),    # Verde claro - Aeróbico
        3: (255, 193, 7),     # Amarillo - Tempo
        4: (255, 87, 34),     # Naranja - Umbral
        5: (244, 67, 54),     # Rojo - VO2 Máx
    }

    WORKOUT_TRANSLATIONS = {
        'Rest': 'Descanso',
        'Easy Run': 'Rodaje Suave',
        'Tempo Run': 'Series Tempo',
        'Intervals': 'Intervalos',
        'Long Run': 'Tirada Larga',
        'Recovery Run': 'Recuperación',
        'Hill Repeats': 'Cuestas',
        'Fartlek': 'Fartlek'
    }

    DAYS_ES = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

    MONTH_NAMES_ES = [
        '', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
        'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
    ]

    def __init__(self, training_plan: dict, athlete_profile: dict = None,
                 training_zones: dict = None, pace_zones: dict = None):
        """
        Inicializa el exportador PDF.

        Args:
            training_plan: Plan de entrenamiento generado
            athlete_profile: Perfil del atleta
            training_zones: Zonas de FC calculadas
            pace_zones: Zonas de ritmo calculadas
        """
        self.plan = training_plan
        self.profile = athlete_profile or {}
        self.zones = training_zones or {}
        self.pace_zones = pace_zones or {}
        self._build_calendar_data()
        self._calculate_pace_zones()

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
                'workouts_count': 0
            }

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
                    'day_name': day_name
                }

                if workout.get('type') != 'Rest':
                    week_summary['workouts_count'] += 1

            self.weekly_summaries.append(week_summary)

    def _calculate_pace_zones(self):
        """Calcula zonas de ritmo usando fórmulas de Jack Daniels VDOT corregidas."""
        # Si ya tenemos pace_zones válidas, usarlas
        if self.pace_zones and len(self.pace_zones) >= 5:
            return

        # Obtener datos del perfil
        vo2_max = self.profile.get('vo2_max', 45)
        best_5k = self.profile.get('best_5k_time')
        best_10k = self.profile.get('best_10k_time')

        # Calcular VDOT usando fórmulas mejoradas de Jack Daniels
        vdot = self._calculate_vdot_from_times(best_5k, best_10k, vo2_max)

        # Calcular ritmos de entrenamiento basados en VDOT
        paces = self._calculate_training_paces_from_vdot(vdot)

        def format_pace(pace_decimal):
            """Convierte ritmo decimal a formato MM:SS."""
            pace_decimal = max(2.5, min(12.0, pace_decimal))
            minutes = int(pace_decimal)
            seconds = int((pace_decimal - minutes) * 60)
            return f"{minutes}:{seconds:02d}"

        # pace_min = ritmo más rápido (número menor), pace_max = ritmo más lento (número mayor)
        self.pace_zones = {
            'zone_1': {
                'name': 'Recuperación',
                'pace_min': format_pace(paces['recovery']),  # Más rápido del rango
                'pace_max': format_pace(paces['recovery'] * 1.12),  # Más lento (12% más)
                'description': 'Muy fácil, conversación fluida'
            },
            'zone_2': {
                'name': 'Aeróbico / Easy',
                'pace_min': format_pace(paces['easy'] * 0.97),  # Rápido
                'pace_max': format_pace(paces['easy'] * 1.05),  # Lento
                'description': 'Rodaje suave, puedes hablar'
            },
            'zone_3': {
                'name': 'Tempo / Maratón',
                'pace_min': format_pace(paces['marathon'] * 0.98),  # Rápido
                'pace_max': format_pace(paces['marathon'] * 1.03),  # Lento
                'description': 'Ritmo controlado, frases cortas'
            },
            'zone_4': {
                'name': 'Umbral / Threshold',
                'pace_min': format_pace(paces['threshold'] * 0.97),  # Rápido
                'pace_max': format_pace(paces['threshold'] * 1.03),  # Lento
                'description': 'Duro pero sostenible ~1h'
            },
            'zone_5': {
                'name': 'VO2 Máx / Intervalos',
                'pace_min': format_pace(paces['interval'] * 0.95),  # Rápido (repeticiones)
                'pace_max': format_pace(paces['interval'] * 1.03),  # Lento (intervalos largos)
                'description': 'Muy duro, series de 3-5 min'
            }
        }

    def _calculate_vdot_from_times(self, best_5k: str, best_10k: str, vo2_max: float) -> float:
        """
        Calcula VDOT usando interpolación de tablas de Jack Daniels.
        Referencia: Daniels' Running Formula (3rd Edition)
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

        if best_5k:
            try:
                parts = str(best_5k).split(':')
                total_minutes = int(parts[0]) + int(parts[1]) / 60
                vdot = self._interpolate_vdot(total_minutes, vdot_table_5k)
            except:
                pass

        if best_10k and vdot is None:
            try:
                parts = str(best_10k).split(':')
                total_minutes = int(parts[0]) + int(parts[1]) / 60
                vdot = self._interpolate_vdot(total_minutes, vdot_table_10k)
            except:
                pass

        if vdot is None:
            vdot = vo2_max * 0.90
            vdot = max(30, min(70, vdot))

        return vdot

    def _interpolate_vdot(self, time_minutes: float, table: list) -> float:
        """Interpola VDOT desde una tabla de tiempos."""
        if time_minutes >= table[0][0]:
            return table[0][1]
        if time_minutes <= table[-1][0]:
            return table[-1][1]

        for i in range(len(table) - 1):
            t1, v1 = table[i]
            t2, v2 = table[i + 1]
            if t2 <= time_minutes <= t1:
                ratio = (t1 - time_minutes) / (t1 - t2)
                return v1 + ratio * (v2 - v1)

        return 40

    def _calculate_training_paces_from_vdot(self, vdot: float) -> dict:
        """
        Calcula ritmos de entrenamiento basados en VDOT usando tablas de Jack Daniels.

        Tabla de ritmos por VDOT (min/km):
        VDOT 35: Easy=6:40-7:18, Marathon=6:02, Threshold=5:41, Interval=5:14
        VDOT 40: Easy=5:54-6:26, Marathon=5:20, Threshold=5:01, Interval=4:38
        """
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

        return {
            'recovery': max(5.5, min(9.0, easy_pace * 1.10)),
            'easy': max(4.5, min(8.0, easy_pace)),
            'marathon': max(3.8, min(7.5, interpolate_pace(marathon_table, vdot))),
            'threshold': max(3.5, min(7.0, interpolate_pace(threshold_table, vdot))),
            'interval': max(3.2, min(6.5, interpolate_pace(interval_table, vdot))),
            'repetition': max(3.0, min(6.0, interpolate_pace(rep_table, vdot))),
        }

    def generate_pdf(self, start_week: int = 1, end_week: int = None,
                     include_details: bool = True) -> bytes:
        """
        Genera el PDF del plan de entrenamiento.

        Args:
            start_week: Semana inicial a incluir
            end_week: Semana final a incluir (None = todas)
            include_details: Si incluir detalles de entrenamientos

        Returns:
            bytes: Contenido del PDF
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm, mm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak, Image, HRFlowable
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
            from reportlab.pdfgen import canvas
        except ImportError:
            raise ImportError("Por favor instala reportlab: pip install reportlab")

        buffer = BytesIO()

        # Colores corporativos
        primary_color = colors.Color(*self.COLORS['primary'])
        dark_color = colors.Color(*self.COLORS['dark'])
        text_color = colors.Color(*self.COLORS['text'])

        # Footer con branding
        def add_footer(canvas_obj, doc):
            canvas_obj.saveState()
            # Línea superior del footer
            canvas_obj.setStrokeColor(primary_color)
            canvas_obj.setLineWidth(1)
            canvas_obj.line(1.5*cm, 1.2*cm, A4[0] - 1.5*cm, 1.2*cm)
            # Texto del footer
            canvas_obj.setFont('Helvetica', 8)
            canvas_obj.setFillColor(colors.Color(*self.COLORS['text_light']))
            canvas_obj.drawString(1.5*cm, 0.8*cm, f"{self.BRAND_NAME} - {self.BRAND_SUBTITLE}")
            canvas_obj.drawRightString(A4[0] - 1.5*cm, 0.8*cm, f"{self.BRAND_URL} | Página {doc.page}")
            canvas_obj.restoreState()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=1.5*cm,
            bottomMargin=2*cm  # Más espacio para el footer
        )

        # Estilos mejorados
        styles = getSampleStyleSheet()

        # Título principal - Grande y llamativo
        styles.add(ParagraphStyle(
            name='Title_ES',
            parent=styles['Title'],
            fontSize=32,
            spaceAfter=10,
            spaceBefore=0,
            alignment=TA_CENTER,
            textColor=dark_color,
            fontName='Helvetica-Bold'
        ))

        # Subtítulo de marca
        styles.add(ParagraphStyle(
            name='Brand_ES',
            parent=styles['Normal'],
            fontSize=14,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=primary_color,
            fontName='Helvetica-Bold'
        ))

        # Encabezados de sección
        styles.add(ParagraphStyle(
            name='Heading_ES',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=15,
            spaceBefore=20,
            textColor=dark_color,
            fontName='Helvetica-Bold',
            borderPadding=(0, 0, 5, 0)
        ))

        # Subencabezados
        styles.add(ParagraphStyle(
            name='SubHeading_ES',
            parent=styles['Heading2'],
            fontSize=13,
            spaceAfter=10,
            spaceBefore=15,
            textColor=primary_color,
            fontName='Helvetica-Bold'
        ))

        # Texto normal
        styles.add(ParagraphStyle(
            name='Normal_ES',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            textColor=text_color,
            leading=14
        ))

        # Texto pequeño
        styles.add(ParagraphStyle(
            name='Small_ES',
            parent=styles['Normal'],
            fontSize=8,
            spaceAfter=4,
            textColor=colors.Color(*self.COLORS['text_light'])
        ))

        # Texto centrado
        styles.add(ParagraphStyle(
            name='Center_ES',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=text_color
        ))

        # Contenido
        story = []

        # Portada
        story.extend(self._create_cover_page(styles, Paragraph, Spacer, Table, TableStyle, colors, HRFlowable))
        story.append(PageBreak())

        # Resumen del plan
        story.extend(self._create_plan_summary(styles, Paragraph, Spacer, Table, TableStyle, colors, HRFlowable))
        story.append(PageBreak())

        # Zonas de entrenamiento (FC + Ritmo)
        if self.zones or self.pace_zones:
            story.extend(self._create_zones_page(styles, Paragraph, Spacer, Table, TableStyle, colors, HRFlowable))
            story.append(PageBreak())

        # Semanas de entrenamiento
        end_week = end_week or len(self.plan.get('weeks', []))
        weeks_to_include = [w for w in self.plan.get('weeks', [])
                          if start_week <= w['week_number'] <= end_week]

        for week in weeks_to_include:
            story.extend(self._create_week_page(
                week, styles, Paragraph, Spacer, Table, TableStyle, colors, include_details, HRFlowable
            ))
            story.append(PageBreak())

        # Construir PDF con footer
        doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
        buffer.seek(0)
        return buffer.getvalue()

    def _create_cover_page(self, styles, Paragraph, Spacer, Table, TableStyle, colors, HRFlowable):
        """Crea la página de portada con diseño profesional."""
        from reportlab.lib.units import cm
        elements = []

        primary_color = colors.Color(*self.COLORS['primary'])
        dark_color = colors.Color(*self.COLORS['dark'])
        secondary_color = colors.Color(*self.COLORS['secondary'])

        # Espaciado superior
        elements.append(Spacer(1, 0.8*cm))

        # ========== LOGO SEPARADO DEL TÍTULO ==========
        # Logo en su propio contenedor con fondo de marca
        logo_table = Table(
            [[Paragraph('<font size="48" color="#2196F3">🏃</font>', styles['Title_ES'])]],
            colWidths=[80]
        )
        logo_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.Color(*self.COLORS['pastel_blue'])),
            ('BOX', (0, 0), (-1, -1), 2, primary_color),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(logo_table)
        elements.append(Spacer(1, 0.4*cm))

        # ========== TÍTULO PRINCIPAL (SEPARADO DEL LOGO) ==========
        title_content = [
            [Paragraph("Plan de Entrenamiento<br/>Personalizado", styles['Title_ES'])],
            [Spacer(1, 0.3*cm)],
            [HRFlowable(width="50%", thickness=2, color=primary_color,
                        spaceBefore=5, spaceAfter=8, hAlign='CENTER')],
            [Paragraph(f"<b>{self.BRAND_NAME}</b><br/><font size='10'>{self.BRAND_SUBTITLE}</font>",
                       styles['Brand_ES'])],
        ]
        title_table = Table(title_content, colWidths=[450])
        title_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1.5, primary_color),
            ('BACKGROUND', (0, 0), (-1, -1), colors.Color(*self.COLORS['pastel_blue'])),
            ('TOPPADDING', (0, 0), (-1, -1), 18),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
            ('LEFTPADDING', (0, 0), (-1, -1), 25),
            ('RIGHTPADDING', (0, 0), (-1, -1), 25),
        ]))
        elements.append(title_table)
        elements.append(Spacer(1, 0.8*cm))

        # ========== INFORMACIÓN DEL OBJETIVO ==========
        target_dist = self.plan.get('target_distance', 21.1)
        target_date = self.plan.get('target_date', datetime.now())
        if isinstance(target_date, datetime):
            target_date_str = target_date.strftime('%Y-%m-%d')
        else:
            target_date_str = str(target_date)

        dist_names = {10: '10K', 15: '15K', 21.1: 'Media Maratón', 42.2: 'Maratón'}
        dist_name = dist_names.get(target_dist, f'{target_dist}K')

        goal_data = [
            ['🎯 OBJETIVO', dist_name],
            ['📅 FECHA', target_date_str],
            ['📆 SEMANAS', f"{self.plan.get('total_weeks', 0)} semanas"],
        ]

        goal_table = Table(goal_data, colWidths=[150, 250])
        goal_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.Color(*self.COLORS['pastel_gray'])),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('TEXTCOLOR', (0, 0), (0, -1), dark_color),
            ('TEXTCOLOR', (1, 0), (1, -1), primary_color),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.Color(*self.COLORS['table_border'])),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.Color(*self.COLORS['table_border'])),
        ]))
        elements.append(goal_table)
        elements.append(Spacer(1, 0.6*cm))

        # ========== PERFIL DEL ATLETA ==========
        if self.profile:
            elements.append(Paragraph("👤 Perfil del Atleta", styles['SubHeading_ES']))
            elements.append(Spacer(1, 0.3*cm))

            profile_data = [
                ['Edad', f"{self.profile.get('age', '-')} años",
                 'Peso', f"{self.profile.get('weight', '-')} kg"],
                ['FC Máxima', f"{self.profile.get('max_hr', '-')} ppm",
                 'FC Reposo', f"{self.profile.get('resting_hr', '-')} ppm"],
                ['VO2 Máx', f"{self.profile.get('vo2_max', '-')} ml/kg/min", '', ''],
            ]

            profile_table = Table(profile_data, colWidths=[80, 100, 80, 100])
            profile_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.Color(*self.COLORS['pastel_blue'])),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.Color(*self.COLORS['text_light'])),
                ('TEXTCOLOR', (2, 0), (2, -1), colors.Color(*self.COLORS['text_light'])),
                ('TEXTCOLOR', (1, 0), (1, -1), dark_color),
                ('TEXTCOLOR', (3, 0), (3, -1), dark_color),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('ALIGN', (3, 0), (3, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOX', (0, 0), (-1, -1), 1, colors.Color(*self.COLORS['table_border'])),
                ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.Color(*self.COLORS['table_border'])),
            ]))
            elements.append(profile_table)
            elements.append(Spacer(1, 0.6*cm))

        # ========== FITNESS SCORE ACTUAL ==========
        fitness_score = self.profile.get('fitness_score', {})
        if fitness_score:
            elements.extend(self._create_fitness_score_section(
                fitness_score, styles, Paragraph, Spacer, Table, TableStyle, colors, HRFlowable
            ))

        elements.append(Spacer(1, 0.6*cm))

        # Fecha de generación
        elements.append(Paragraph(
            f"Generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}",
            styles['Small_ES']
        ))

        return elements

    def _create_fitness_score_section(self, fitness_score, styles, Paragraph, Spacer, Table, TableStyle, colors, HRFlowable):
        """Crea la sección de Fitness Score con métricas detalladas y gráfico de evolución."""
        from reportlab.lib.units import cm
        from reportlab.graphics.shapes import Drawing, Rect, String, Line, PolyLine
        from reportlab.graphics.charts.lineplots import LinePlot
        from reportlab.graphics import renderPDF
        from reportlab.lib.colors import Color
        from reportlab.platypus import KeepTogether
        elements = []

        primary_color = colors.Color(*self.COLORS['primary'])
        dark_color = colors.Color(*self.COLORS['dark'])
        secondary_color = colors.Color(*self.COLORS['secondary'])

        elements.append(Paragraph("🏆 Fitness Score - Estado de Forma", styles['SubHeading_ES']))
        elements.append(Spacer(1, 0.3*cm))

        score_value = fitness_score.get('fitness_score', 0)
        ctl_value = fitness_score.get('ctl', 0)
        atl_value = fitness_score.get('atl', 0)
        tsb_value = fitness_score.get('tsb', 0)
        form_status = fitness_score.get('form_status', '-')
        percentile = fitness_score.get('percentile', 0)
        percentile_label = fitness_score.get('percentile_label', '-')

        # Determinar color según el score
        if score_value >= 70:
            score_color = colors.Color(0.30, 0.69, 0.31)  # Verde
            score_bg = colors.Color(*self.COLORS['pastel_green'])
        elif score_value >= 50:
            score_color = colors.Color(*self.COLORS['primary'])  # Azul
            score_bg = colors.Color(*self.COLORS['pastel_blue'])
        elif score_value >= 30:
            score_color = colors.Color(1.0, 0.60, 0.0)  # Naranja
            score_bg = colors.Color(*self.COLORS['pastel_orange'])
        else:
            score_color = colors.Color(0.96, 0.26, 0.21)  # Rojo
            score_bg = colors.Color(1.0, 0.93, 0.93)

        # ========== TABLA UNIFICADA DE FITNESS SCORE ==========
        # Color TSB según valor
        if tsb_value > 10:
            tsb_hex = "#4CAF50"
        elif tsb_value > -10:
            tsb_hex = "#2196F3"
        elif tsb_value > -25:
            tsb_hex = "#FF9800"
        else:
            tsb_hex = "#F54336"

        # Ancho total fijo para todas las tablas: 480px
        total_width = 480

        # Tabla principal con Score y Percentil (4 columnas para alinear con métricas)
        main_data = [
            # Header row
            ['🎯 FITNESS SCORE', '', '📊 PERCENTIL', ''],
            # Values row
            [
                Paragraph(f"<font size='15' color='#{int(score_color.red*255):02x}{int(score_color.green*255):02x}{int(score_color.blue*255):02x}'><b>{score_value:.0f}</b></font><font size='14' color='#666666'>/100</font>", styles['Center_ES']),
                '',
                Paragraph(f"<font size='10' color='#2196F3'><b>{percentile}%</b></font><br/><font size='9' color='#666666'>{percentile_label}</font>", styles['Center_ES']),
                ''
            ],
        ]

        main_table = Table(main_data, colWidths=[total_width/2, 0, total_width/2, 0])
        main_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(*self.COLORS['pastel_gray'])),
            ('SPAN', (0, 0), (1, 0)),  # Merge first two columns for header
            ('SPAN', (2, 0), (3, 0)),  # Merge last two columns for header
            ('SPAN', (0, 1), (1, 1)),  # Merge first two columns for value
            ('SPAN', (2, 1), (3, 1)),  # Merge last two columns for value
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.Color(*self.COLORS['text_light'])),
            # Values
            ('BACKGROUND', (0, 1), (1, 1), score_bg),
            ('BACKGROUND', (2, 1), (3, 1), colors.Color(*self.COLORS['pastel_blue'])),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, 0), 8),
            ('PADDING', (0, 1), (-1, 1), 18),
            ('BOX', (0, 0), (-1, -1), 1, colors.Color(*self.COLORS['table_border'])),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.Color(*self.COLORS['table_border'])),
            ('LINEBEFORE', (2, 0), (2, -1), 0.5, colors.Color(*self.COLORS['table_border'])),
        ]))
        elements.append(main_table)
        elements.append(Spacer(1, 0.15*cm))

        # ========== MÉTRICAS CTL/ATL/TSB/ESTADO ==========
        # Anchos proporcionales: 100 + 100 + 80 + 200 = 480
        metrics_data = [
            ['💪 CTL (Forma)', '⚡ ATL (Fatiga)', '⚖️ TSB', '📈 Estado'],
            [
                Paragraph(f"<font size='14' color='#4CAF50'><b>{ctl_value:.1f}</b></font>", styles['Center_ES']),
                Paragraph(f"<font size='14' color='#FF5722'><b>{atl_value:.1f}</b></font>", styles['Center_ES']),
                Paragraph(f"<font size='14' color='{tsb_hex}'><b>{tsb_value:.1f}</b></font>", styles['Center_ES']),
                Paragraph(f"<font size='10' color='#2E3336'>{form_status}</font>", styles['Center_ES'])
            ]
        ]

        metrics_table = Table(metrics_data, colWidths=[100, 100, 80, 200])
        metrics_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(*self.COLORS['pastel_gray'])),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.Color(*self.COLORS['text_light'])),
            # Values row backgrounds
            ('BACKGROUND', (0, 1), (0, 1), colors.Color(*self.COLORS['pastel_green'])),
            ('BACKGROUND', (1, 1), (1, 1), colors.Color(*self.COLORS['pastel_orange'])),
            ('BACKGROUND', (2, 1), (2, 1), colors.Color(*self.COLORS['pastel_purple'])),
            ('BACKGROUND', (3, 1), (3, 1), colors.Color(*self.COLORS['pastel_blue'])),
            # Alignment and padding
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, 0), 8),
            ('PADDING', (0, 1), (-1, 1), 12),
            ('BOX', (0, 0), (-1, -1), 1, colors.Color(*self.COLORS['table_border'])),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.Color(*self.COLORS['table_border'])),
            ('LINEBEFORE', (1, 0), (-1, -1), 0.5, colors.Color(*self.COLORS['table_border'])),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3*cm))

        # ========== GRÁFICO DE EVOLUCIÓN (últimas 10 semanas) ==========
        evolution_data = fitness_score.get('evolution_data', [])
        if evolution_data and len(evolution_data) > 1:
            # Create elements for the evolution section that should stay together
            evolution_elements = []
            evolution_elements.append(Paragraph("📈 Evolución Últimas 10 Semanas", styles['SubHeading_ES']))
            evolution_elements.append(Spacer(1, 0.2*cm))

            # Tomar últimos 70 días (10 semanas) de datos
            recent_data = evolution_data[-70:] if len(evolution_data) > 70 else evolution_data

            # Crear gráfico con Drawing
            chart_width = 400
            chart_height = 120
            drawing = Drawing(chart_width, chart_height)

            # Preparar datos para el gráfico
            fs_scores = [d.get('fitness_score', 0) for d in recent_data]
            ctl_scores = [d.get('ctl', 0) for d in recent_data]
            atl_scores = [d.get('atl', 0) for d in recent_data]

            if fs_scores:
                max_val = max(max(fs_scores), max(ctl_scores), max(atl_scores), 1) * 1.1
                min_val = min(min(fs_scores), min(ctl_scores), min(atl_scores), 0) * 0.9

                # Área del gráfico
                margin_left = 35
                margin_bottom = 20
                margin_top = 10
                margin_right = 10
                plot_width = chart_width - margin_left - margin_right
                plot_height = chart_height - margin_bottom - margin_top

                # Fondo del gráfico
                drawing.add(Rect(margin_left, margin_bottom, plot_width, plot_height,
                               fillColor=Color(0.98, 0.98, 0.99), strokeColor=Color(0.9, 0.9, 0.9)))

                # Líneas de grid horizontales
                for i in range(5):
                    y = margin_bottom + (i * plot_height / 4)
                    drawing.add(Line(margin_left, y, margin_left + plot_width, y,
                                   strokeColor=Color(0.92, 0.92, 0.92), strokeWidth=0.5))
                    val = min_val + (i * (max_val - min_val) / 4)
                    drawing.add(String(margin_left - 5, y - 3, f"{val:.0f}",
                                      fontSize=6, fillColor=Color(0.5, 0.5, 0.5), textAnchor='end'))

                # Función para convertir valor a coordenada Y
                def val_to_y(val):
                    if max_val == min_val:
                        return margin_bottom + plot_height / 2
                    return margin_bottom + ((val - min_val) / (max_val - min_val)) * plot_height

                # Dibujar líneas de datos
                n_points = len(fs_scores)
                if n_points > 1:
                    x_step = plot_width / (n_points - 1)

                    # Fitness Score (azul)
                    fs_points = []
                    for i, score in enumerate(fs_scores):
                        x = margin_left + i * x_step
                        y = val_to_y(score)
                        fs_points.extend([x, y])
                    drawing.add(PolyLine(fs_points, strokeColor=Color(0.13, 0.59, 0.95), strokeWidth=2))

                    # CTL (verde)
                    ctl_points = []
                    for i, score in enumerate(ctl_scores):
                        x = margin_left + i * x_step
                        y = val_to_y(score)
                        ctl_points.extend([x, y])
                    drawing.add(PolyLine(ctl_points, strokeColor=Color(0.30, 0.69, 0.31), strokeWidth=1.5))

                    # ATL (naranja)
                    atl_points = []
                    for i, score in enumerate(atl_scores):
                        x = margin_left + i * x_step
                        y = val_to_y(score)
                        atl_points.extend([x, y])
                    drawing.add(PolyLine(atl_points, strokeColor=Color(1.0, 0.34, 0.13), strokeWidth=1.5))

                # Leyenda
                legend_y = chart_height - 8
                drawing.add(Line(chart_width - 120, legend_y, chart_width - 105, legend_y,
                               strokeColor=Color(0.13, 0.59, 0.95), strokeWidth=2))
                drawing.add(String(chart_width - 100, legend_y - 3, "Score", fontSize=7, fillColor=Color(0.3, 0.3, 0.3)))

                drawing.add(Line(chart_width - 70, legend_y, chart_width - 55, legend_y,
                               strokeColor=Color(0.30, 0.69, 0.31), strokeWidth=1.5))
                drawing.add(String(chart_width - 50, legend_y - 3, "CTL", fontSize=7, fillColor=Color(0.3, 0.3, 0.3)))

                drawing.add(Line(chart_width - 30, legend_y, chart_width - 15, legend_y,
                               strokeColor=Color(1.0, 0.34, 0.13), strokeWidth=1.5))
                drawing.add(String(chart_width - 10, legend_y - 3, "ATL", fontSize=7, fillColor=Color(0.3, 0.3, 0.3)))

                evolution_elements.append(drawing)

            # Use KeepTogether to ensure title and graph stay on same page
            elements.append(KeepTogether(evolution_elements))

        return elements

    def _create_plan_summary(self, styles, Paragraph, Spacer, Table, TableStyle, colors, HRFlowable):
        """Crea la página de resumen del plan con diseño mejorado."""
        from reportlab.lib.units import cm
        elements = []

        primary_color = colors.Color(*self.COLORS['primary'])
        dark_color = colors.Color(*self.COLORS['dark'])
        secondary_color = colors.Color(*self.COLORS['secondary'])

        elements.append(Paragraph("📊 Resumen del Plan de Entrenamiento", styles['Heading_ES']))
        elements.append(HRFlowable(width="100%", thickness=1, color=primary_color,
                                   spaceBefore=5, spaceAfter=15, hAlign='LEFT'))

        # Estadísticas generales en formato tarjeta
        total_distance = sum(w.get('total_distance', 0) for w in self.plan.get('weeks', []))
        total_weeks = len(self.plan.get('weeks', []))
        peak_distance = self.plan.get('peak_week_distance', 0)
        avg_weekly = total_distance / total_weeks if total_weeks > 0 else 0

        stats_data = [
            ['📏 Distancia Total', '📆 Semanas', '⬆️ Pico Semanal', '📈 Media Semanal'],
            [f"{total_distance:.0f} km", str(total_weeks), f"{peak_distance:.0f} km", f"{avg_weekly:.1f} km"],
        ]

        stats_table = Table(stats_data, colWidths=[120, 100, 120, 120])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(*self.COLORS['pastel_gray'])),
            ('BACKGROUND', (0, 1), (-1, 1), colors.Color(*self.COLORS['pastel_blue'])),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.Color(*self.COLORS['text_light'])),
            ('TEXTCOLOR', (0, 1), (-1, 1), primary_color),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.Color(*self.COLORS['table_border'])),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.Color(*self.COLORS['table_border'])),
            ('LINEBEFORE', (1, 0), (-1, -1), 0.5, colors.Color(*self.COLORS['table_border'])),
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 0.8*cm))

        # Resumen por semana
        elements.append(Paragraph("📅 Progresión Semanal", styles['SubHeading_ES']))
        elements.append(Spacer(1, 0.3*cm))

        week_headers = ['Sem', 'Fase', 'Distancia', 'Sesiones', 'Tipo']
        week_data = [week_headers]

        phase_names = {'base': 'Base', 'build': 'Construcción', 'peak': 'Pico', 'taper': 'Tapering'}
        phase_colors = {
            'base': colors.Color(0.55, 0.76, 0.29),
            'build': colors.Color(1.0, 0.6, 0.0),
            'peak': colors.Color(0.96, 0.26, 0.21),
            'taper': colors.Color(0.13, 0.59, 0.95)
        }

        for week in self.plan.get('weeks', []):
            phase = phase_names.get(week.get('phase', 'build'), week.get('phase', ''))
            workouts_count = sum(1 for w in week.get('workouts', {}).values() if w.get('type') != 'Rest')

            week_data.append([
                str(week['week_number']),
                phase,
                f"{week.get('total_distance', 0):.1f} km",
                str(workouts_count),
                '🔄 Rec' if week.get('is_recovery') else '💪 Normal'
            ])

        week_table = Table(week_data, colWidths=[45, 90, 80, 60, 85])

        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(*self.COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 1, colors.Color(*self.COLORS['table_border'])),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.Color(*self.COLORS['table_border'])),
        ]

        # Colorear filas alternadas y por fase con colores pastel
        for i, week in enumerate(self.plan.get('weeks', []), 1):
            if week.get('is_recovery'):
                style_commands.append(('BACKGROUND', (0, i), (-1, i), colors.Color(*self.COLORS['pastel_blue'])))
            elif i % 2 == 0:
                style_commands.append(('BACKGROUND', (0, i), (-1, i), colors.Color(*self.COLORS['table_alt_row'])))
            else:
                style_commands.append(('BACKGROUND', (0, i), (-1, i), colors.white))

        week_table.setStyle(TableStyle(style_commands))
        elements.append(week_table)

        return elements


    def _create_zones_page(self, styles, Paragraph, Spacer, Table, TableStyle, colors, HRFlowable):
        """Crea la página de zonas de entrenamiento (FC + Ritmo)."""
        from reportlab.lib.units import cm
        elements = []

        primary_color = colors.Color(*self.COLORS['primary'])
        dark_color = colors.Color(*self.COLORS['dark'])

        elements.append(Paragraph("🎯 Zonas de Entrenamiento", styles['Heading_ES']))
        elements.append(HRFlowable(width="100%", thickness=1, color=primary_color,
                                   spaceBefore=5, spaceAfter=15, hAlign='LEFT'))

        # ========== ZONAS DE FRECUENCIA CARDÍACA ==========
        elements.append(Paragraph("❤️ Zonas de Frecuencia Cardíaca", styles['SubHeading_ES']))
        elements.append(Spacer(1, 0.3*cm))

        zone_headers = ['Zona', 'Nombre', 'FC Mín', 'FC Máx', '% FC Máx', 'Descripción']
        zone_data = [zone_headers]

        zone_names = {
            1: ('Recuperación', 'Muy fácil, conversación fluida'),
            2: ('Aeróbico', 'Cómodo, puedes hablar'),
            3: ('Tempo', 'Moderado, frases cortas'),
            4: ('Umbral', 'Duro, palabras sueltas'),
            5: ('VO2 Máx', 'Muy duro, sin hablar')
        }

        zone_colors_list = [
            colors.Color(0.30, 0.69, 0.31),  # Verde - Z1
            colors.Color(0.55, 0.76, 0.29),  # Verde claro - Z2
            colors.Color(1.00, 0.76, 0.03),  # Amarillo - Z3
            colors.Color(1.00, 0.34, 0.13),  # Naranja - Z4
            colors.Color(0.96, 0.26, 0.21),  # Rojo - Z5
        ]

        max_hr = self.profile.get('max_hr', 190)

        # Porcentajes de FC para cada zona
        zone_percentages = {
            1: (50, 60),
            2: (60, 70),
            3: (70, 80),
            4: (80, 90),
            5: (90, 100)
        }

        for zone_num in range(1, 6):
            zone_info = self.zones.get(f'zone_{zone_num}', {})
            name, desc = zone_names.get(zone_num, ('', ''))
            pct_min, pct_max = zone_percentages[zone_num]

            zone_data.append([
                f"Z{zone_num}",
                name,
                f"{zone_info.get('hr_min', int(max_hr * pct_min / 100))}",
                f"{zone_info.get('hr_max', int(max_hr * pct_max / 100))}",
                f"{pct_min}-{pct_max}%",
                desc
            ])

        hr_table = Table(zone_data, colWidths=[40, 85, 55, 55, 65, 180])

        # Pastel backgrounds for each zone row
        zone_pastel_colors = [
            colors.Color(0.91, 0.96, 0.91),  # Pastel green Z1
            colors.Color(0.93, 0.97, 0.91),  # Pastel light green Z2
            colors.Color(1.00, 0.97, 0.88),  # Pastel yellow Z3
            colors.Color(1.00, 0.93, 0.88),  # Pastel orange Z4
            colors.Color(1.00, 0.91, 0.91),  # Pastel red Z5
        ]

        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(*self.COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('ALIGN', (-1, 1), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 1, colors.Color(*self.COLORS['table_border'])),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.Color(*self.COLORS['table_border'])),
        ]

        # Colorear indicadores de zona y fondo pastel de filas
        for i, (zone_color, pastel_color) in enumerate(zip(zone_colors_list, zone_pastel_colors)):
            style_commands.append(('BACKGROUND', (0, i+1), (0, i+1), zone_color))
            style_commands.append(('TEXTCOLOR', (0, i+1), (0, i+1), colors.white))
            style_commands.append(('FONTNAME', (0, i+1), (0, i+1), 'Helvetica-Bold'))
            style_commands.append(('BACKGROUND', (1, i+1), (-1, i+1), pastel_color))

        hr_table.setStyle(TableStyle(style_commands))
        elements.append(hr_table)
        elements.append(Spacer(1, 0.8*cm))

        # ========== ZONAS DE RITMO ==========
        elements.append(Paragraph("⏱️ Zonas de Ritmo (Pace)", styles['SubHeading_ES']))
        elements.append(Spacer(1, 0.3*cm))

        pace_headers = ['Zona', 'Nombre', 'Ritmo Mín', 'Ritmo Máx', 'Descripción']
        pace_data = [pace_headers]

        for zone_num in range(1, 6):
            pace_info = self.pace_zones.get(f'zone_{zone_num}', {})

            pace_data.append([
                f"Z{zone_num}",
                pace_info.get('name', '-'),
                f"{pace_info.get('pace_min', '-')} /km",
                f"{pace_info.get('pace_max', '-')} /km",
                pace_info.get('description', '-')
            ])

        pace_table = Table(pace_data, colWidths=[40, 120, 80, 80, 180])

        pace_style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(*self.COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('ALIGN', (-1, 1), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 1, colors.Color(*self.COLORS['table_border'])),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.Color(*self.COLORS['table_border'])),
        ]

        # Colorear indicadores de zona de ritmo con fondo pastel
        for i, (zone_color, pastel_color) in enumerate(zip(zone_colors_list, zone_pastel_colors)):
            pace_style_commands.append(('BACKGROUND', (0, i+1), (0, i+1), zone_color))
            pace_style_commands.append(('TEXTCOLOR', (0, i+1), (0, i+1), colors.white))
            pace_style_commands.append(('FONTNAME', (0, i+1), (0, i+1), 'Helvetica-Bold'))
            pace_style_commands.append(('BACKGROUND', (1, i+1), (-1, i+1), pastel_color))

        pace_table.setStyle(TableStyle(pace_style_commands))
        elements.append(pace_table)

        # Nota informativa
        elements.append(Spacer(1, 0.8*cm))
        elements.append(Paragraph(
            "💡 <i>Las zonas de ritmo están calculadas según la fórmula de Jack Daniels (VDOT) "
            "basándose en tu VO2max y/o mejores marcas personales.</i>",
            styles['Small_ES']
        ))

        return elements

    def _create_week_page(self, week, styles, Paragraph, Spacer, Table, TableStyle, colors, include_details, HRFlowable):
        """Crea la página de una semana de entrenamiento con diseño mejorado."""
        from reportlab.lib.units import cm
        elements = []

        primary_color = colors.Color(*self.COLORS['primary'])
        dark_color = colors.Color(*self.COLORS['dark'])

        week_num = week['week_number']
        phase_names = {'base': 'Base', 'build': 'Construcción', 'peak': 'Pico', 'taper': 'Tapering'}
        phase_icons = {'base': '🏗️', 'build': '📈', 'peak': '🏔️', 'taper': '🎯'}
        phase = phase_names.get(week.get('phase', 'build'), week.get('phase', ''))
        phase_icon = phase_icons.get(week.get('phase', 'build'), '📅')
        recovery_badge = ' 🔄 Recuperación' if week.get('is_recovery') else ''

        # Título de la semana con estilo
        elements.append(Paragraph(
            f"{phase_icon} Semana {week_num} - Fase {phase}{recovery_badge}",
            styles['Heading_ES']
        ))
        elements.append(HRFlowable(width="100%", thickness=1, color=primary_color,
                                   spaceBefore=5, spaceAfter=10, hAlign='LEFT'))

        start_date = week.get('start_date', datetime.now().date())
        if isinstance(start_date, datetime):
            start_date = start_date.date()

        end_date = start_date + timedelta(days=6)

        # Info de la semana en tabla compacta
        info_data = [[
            f"📅 {start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m/%Y')}",
            f"📏 {week.get('total_distance', 0):.1f} km total",
            f"💪 {sum(1 for w in week.get('workouts', {}).values() if w.get('type') != 'Rest')} sesiones"
        ]]

        info_table = Table(info_data, colWidths=[160, 140, 100])
        info_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.Color(*self.COLORS['text_light'])),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'CENTER'),
            ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 0.4*cm))

        # Tabla de entrenamientos de la semana
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        days_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        days_short = ['LUN', 'MAR', 'MIÉ', 'JUE', 'VIE', 'SÁB', 'DOM']

        workout_headers = ['Día', 'Entrenamiento', 'Dist.', 'Zona', 'Descripción']
        workout_data = [workout_headers]

        for day_en, day_es, day_short in zip(days_order, days_es, days_short):
            workout = week.get('workouts', {}).get(day_en, {'type': 'Rest'})
            workout_type = workout.get('type', 'Rest')
            type_es = self.WORKOUT_TRANSLATIONS.get(workout_type, workout_type)

            # Iconos para tipos de entrenamiento
            type_icons = {
                'Rest': '😴', 'Easy Run': '🏃', 'Tempo Run': '⚡',
                'Intervals': '🔥', 'Long Run': '🛤️', 'Recovery Run': '🚶',
                'Hill Repeats': '⛰️', 'Fartlek': '🎲'
            }
            icon = type_icons.get(workout_type, '🏃')

            workout_data.append([
                day_short,
                f"{icon} {type_es}",
                f"{workout.get('distance', 0):.1f}" if workout_type != 'Rest' else '-',
                f"Z{workout.get('zone', '-')}" if workout_type != 'Rest' else '-',
                (workout.get('description', 'Descanso')[:45] + '...') if len(workout.get('description', '')) > 45 else workout.get('description', 'Descanso')
            ])

        workout_table = Table(workout_data, colWidths=[40, 95, 40, 40, 220])

        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(*self.COLORS['table_header'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (3, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('ALIGN', (4, 1), (4, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 1, colors.Color(*self.COLORS['table_border'])),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.Color(*self.COLORS['table_border'])),
        ]

        # Colorear filas según tipo de entrenamiento con colores pastel suaves
        for i, (day_en, _) in enumerate(zip(days_order, days_es)):
            workout = week.get('workouts', {}).get(day_en, {'type': 'Rest'})
            workout_type = workout.get('type', 'Rest')
            rgb = self.WORKOUT_COLORS.get(workout_type, (200, 200, 200))
            # Color pastel muy suave para el fondo (más claro y menos saturado)
            pastel_r = 0.9 + (rgb[0]/255) * 0.1
            pastel_g = 0.9 + (rgb[1]/255) * 0.1
            pastel_b = 0.9 + (rgb[2]/255) * 0.1
            row_color = colors.Color(min(1.0, pastel_r), min(1.0, pastel_g), min(1.0, pastel_b))
            style_commands.append(('BACKGROUND', (0, i+1), (-1, i+1), row_color))

        workout_table.setStyle(TableStyle(style_commands))
        elements.append(workout_table)

        # Detalles de entrenamientos con segmentos
        if include_details:
            elements.append(Spacer(1, 0.5*cm))
            elements.extend(self._create_workout_details(week, styles, Paragraph, Spacer, Table, TableStyle, colors))

        return elements

    def _create_workout_details(self, week, styles, Paragraph, Spacer, Table, TableStyle, colors):
        """Crea los detalles de los entrenamientos con segmentos y recuperaciones."""
        from reportlab.lib.units import cm
        elements = []

        primary_color = colors.Color(*self.COLORS['primary'])

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        days_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

        type_icons = {
            'Easy Run': '🏃', 'Tempo Run': '⚡', 'Intervals': '🔥',
            'Long Run': '🛤️', 'Recovery Run': '🚶', 'Hill Repeats': '⛰️', 'Fartlek': '🎲'
        }

        for day_en, day_es in zip(days_order, days_es):
            workout = week.get('workouts', {}).get(day_en, {'type': 'Rest'})
            workout_type = workout.get('type', 'Rest')

            if workout_type == 'Rest':
                continue

            segments = workout.get('segments', [])
            if not segments:
                continue

            type_es = self.WORKOUT_TRANSLATIONS.get(workout_type, workout_type)
            icon = type_icons.get(workout_type, '🏃')

            rgb = self.WORKOUT_COLORS.get(workout_type, (100, 100, 100))
            workout_color = colors.Color(rgb[0]/255, rgb[1]/255, rgb[2]/255)

            # Título con descripción completa
            description = workout.get('description', '')
            elements.append(Paragraph(
                f"{icon} <b>{day_es}: {type_es}</b> — {workout.get('distance', 0):.1f} km",
                styles['SubHeading_ES']
            ))

            # Mostrar descripción del entrenamiento
            if description:
                elements.append(Paragraph(
                    f"<b>🎯 Objetivo:</b> {description}",
                    styles['Normal_ES']
                ))

            # Mostrar estructura si existe
            structure = workout.get('structure', '')
            if structure:
                elements.append(Paragraph(
                    f"<b>📋 Estructura:</b> {structure}",
                    styles['Normal_ES']
                ))

            # Información adicional para entrenamientos específicos
            extra_info = []

            # Para Series/Intervalos
            if workout_type == 'Intervals':
                if workout.get('reps'):
                    extra_info.append(f"<b>Repeticiones:</b> {workout['reps']}x {workout.get('rep_distance', '')}m")
                if workout.get('recovery_time'):
                    rec_time = workout['recovery_time']
                    time_str = f"{rec_time // 60}:{rec_time % 60:02d}" if rec_time >= 60 and rec_time % 60 else (f"{rec_time // 60}min" if rec_time >= 60 else f"{rec_time}s")
                    extra_info.append(f"<b>Recuperación:</b> {time_str}")
                    if workout.get('recovery_distance'):
                        extra_info[-1] += f" ({workout['recovery_distance']}m {workout.get('recovery_type', 'trote')})"
                if workout.get('pace_type'):
                    extra_info.append(f"<b>Ritmo objetivo:</b> {workout['pace_type'].upper()}")

            # Para Fartlek
            elif workout_type == 'Fartlek':
                if workout.get('num_cambios'):
                    extra_info.append(f"<b>Cambios de ritmo:</b> {workout['num_cambios']}")
                if workout.get('duracion_rapido'):
                    dur = workout['duracion_rapido']
                    dur_str = f"{dur // 60}:{dur % 60:02d}" if dur >= 60 and dur % 60 else (f"{dur // 60}min" if dur >= 60 else f"{dur}s")
                    extra_info.append(f"<b>Duración rápido:</b> {dur_str}")
                if workout.get('duracion_suave'):
                    dur = workout['duracion_suave']
                    dur_str = f"{dur // 60}:{dur % 60:02d}" if dur >= 60 and dur % 60 else (f"{dur // 60}min" if dur >= 60 else f"{dur}s")
                    extra_info.append(f"<b>Duración suave:</b> {dur_str}")

            # Para Cuestas
            elif workout_type == 'Hill Repeats':
                if workout.get('reps'):
                    extra_info.append(f"<b>Repeticiones:</b> {workout['reps']}x {workout.get('rep_duration', '')}s")
                if workout.get('incline'):
                    extra_info.append(f"<b>Pendiente:</b> {workout['incline']}")
                if workout.get('recovery_duration'):
                    rec_time = workout['recovery_duration']
                    time_str = f"{rec_time // 60}:{rec_time % 60:02d}" if rec_time >= 60 and rec_time % 60 else (f"{rec_time // 60}min" if rec_time >= 60 else f"{rec_time}s")
                    extra_info.append(f"<b>Recuperación:</b> {time_str} bajando")

            if extra_info:
                elements.append(Paragraph(
                    " | ".join(extra_info),
                    styles['Normal_ES']
                ))

            elements.append(Spacer(1, 0.2*cm))

            seg_headers = ['#', 'Segmento', 'Volumen', 'Ritmo', 'Recuperación', 'Zona']
            seg_data = [seg_headers]

            for idx, segment in enumerate(segments, 1):
                # Formatear volumen
                dist_dur = ''
                if segment.get('reps'):
                    rep_val = segment.get('rep_distance', segment.get('rep_duration', ''))
                    unit = 'm' if segment.get('rep_distance') else 's'
                    dist_dur = f"{segment['reps']}x{rep_val}{unit}"
                elif segment.get('distance_km'):
                    dist_dur = f"{segment['distance_km']:.1f} km"
                elif segment.get('duration_min'):
                    dist_dur = f"{segment['duration_min']:.0f} min"

                # Formatear recuperación detallada
                recovery = '-'
                if segment.get('recovery_duration'):
                    rec_time = segment['recovery_duration']
                    if rec_time >= 60:
                        recovery = f"{rec_time // 60}:{rec_time % 60:02d}" if rec_time % 60 else f"{rec_time // 60}min"
                    else:
                        recovery = f"{rec_time}s"
                    if segment.get('recovery_distance'):
                        recovery += f" {segment['recovery_distance']}m"
                elif segment.get('rest_after'):
                    recovery = segment['rest_after'][:18]

                # Formatear ritmo
                pace = segment.get('pace', '-')
                if segment.get('fast_pace') and segment.get('slow_pace'):
                    pace = f"{segment['fast_pace']}-{segment['slow_pace']}"
                elif segment.get('pace_range') and segment.get('pace_range') != pace:
                    pace = segment.get('pace_range', pace)

                seg_data.append([
                    str(idx),
                    segment.get('name', 'Segmento')[:22],
                    dist_dur,
                    pace[:14] if len(str(pace)) > 14 else pace,
                    recovery[:18] if len(str(recovery)) > 18 else recovery,
                    f"Z{segment.get('zone', '-')}"
                ])

            seg_table = Table(seg_data, colWidths=[20, 95, 60, 80, 95, 32])

            # Pastel zone colors for segment rows
            zone_colors = {
                '1': colors.Color(0.91, 0.96, 0.91),  # Pastel green
                '2': colors.Color(0.93, 0.97, 0.91),  # Pastel light green
                '3': colors.Color(1.00, 0.97, 0.90),  # Pastel yellow
                '4': colors.Color(1.00, 0.93, 0.90),  # Pastel orange
                '5': colors.Color(1.00, 0.92, 0.92),  # Pastel red
                '4-5': colors.Color(1.00, 0.92, 0.92),
                '2-4': colors.Color(1.00, 0.95, 0.88),
            }

            style_commands = [
                ('BACKGROUND', (0, 0), (-1, 0), workout_color),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 7),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (1, 1), (1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.Color(*self.COLORS['table_border'])),
                ('LINEBELOW', (0, 0), (-1, -2), 0.3, colors.Color(*self.COLORS['table_border'])),
            ]

            for idx, segment in enumerate(segments, 1):
                zone = str(segment.get('zone', '2'))
                if zone in zone_colors:
                    style_commands.append(('BACKGROUND', (0, idx), (-1, idx), zone_colors[zone]))
                else:
                    # Default pastel for unlisted zones
                    style_commands.append(('BACKGROUND', (0, idx), (-1, idx), colors.Color(*self.COLORS['table_alt_row'])))

            seg_table.setStyle(TableStyle(style_commands))
            elements.append(seg_table)

            # Añadir notas de cada segmento si existen
            notes_text = []
            for idx, segment in enumerate(segments, 1):
                if segment.get('notes'):
                    notes_text.append(f"• Seg {idx}: {segment['notes']}")

            if notes_text:
                elements.append(Spacer(1, 0.1*cm))
                elements.append(Paragraph(
                    "<font size=7>" + "<br/>".join(notes_text[:3]) + "</font>",
                    styles['Normal_ES']
                ))

            elements.append(Spacer(1, 0.4*cm))

        return elements


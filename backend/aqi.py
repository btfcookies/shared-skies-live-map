"""PM2.5 color scale and EPA AQI conversion."""

import math

import numpy as np


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color. Uses round-half-up (int(c+0.5)) — matching
    JS Math.round for non-negative values — so the backend's per-tract `color`
    is BYTE-IDENTICAL to the frontend legend's pm25Color (avoids Python's
    banker's rounding diverging by 1 LSB at exact .5 midpoints)."""
    return '#{:02x}{:02x}{:02x}'.format(
        max(0, min(255, int(rgb[0] + 0.5))),
        max(0, min(255, int(rgb[1] + 0.5))),
        max(0, min(255, int(rgb[2] + 0.5))),
    )


def interpolate_color(color1, color2, factor):
    """Interpolate between two colors based on factor (0-1)."""
    factor = max(0, min(1, factor))
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)

    rgb = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * factor for i in range(3))
    return rgb_to_hex(rgb)


def pm25_color_gradient(pm25: float) -> str:
    """Get gradient color based on PM2.5 value.

    Bands: Good 0-9 (green), Moderate 9-13 (yellow), Elevated 13-17 (orange),
    High 17+ (red→dark red). Each band darkens toward its upper boundary, and
    the High band keeps darkening with concentration so wildfire-smoke days read
    dramatically dark. 9 = U.S. EPA annual NAAQS (2024).
    """
    if pm25 < 0:
        pm25 = 0

    # Green: 0.0-9.0  (within the EPA annual standard)
    if pm25 <= 9.0:
        factor = pm25 / 9.0
        return interpolate_color("#90EE90", "#00b894", factor)

    # Yellow: 9.0-13.0
    elif pm25 <= 13.0:
        factor = (pm25 - 9.0) / 4.0
        return interpolate_color("#FFFF99", "#FFD700", factor)

    # Orange: 13.0-17.0
    elif pm25 <= 17.0:
        factor = (pm25 - 13.0) / 4.0
        return interpolate_color("#FFB347", "#E8590C", factor)

    # Red → dark red: 17.0+  (darkens with concentration; saturates ~55, the EPA
    # 24-hr Unhealthy threshold, so smoke/dust days read dramatically dark).
    else:
        factor = min(1.0, (pm25 - 17.0) / 38.0)
        return interpolate_color("#FF6B6B", "#800000", factor)


# U.S. EPA PM2.5 AQI breakpoints (May 2024 revision). Used to show an
# AQI-equivalent next to our µg/m³ so users can compare directly with apps that
# display AQI (PurpleAir map, AirNow). (C_lo, C_hi, AQI_lo, AQI_hi)
_EPA_AQI_BREAKPOINTS = [
    (0.0,   9.0,   0,   50),
    (9.1,   35.4,  51,  100),
    (35.5,  55.4,  101, 150),
    (55.5,  125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]


def pm25_to_epa_aqi(pm25: float) -> int:
    """Convert PM2.5 (µg/m³) to the U.S. EPA AQI (2024 breakpoints)."""
    c = max(0.0, math.floor(float(pm25) * 10) / 10)  # EPA truncates to 0.1
    for c_lo, c_hi, a_lo, a_hi in _EPA_AQI_BREAKPOINTS:
        if c <= c_hi:
            c_lo_eff = c_lo if c >= c_lo else 0.0
            return int(round((a_hi - a_lo) / (c_hi - c_lo_eff) * (c - c_lo_eff) + a_lo))
    return 500


def pm25_info(pm25: float) -> dict:
    """Get category info. Bands: Good 0-9 / Moderate 9-13 / Elevated 13-17 /
    High 17+. 9 µg/m³ = U.S. EPA annual NAAQS; 15 µg/m³ = WHO 24-hr guideline."""
    color = pm25_color_gradient(pm25)

    if pm25 <= 9.0:
        return {
            "category": "Good",
            "color": color,
            "aqi_range": "0–9",
            "health_msg": "Air quality is good — within the U.S. EPA annual PM2.5 standard (9 µg/m³).",
        }
    elif pm25 <= 13.0:
        return {
            "category": "Moderate",
            "color": color,
            "aqi_range": "9–13",
            "health_msg": "Moderate — above the EPA annual standard. Unusually sensitive people may want to limit prolonged outdoor exertion.",
        }
    elif pm25 <= 17.0:
        return {
            "category": "Elevated",
            "color": color,
            "aqi_range": "13–17",
            "health_msg": "Elevated — above the WHO 24-hour guideline (15 µg/m³). Sensitive groups should limit prolonged outdoor activity.",
        }
    else:
        return {
            "category": "High",
            "color": color,
            "aqi_range": "17+",
            "health_msg": "⚠️ High — everyone may begin to feel effects; sensitive groups are at greater risk. Often driven by wildfire smoke or dust.",
        }


def safe_float(val) -> float:
    """Convert to float, handle NaN and None."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if np.isnan(f) else round(f, 1)
    except (TypeError, ValueError):
        return 0.0

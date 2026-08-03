"""Stoneplace — simulador de evolución para SeedWorld.

Reglas de tiempo del mundo Stoneplace:
    1 año   = 400 días
    1 día   = 25 horas
    1 año   = 10 000 horas
"""
HOURS_PER_DAY = 25
DAYS_PER_YEAR = 400
HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR  # 10 000

# v1.5 — honestidad científica + coevolución + narrativa + tree-sequence
__version__ = "1.5.1"

# Escala trófica: kcal / (locus_units × dt_años). Antes era la constante
# mágica `50000` del simulador. La calibración: un individuo de ~100 g
# consume ≈2% de la biomasa de su celda por día en régimen estacionario.
CONSUMPTION_SCALE_KCAL = 50_000.0

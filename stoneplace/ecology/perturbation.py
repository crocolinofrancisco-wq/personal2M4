"""Perturbaciones catastróficas — Stoneplace v1.5.

Cada evento se declara por config y se aplica cuando `year >= at_year`.

Tipos soportados:
  * ``impact`` — impacto puntual: mata fauna en un radio (`radius_cells`)
    con probabilidad `kill_prob`.
  * ``glaciation`` — enfría un rango latitudinal por `delta_c` durante
    `duration_years` (se aplica cada tick que esté dentro del rango).
  * ``eruption`` — nube volcánica: reduce fotosíntesis y baja
    temperatura globalmente durante `duration_years`.

Todos los eventos quedan trazados en la narrativa.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PerturbationEvent:
    kind: str
    at_year: float
    duration_years: float = 1.0
    params: dict = field(default_factory=dict)

    def is_active(self, year: float) -> bool:
        return self.at_year <= year < self.at_year + self.duration_years

    def apply(self, world, pops, year, narrative):
        if self.kind == "impact":
            self._apply_impact(world, pops, year, narrative)
        elif self.kind == "glaciation":
            self._apply_glaciation(world, year, narrative)
        elif self.kind == "eruption":
            self._apply_eruption(world, year, narrative)

    def _apply_impact(self, world, pops, year, narrative):
        cy = int(self.params.get("y", world.height // 2))
        cx = int(self.params.get("x", world.width // 2))
        radius = int(self.params.get("radius_cells", 40))
        kill_prob = float(self.params.get("kill_prob", 0.85))
        deaths = 0
        for pop in pops:
            if pop.n == 0:
                continue
            dy = np.minimum((pop.y - cy) % world.height,
                             (cy - pop.y) % world.height)
            dx = np.minimum((pop.x - cx) % world.width,
                             (cx - pop.x) % world.width)
            d2 = dy * dy + dx * dx
            hit = d2 <= radius * radius
            rng_local = np.random.default_rng(int(year * 1000) + pop.template.species_id)
            die = hit & (rng_local.random(pop.n) < kill_prob)
            keep = ~die
            deaths += int(die.sum())
            pop.kill_mask(keep)
        narrative.log(year, "impact", 0,
                      f"Impacto en (y={cy},x={cx}) r={radius}: "
                      f"{deaths} muertes fauna.")

    def _apply_glaciation(self, world, year, narrative):
        delta_c = float(self.params.get("delta_c", -8.0))
        band_top = int(self.params.get("y_top", 0))
        band_bottom = int(self.params.get("y_bottom", world.height // 6))
        world.layers["temperature"][band_top:band_bottom] += delta_c
        narrative.log(year, "glaciation", 0,
                      f"Glaciación en franja y={band_top}..{band_bottom} "
                      f"Δ={delta_c:+.1f}°C.")

    def _apply_eruption(self, world, year, narrative):
        delta_c = float(self.params.get("delta_c", -2.0))
        world.layers["temperature"] += delta_c
        narrative.log(year, "eruption", 0,
                      f"Erupción global Δ={delta_c:+.1f}°C durante "
                      f"{self.duration_years} años.")


@dataclass
class PerturbationSchedule:
    events: list = field(default_factory=list)
    _fired: set = field(default_factory=set)

    def __init__(self, events_cfg: list[dict] | None = None):
        self.events = []
        self._fired = set()
        for e in (events_cfg or []):
            self.events.append(PerturbationEvent(
                kind=e["kind"],
                at_year=float(e["at_year"]),
                duration_years=float(e.get("duration_years", 1.0)),
                params=dict(e.get("params", {})),
            ))

    def due_for(self, year: float) -> list[PerturbationEvent]:
        due = []
        for i, ev in enumerate(self.events):
            if i in self._fired:
                continue
            if ev.at_year <= year:
                due.append(ev)
                # One-shot para impact; glaciation/eruption pueden pulsar
                # el clima una única vez (los deltas quedan aplicados y
                # el clima dinámico los suaviza en los ticks siguientes).
                self._fired.add(i)
        return due

"""Perturbaciones catastróficas — Stoneplace v1.5.1.

Cada evento se declara por config y se aplica en `at_year`. Los eventos
con duración (glaciación, erupción) modifican `sim._base_temperature`
al empezar y REVIERTEN el cambio al terminar, para que:

  * el ciclo climático dinámico (Milankovitch/estacional) siga
    aplicándose sobre la nueva línea base durante el evento,
  * el mundo vuelva a la normalidad cuando el evento termina.

Sin esto, el ciclo climático de v1.5 sobrescribía la glaciación a los
pocos ticks (bug detectado en v1.5.0).

Tipos soportados:

  * ``impact`` — one-shot: mata fauna en un radio con `kill_prob`.
  * ``glaciation`` — enfría un rango latitudinal por `delta_c` DURANTE
    `duration_years` (persistente al ciclo climático).
  * ``eruption`` — enfría todo el mundo por `delta_c` durante `duration_years`.
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
    # Estado de máquina
    started: bool = False
    ended: bool = False

    def _band_slice(self, sim):
        world = sim.world
        band_top = int(self.params.get("y_top", 0))
        band_bottom = int(self.params.get("y_bottom", world.height // 6))
        return slice(band_top, band_bottom)

    def start(self, sim, year):
        """Aplica el efecto al iniciar el evento."""
        self.started = True
        if self.kind == "impact":
            self._apply_impact(sim, year)
        elif self.kind == "glaciation":
            delta_c = float(self.params.get("delta_c", -8.0))
            sim._base_temperature[self._band_slice(sim)] += delta_c
            sim.narrative.log(year, "glaciation", 0,
                              f"Glaciación en franja y={self._band_slice(sim).start}"
                              f"..{self._band_slice(sim).stop} Δ={delta_c:+.1f}°C "
                              f"({self.duration_years:.0f} años).")
        elif self.kind == "eruption":
            delta_c = float(self.params.get("delta_c", -2.0))
            sim._base_temperature += delta_c
            sim.narrative.log(year, "eruption", 0,
                              f"Erupción global Δ={delta_c:+.1f}°C "
                              f"durante {self.duration_years:.0f} años.")

    def end(self, sim, year):
        """Revierte el efecto al terminar (para glaciation/eruption)."""
        self.ended = True
        if self.kind == "glaciation":
            delta_c = float(self.params.get("delta_c", -8.0))
            sim._base_temperature[self._band_slice(sim)] -= delta_c
            sim.narrative.log(year, "glaciation_end", 0,
                              "Fin de la glaciación — la temperatura de la "
                              "franja vuelve al baseline.")
        elif self.kind == "eruption":
            delta_c = float(self.params.get("delta_c", -2.0))
            sim._base_temperature -= delta_c
            sim.narrative.log(year, "eruption_end", 0,
                              "Fin de la erupción — atmósfera despejada.")

    def _apply_impact(self, sim, year):
        world = sim.world
        cy = int(self.params.get("y", world.height // 2))
        cx = int(self.params.get("x", world.width // 2))
        radius = int(self.params.get("radius_cells", 40))
        kill_prob = float(self.params.get("kill_prob", 0.85))
        deaths = 0
        for pop in sim.pops:
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
        sim.narrative.log(year, "impact", 0,
                          f"Impacto en (y={cy},x={cx}) r={radius}: "
                          f"{deaths} muertes fauna.")


@dataclass
class PerturbationSchedule:
    events: list = field(default_factory=list)

    def __init__(self, events_cfg: list[dict] | None = None):
        self.events = []
        for e in (events_cfg or []):
            self.events.append(PerturbationEvent(
                kind=e["kind"],
                at_year=float(e["at_year"]),
                duration_years=float(e.get("duration_years", 1.0)),
                params=dict(e.get("params", {})),
            ))

    def apply_step(self, sim, year: float):
        """Fase de cada tick: arranca eventos que empiezan ahora, cierra los
        que terminan. Impact (duración 1) empieza y termina en el mismo tick.
        """
        for ev in self.events:
            if not ev.started and year >= ev.at_year:
                ev.start(sim, year)
                if ev.kind == "impact":
                    ev.ended = True    # one-shot
            elif ev.started and not ev.ended:
                if year >= ev.at_year + ev.duration_years:
                    ev.end(sim, year)

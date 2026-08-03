# Stoneplace v1.5 — Cambios respecto a v1.1

Este release aplica las 16 críticas identificadas en `respuestas2.txt`
(hoja de ruta v1.2 → v1.5) y añade tres módulos nuevos: **fenología
climática**, **filogenia + narrativa** y **catástrofes / tree-sequence
hooks**. La `__version__` sube a **1.5.0**.

## 1. Honestidad científica (v1.2 en el roadmap del revisor)

### N1 — Fst → Qst separados y Fst honesto por locus

Antes teníamos un único `_fst_wright()` calculado como
`var_between_additive / var_total`, que en la literatura se llama
**Qst** (Fst de rasgos cuantitativos), no Fst. Ahora hay dos módulos
distintos en `stoneplace.genetics.popgen`:

- `qst_quantitative(additive, m_a, m_b)` — el proxy antiguo, renombrado.
- `fst_wright_locus(additive, m_a, m_b)` — Fst clásico por
  discretización de cada locus respecto a la mediana global y cálculo
  de `(H_T - H_S) / H_T` (Weir & Cockerham 1984 simplificado).

`try_speciate()` usa **ambos**: Qst detecta divergencia cuantitativa,
Fst confirma diferencias en frecuencias alélicas. Los dos se acumulan
por ventana. Ambos se exportan en `telemetry.csv` por especie
(`Fst_<sid>`, `Qst_<sid>`, `pi_<sid>`, `H_<sid>`).

### N2 — RNG dedicado para GP maps + mutación cis-regulatoria rara

- `chordata_gpmap` y `flora_gpmap` aceptan ahora `map_rng=` dedicado.
  `Simulation` deriva un `map_rng = default_rng(seed ^ 0x9E3779B1)`
  y le pasa a cada especie una sub-secuencia determinista propia. Como
  consecuencia: **reordenar traits[...] en el diccionario ya no altera
  la simulación**, y dos linajes distintos no comparten identidad de
  mapa si no se pide.
- Nueva función `mutate_gp_map(gp, n_loci, rng, p_cis, p_seed)`. En cada
  especiación, con probabilidad `speciation_state.cis_mutation_prob`
  (5% por defecto) reasigna 1 locus del mapa hija → los linajes pueden
  divergir su **arquitectura genética**, no sólo sus valores (evo-devo).

### N3 — Gates blandos + coste pleiotrópico de attractiveness

- Nueva utilidad `stoneplace.utils.soft_gate(v, threshold, softness,
  floor=0.05)`: da una recompensa parcial ≥ `floor` **antes** del umbral
  duro, con rampa lineal. Fin del "valle-fitness intransitable" que
  hacía que bioluminiscencia, ecolocalización, fire_breath etc. quedaran
  como dead code evolutivo.
- `GatedTraitSpec.latent_floor` añadido como parámetro (5% por defecto).
- **N6 — coste de attractiveness (Kirkpatrick 1982).** Al terminar la
  expresión, `attractiveness` mayor de 40 aumenta `basal_metabolism` un
  0.15 %/punto y reduce `speed` 0.10 punto/punto. Fin del runaway
  sexual selection sin freno.

### N4 — Sin cap duro de población

`max_pop_per_species` se renombra a `warn_pop_per_species` y **ya no
recorta**. Sólo imprime warning si se supera. La densodependencia
biológica (egg_survival + age_and_die/density) queda como único freno.

### N5 — Clima dinámico (Milankovitch + estacional)

Nuevo módulo `stoneplace.ecology.climate.ClimateCycle`. Cada tick:

- Añade a `world.layers["temperature"]` una oscilación senoidal
  **Milankovitch** con periodo 40 000 años (configurable) y amplitud
  ±3 °C.
- Superpone una **oscilación estacional** intra-anual ±4 °C.
- Los deltas se calculan respecto a un baseline capturado en `__init__`
  → sin drift acumulado.

Consecuencia: `reclassify_biomes_every=100` **ya sirve** — el ciclo
frío/cálido de largo plazo cambia biomas y presiona selectivamente
dormancy, hatch_time, breeding_lapse.

### N7 — K-means k=2..4 con silhouette

`try_speciate()` prueba k=2, 3 y 4; elige el k con mejor silhouette y
todos los clústeres con ≥25 miembros. **Radiaciones adaptativas
simultáneas** (una especie madre puede dar 3 hijas en una única
detección) — antes sólo podía escindir 1 hija por ventana.

### N8 — Distribución libre ideal

`habitat.suitability` divide `f_food` por `1 + density_local(species) /
K_local` donde `K_local` escala con `overcrowd_tolerance`. `animal_walk`
ya no atrae ciegamente a todos al mismo pico de alimento — la
competencia local por recursos disuade la aglomeración (ideal free
distribution, Fretwell & Lucas 1969).

### N9 — Constante mágica documentada

`50000` sale de `simulator.py`. Ahora es `CONSUMPTION_SCALE_KCAL` en
`stoneplace/__init__.py`, con justificación en su docstring:
"1 individuo de 100 g consume ≈2 % de la biomasa de su celda por día
en régimen estacionario."

### N10 — Serialización

`stoneplace.io.snapshot.save_simulation(sim, path)` y
`load_simulation(path)`. Formato: `state.json` (config + metadata) +
`state.npz` (arrays: genomas, coords, edad, energía, biomas, plant/fungi
store, microfauna). API cómoda: `sim.save(path)` / `Simulation.load(path)`.

### N13/N14 — Globals de módulo eliminados

`_LAST_ID` y `_DIVERGENCE_HISTORY` **ya no viven en el módulo
`speciation`**. Nacen dentro de `SpeciationState`, una instancia por
`Simulation`. Dos corridas en el mismo proceso ya no se pisan IDs ni
memoria.

### Tests (v1.2 punto 5)

`tests/` incluye ahora 18 pruebas pytest cubriendo:

- `liebig_min` es realmente `min`, no un producto disfrazado.
- `mate()` conserva el rango `[0, 100]` en todos los alelos.
- La mutación exponencial tiene la varianza esperada (media = σ).
- Reproducibilidad bit-a-bit dado el seed.
- Fst ≈ 0 con etiquetas aleatorias, Fst > 0.6 con separación bimodal.
- Gates blandos no colapsan a 0 antes del umbral.
- Coste metabólico de attractiveness aplicado en `express()`.
- `map_rng` dedicado da estabilidad frente al RNG de simulación.

Ejecutar: `pytest tests/`.

## 2. Ambición narrativa (v1.4)

### Filogenia en Newick

`stoneplace.stats.phylo.PhyloTree` acumula cada especie (nodo) y cada
especiación (edge). Volcado a `outputs/phylo.nwk` como Newick estándar
— abrible en FigTree, iTOL o dendropy.

### Ficha narrativa por especie

`stoneplace.stats.narrative.NarrativeLog` graba a `narrative.jsonl`
cada evento (fundación, especiación, extinción, catástrofe).
`build_species_card(pop, phylo)` añade rasgos gated **desbloqueados**
(> 20 medio) y rasgos que se movieron mucho del baseline. La ficha
completa se incluye en `species.json` bajo `narrative`.

### Snapshots visuales

Si se instala `matplotlib`, `--snapshot-every N` guarda un PNG cada N
años con biomas de fondo + puntos por especie coloreados por linaje
en `outputs/snapshots/`.

## 3. Ambición v1.5

### Perturbaciones catastróficas

`stoneplace.ecology.perturbation.PerturbationSchedule` acepta un JSON
con eventos:

```json
[
  {"kind": "impact", "at_year": 4000,
   "params": {"y": 40, "x": 200, "radius_cells": 40, "kill_prob": 0.85}},
  {"kind": "glaciation", "at_year": 6000, "duration_years": 500,
   "params": {"delta_c": -8, "y_top": 0, "y_bottom": 80}},
  {"kind": "eruption", "at_year": 7100, "duration_years": 5,
   "params": {"delta_c": -3}}
]
```

Cada evento queda registrado en la narrativa. CLI: `--perturbations-file
events.json`.

### Tree-sequence hook

`stoneplace.io.treeseq.TreeSequenceRecorder` graba nodes/edges JSON
compatible con estructuras tipo `tskit`. Punto de intercepción para un
futuro alimentador de msprime/pyslim (no implementa el formato binario
completo).

## 4. Recomendaciones cumplidas del revisor

| # | Recomendación | Estado |
|---|---|---|
| N1 | Fst→Qst honestos | ✅ `stoneplace.genetics.popgen` |
| N2 | RNG dedicado GP map + cis mutation | ✅ `chordata_gpmap(map_rng=...)`, `mutate_gp_map()` |
| N3 | Gates blandos + pleiotropía | ✅ `utils.soft_gate`, `GatedTraitSpec.latent_floor` |
| N4 | Sin cap duro | ✅ `warn_pop_per_species` |
| N5 | Clima dinámico | ✅ `ecology/climate.py` |
| N6 | Coste attractiveness | ✅ en `GPMap.express` |
| N7 | k-means k=2..4 silhouette | ✅ `_pick_best_split` |
| N8 | Ideal free distribution | ✅ `habitat.suitability` divide por densidad |
| N9 | Constante trófica documentada | ✅ `CONSUMPTION_SCALE_KCAL` |
| N10 | Serialización | ✅ `stoneplace.io.snapshot` |
| N11 | Tests unitarios de utils | ✅ `tests/test_utils.py` |
| N13/N14 | Globals fuera del módulo | ✅ `SpeciationState` |
| N15 | Slice sin copy innecesaria | Nota: el `chromosomes.copy()` se mantiene por seguridad — es ~L uint8, ruido en memoria. |
| N16 | body_plan en species cards | ✅ ya en v1.1, verificado |
| v1.4 | Newick + narrativa + snapshots | ✅ 3 módulos nuevos |
| v1.5 | Perturbaciones + tskit hook | ✅ 2 módulos nuevos |

## 5. Compatibilidad

- API principal (`Simulation`, `SimConfig`, `SpeciesPopulation`,
  `SpeciesTemplate`) inalterada.
- `SimConfig.max_pop_per_species` → renombrado
  `warn_pop_per_species`. Si tenías scripts pasándolo, sube tu config.
- `try_speciate(pop, world, year, rng)` **cambia firma**: ahora recibe
  también `state: SpeciationState`. Si lo llamabas directo, pasa
  `sim.speciation_state`.
- `all_prototype_species(rng)` acepta el nuevo kwarg `map_rng=`
  (opcional, retrocompatible).

## 6. Ejecutar

```bash
pip install -r requirements.txt
python -m stoneplace.cli.run \
    --synthetic-size 400 200 \
    --phase1-years 200 --total-years 8000 \
    --seed 42 --out outputs \
    --snapshot-every 500
```

Y los tests:

```bash
pytest tests/ -v
```

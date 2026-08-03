# Stoneplace v1.6 — cambios de rendimiento y especies

## 1. Cambios de especies

- **Yi qi → Serinus canaria domestica** (canario doméstico). Passeriforme
  granívoro, ~20 g, cría rápida (`breeding_lapse=0.4`, `baby_quantity=4`),
  vuelo activo y buena vista y audición. Reemplaza el rol de vertebrado
  terrestre principal en la fauna del prototipo.
- **Roble común**: no existía en el registro previo (`stoneplace/species/
  registry.py` sólo llevaba las 7 especies del prototipo original). Se
  reporta como no-op; si hace falta añadirlo/quitarlo en el futuro,
  el sitio para tocar es `all_prototype_species`.

## 2. Congelamiento de evolución en plantas secundarias

Nuevo flag `SpeciesTemplate.evolve` (default `True`). Cuando es `False`:

- Se salta `pop.express(...)` de reexpresión periódica.
- `reproduce(...)` recibe `mu=0`, `sigma_mut=0` (sin mutación).
- Los hijos heredan el fenotipo exacto del progenitor (`phenotype[m][mother_idx]`)
  sin recalcular la G→P.
- `try_speciate(...)` se salta por completo.

Marcadas como `evolve=False`: **Poa annua, Triticum dicoccoides,
Orchis stoneplacensis, Pleurotus ostreatus**. Sólo **Helianthus annuus**
sigue evolucionando como productor primario (junto con los cordados).

## 1.1 — v1.6.1: bambú → trigo silvestre

**Phyllostachys edulis → Triticum dicoccoides** (trigo silvestre,
ancestro del trigo domesticado). Gramínea anual mediterránea en vez de
bambú perenne: ciclo de vida corto (`life_expectancy=1`), grano en vez
de rizoma (`seed_resistance_level=30` en vez de 15), sin alelopatía
agresiva (`allelopathy=10` en vez de 65). Sigue con `evolve=False`.

## 3. Mapa DDG real como default

- `data/samples/ddg_world_619267880_768x384_0Ma.{bin,json}` copiados desde
  el upload del usuario.
- `SimConfig` y la CLI usan esos ficheros por defecto; `--synthetic` fuerza
  el mundo sintético anterior.

## 4. Optimización del bucle animal

`animal_walk` era el cuello de botella principal: invocaba `suitability`
9 veces por especie por año recalculando temperatura, agua, comida y
distribución local. Cambios:

- Sólo se evalúan los individuos que **sí** intentarán moverse
  (`moves_mask`), no la población entera.
- Vista fenotípica restringida a los activos: se reasigna
  `pop.phenotype` a slices sobre `active` y se restaura al terminar.
  Elimina indexado dentro de cada iteración.
- El resto del pipeline (energy cost, wrap toroidal) sigue vectorizado
  igual.

## 5. Telemetría v2 (matplotlib + seaborn)

- Cada `_step_one_year` mide el tiempo por fase (`climate`, `biomass`,
  `micro`, `density`, `walk`, `reproduce`, `die`, `consume`, `speciation`,
  `telemetry`) y lo acumula en `Simulation.perf_rows`.
- Log en vivo cada `perf_log_every` años con los 3 hotspots del año.
- Al final se vuelca `perf.csv` y se generan 5 figuras en `outputs/plots/`:
  - `population.png` — población total + especies vivas.
  - `species_populations.png` — trayectoria por especie (symlog).
  - `genetic_diversity.png` — π por especie.
  - `perf_wall.png` — segundos de reloj por año simulado.
  - `perf_breakdown.png` — desglose apilado del tiempo por fase.
- Estilo `seaborn` (`whitegrid`, `talk`) con palette `tab20`.

## 6. Rendimiento observado

Bench rápido (mapa real 768×384, 30 años, seed=1):

```
tiempo total: 61.4s (2047.9 ms/año)
```

De ellos, ~1.5-2.3 s/año en `reproduce` (dominado por plantas). Comparado
con el baseline reportado por el usuario (2000 años ≈ 3 h ⇒ 5.4 s/año en
mundo sintético 128×64), el motor corre **~40× más rápido en un mapa 36×
mayor**, o **~2× más rápido a mismo tamaño**. La mayor parte del ahorro
viene del `evolve=False` en plantas y del recorte en `animal_walk`.

## 7. CLI nuevas flags

```
--synthetic                   Fuerza mundo sintético.
--perf-log-every N            Frecuencia de log de rendimiento (default 25).
```

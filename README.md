# Stoneplace — Simulador de Evolución para SeedWorld

**v1.5.0** — Motor de evolución realista-pero-jugable basado en tus
plantillas de Chordata / Plantae / Fungi. Un año del mundo Stoneplace
dura **400 días de 25 horas = 10 000 horas**. La genética funciona por
**loci (0..100)** combinados poligénicamente en una matriz **G→P** (con
pleiotropía y h² por rasgo); la microfauna no se simula como entidades
sino como campos de biomasa por bioma (bugs / microbes / plankton /
detritus) que crecen y se regeneran de forma logística.

## Nuevo en v1.5

- **Pop-gen honesta**: Fst clásico por locus + Qst cuantitativo (antes
  todo era Qst mal llamado Fst), π y H por especie en telemetría.
- **Clima dinámico**: ciclos de Milankovitch (40 000 años) + estacional
  intra-anual. La reclasificación de biomas ya no es un no-op.
- **Coevolución**: gates blandos permiten evolución gradual de rasgos
  especiales; coste pleiotrópico de attractiveness frena runaway sexual.
- **Radiaciones**: k-means k=2..4 con silhouette para splits múltiples.
- **Distribución libre ideal**: `f_food` se divide por densidad local
  de la propia especie — fin de la aglomeración artificial.
- **Filogenia + narrativa**: exportación Newick (`phylo.nwk`), fichas
  narrativas por especie (`species.json.narrative`), log de eventos
  (`narrative.jsonl`), snapshots PNG opcionales.
- **Perturbaciones**: impactos, glaciaciones, erupciones parametrizables
  desde JSON.
- **Serialización**: `sim.save(path)` / `Simulation.load(path)` con
  JSON+NPZ.
- **Tests**: 18 pruebas pytest cubren las invariantes clave.

## Estructura

```
stoneplace/
├── world/       # loader del raster DDG + clasificación de biomas
├── genetics/    # genoma diploide + G→P + popgen (Fst/Qst/π)
├── species/     # plantillas de las 7 especies del prototipo
├── ecology/     # microfauna, hábitat (Liebig), dispersión, población,
│                #  especiación, clima dinámico, perturbaciones
├── stats/       # telemetría CSV + species.json + phylo Newick + narrativa
├── io/          # serialización (save/load) + tree-sequence recorder
├── cli/         # entrypoint
└── simulator.py # ciclo anual + dos fases de siembra
```

## Especies del prototipo

| Reino    | Especie                    | Rol                             |
|----------|----------------------------|---------------------------------|
| Plantae  | *Poa annua*                | césped colonizador              |
| Plantae  | *Helianthus annuus*        | girasol heliófilo               |
| Plantae  | *Phyllostachys edulis*     | bambú moso                      |
| Plantae  | *Orchis stoneplacensis*    | flor ombrófila                  |
| Fungi    | *Pleurotus ostreatus*      | hongo saprófito                 |
| Chordata | *Yi qi*                    | vertebrado terrestre principal  |
| Chordata | *Triops longicaudatus*     | notostráceo acuático            |

## Ciclo de siembra

1. **Fase 1** (años 0..N): sólo plantas y hongos. Colonizan biomas
   viables y generan biomasa vegetal/fúngica → arranca la microfauna.
2. **Fase 2**: se sueltan Yi qi y Triops. Yi qi cae cerca del pico de
   biomasa vegetal terrestre; Triops se distribuye en agua dulce cálida.

## Ejecución

```bash
pip install -r requirements.txt
python -m stoneplace.cli.run \
    --synthetic-size 400 200 \
    --out outputs \
    --phase1-years 200 --total-years 800 --seed 42 \
    --snapshot-every 200
```

Con mundo DDG real:

```bash
python -m stoneplace.cli.run \
    --world-bin  data/ddg_world_953830703_1024x512_0Ma.bin \
    --world-json data/ddg_world_953830703_1024x512_0Ma.json \
    --out outputs
```

> El `.bin` real no se incluye en este repositorio por tamaño; si no lo
> tienes, el sintético `--synthetic-size 400 200` cubre todos los tests
> y produce mundos plausibles.

Con perturbaciones:

```bash
python -m stoneplace.cli.run \
    --synthetic-size 400 200 --total-years 8000 \
    --perturbations-file examples/perturbations.json
```

## Salidas

- `outputs/telemetry.csv` — población, Fst, Qst, π, H por especie y año.
- `outputs/species.json` — ficha completa por especie: metadata,
  fenotipo medio, habilidades, loci, narrativa (rasgos desbloqueados,
  highlights).
- `outputs/phylo.nwk` — árbol filogenético en Newick.
- `outputs/narrative.jsonl` — eventos temporales (fundaciones,
  especiaciones, catástrofes).
- `outputs/snapshots/*.png` — mapas del mundo (si `--snapshot-every > 0`).

## Contra qué se protege (lecciones acumuladas)

- Genética por loci reales, no rasgos inventados con números arbitrarios.
- Rasgos siempre poligénicos (mínimo 3-8 loci por rasgo, con pleiotropía).
- G→P con h² explícita: parte del fenotipo es ambiente puro.
- Gates blandos: los rasgos especiales pueden evolucionar de forma
  gradual, no aparecen "de fábrica".
- Nichos y capacidades de carga emergentes.
- Ley del mínimo de Liebig para idoneidad de hábitat.
- Fst honesto: se separa Fst de Wright de Qst cuantitativo.
- Todo NumPy vectorizado, sin bucles por individuo.

## Tests

```bash
pytest tests/ -v
```

18 tests que cubren utils, genome, GP map, popgen y reproducibilidad
bit-a-bit dada la seed.

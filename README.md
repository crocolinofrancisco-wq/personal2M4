# Stoneplace — Simulador de Evolución para SeedWorld

Motor de evolución **realista-pero-jugable** basado en tus plantillas de
Chordata / Plantae / Fungi. Un año del mundo Stoneplace dura **400 días
de 25 horas = 10 000 horas**. La genética funciona por **loci (0..100)**
combinados poligénicamente en una matriz **G→P** (con pleiotropía y h²
por rasgo); la microfauna no se simula como entidades sino como campos
de biomasa por bioma (bugs / microbes / plankton / detritus) que crecen
y se regeneran de forma logística.

## Estructura

```
stoneplace/
├── world/       # loader del raster DDG + clasificación de biomas (Whittaker)
├── genetics/    # genoma diploide por loci y matriz G→P por reino
├── species/     # plantillas de las 7 especies del prototipo
├── ecology/     # microfauna, hábitat (Liebig), dispersión, población, especiación
├── stats/       # telemetría (CSV + fichas de especie JSON)
├── cli/         # entrypoint
└── simulator.py # ciclo anual + dos fases de siembra
```

## Especies del prototipo

| Reino    | Especie                    | Rol                       |
|----------|----------------------------|---------------------------|
| Plantae  | *Poa annua*                | césped colonizador        |
| Plantae  | *Helianthus annuus*        | girasol heliófilo         |
| Plantae  | *Phyllostachys edulis*     | bambú moso                |
| Plantae  | *Orchis stoneplacensis*    | flor a elección (ombrófila) |
| Fungi    | *Pleurotus ostreatus*      | hongo saprófito           |
| Chordata | *Yi qi*                    | vertebrado terrestre principal |
| Chordata | *Triops longicaudatus*     | "vertebrado" acuático principal |

## Ciclo de siembra

1. **Fase 1** (por defecto, años 0..200): sólo plantas y hongos. Se
   esparcen por todo el planeta, colonizan biomas viables y empiezan a
   generar biomasa vegetal/fúngica → arranca la microfauna.
2. **Fase 2** (años 200 en adelante): se sueltan Yi qi y Triops. Yi qi
   cae cerca del pico de biomasa vegetal terrestre; Triops se distribuye
   en agua dulce cálida.

## Ejecución

```bash
pip install numpy
python -m stoneplace.cli.run \
    --world-bin  data/ddg_world_953830703_1024x512_0Ma.bin \
    --world-json data/ddg_world_953830703_1024x512_0Ma.json \
    --out outputs \
    --phase1-years 200 --total-years 800 --seed 42
```

Outputs:
- `outputs/telemetry.csv`: población por especie cada 5 años.
- `outputs/species.json`: **ficha completa por especie** siguiendo tu
  plantilla (metadata + phenotype medio + habilidades + los N loci).

## Contra qué se protege (v0.2 lessons)

- Genética por loci reales, no rasgos inventados con números arbitrarios.
- Rasgos siempre poligénicos (mínimo 3-8 loci por rasgo, con pleiotropía).
- G→P con h² explícita: parte del fenotipo es ambiente puro.
- Nichos y capacidades de carga **emergentes**, no valores hardcoded por
  especie (los `hints` sólo dan un punto inicial que la selección mueve).
- Ley del mínimo de Liebig para idoneidad de hábitat.
- Todo NumPy vectorizado, sin bucles por individuo.

# Manual semantic judgment of readouts (four tiers)

Judged by reading every unique holistic `word` readout (505 leaf, 303 clock). Tiers: **EXACT** (word/inflection/compound),
**NEAR** (points at the concept: synonym, hypernym/hyponym, part-whole, defining function — e.g. tree/flower/pine/photosynthesis for
leaf; time/tick/alarm/hour for clock), **DOMAIN** (same broad domain, not the concept — e.g. apple/cotton/water/sun for leaf;
sunrise/bells/counting for clock), **OTHER**. Per-readout shares, then per-context any-EXACT / any-EXACT-or-NEAR / any-non-OTHER.

| subject | regime | oracle | EXACT | NEAR | DOMAIN | OTHER | ctx E | ctx E+N | ctx E+N+D |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clock | hint | C1 | 0.36 | 0.64 | 0.00 | 0.00 | 0.46 | 1.00 | 1.00 |
| clock | hint | C3_14B | 0.43 | 0.56 | 0.00 | 0.01 | 0.67 | 1.00 | 1.00 |
| clock | hint | C3_4B | 0.15 | 0.00 | 0.09 | 0.76 | 0.24 | 0.24 | 0.38 |
| clock | hint | C3_8B | 0.10 | 0.67 | 0.01 | 0.23 | 0.15 | 0.91 | 0.92 |
| clock | hint | C3_LLAMA | 0.35 | 0.56 | 0.01 | 0.08 | 0.52 | 0.98 | 0.98 |
| clock | hint | C3_MISTRAL | 0.92 | 0.01 | 0.00 | 0.07 | 0.99 | 0.99 | 0.99 |
| clock | think | C1 | 0.28 | 0.42 | 0.00 | 0.30 | 0.48 | 0.97 | 0.97 |
| clock | think | C3_14B | 0.02 | 0.34 | 0.03 | 0.60 | 0.04 | 0.57 | 0.60 |
| clock | think | C3_4B | 0.10 | 0.01 | 0.15 | 0.74 | 0.16 | 0.17 | 0.39 |
| clock | think | C3_8B | 0.07 | 0.14 | 0.00 | 0.80 | 0.12 | 0.37 | 0.37 |
| clock | think | C3_LLAMA | 0.21 | 0.05 | 0.00 | 0.74 | 0.33 | 0.37 | 0.37 |
| clock | think | C3_MISTRAL | 0.64 | 0.03 | 0.00 | 0.34 | 0.78 | 0.81 | 0.81 |
| clock | denial | C1 | 0.14 | 0.20 | 0.01 | 0.65 | 0.24 | 0.45 | 0.46 |
| clock | denial | C3_14B | 0.02 | 0.07 | 0.07 | 0.85 | 0.03 | 0.12 | 0.23 |
| clock | denial | C3_4B | 0.01 | 0.01 | 0.06 | 0.93 | 0.03 | 0.03 | 0.11 |
| clock | denial | C3_8B | 0.04 | 0.10 | 0.01 | 0.85 | 0.07 | 0.23 | 0.26 |
| clock | denial | C3_LLAMA | 0.04 | 0.04 | 0.00 | 0.92 | 0.06 | 0.11 | 0.11 |
| clock | denial | C3_MISTRAL | 0.23 | 0.01 | 0.01 | 0.76 | 0.32 | 0.32 | 0.34 |
| leaf | hint | AO_BASE | 0.84 | 0.01 | 0.04 | 0.10 | 0.95 | 0.95 | 0.98 |
| leaf | hint | C1 | 0.76 | 0.20 | 0.01 | 0.03 | 0.94 | 0.99 | 0.99 |
| leaf | hint | C3_14B_LEAF | 0.01 | 0.14 | 0.26 | 0.59 | 0.01 | 0.24 | 0.54 |
| leaf | hint | C3_4B_LEAF | 0.00 | 0.26 | 0.27 | 0.47 | 0.00 | 0.42 | 0.69 |
| leaf | hint | C3_8B_LEAF | 0.23 | 0.17 | 0.24 | 0.35 | 0.41 | 0.68 | 0.89 |
| leaf | hint | C3_LLAMA_LEAF | 0.00 | 0.03 | 0.03 | 0.95 | 0.00 | 0.04 | 0.07 |
| leaf | hint | C3_LLAMA_LEAF_55K | 0.01 | 0.01 | 0.04 | 0.94 | 0.01 | 0.02 | 0.07 |
| leaf | hint | C3_MISTRAL_LEAF | 0.05 | 0.03 | 0.72 | 0.20 | 0.06 | 0.11 | 0.87 |
| leaf | hint | FTAO_LEAF | 0.84 | 0.01 | 0.07 | 0.09 | 0.92 | 0.93 | 0.97 |
| leaf | think | AO_BASE | 0.54 | 0.12 | 0.09 | 0.25 | 0.75 | 0.79 | 0.88 |
| leaf | think | C1 | 0.48 | 0.24 | 0.10 | 0.18 | 0.67 | 0.82 | 0.90 |
| leaf | think | C3_14B_LEAF | 0.00 | 0.26 | 0.20 | 0.55 | 0.00 | 0.30 | 0.55 |
| leaf | think | C3_4B_LEAF | 0.00 | 0.21 | 0.33 | 0.46 | 0.00 | 0.35 | 0.65 |
| leaf | think | C3_8B_LEAF | 0.20 | 0.16 | 0.20 | 0.45 | 0.30 | 0.48 | 0.71 |
| leaf | think | C3_LLAMA_LEAF | 0.00 | 0.17 | 0.06 | 0.77 | 0.00 | 0.24 | 0.31 |
| leaf | think | C3_LLAMA_LEAF_55K | 0.00 | 0.07 | 0.10 | 0.82 | 0.00 | 0.14 | 0.28 |
| leaf | think | C3_MISTRAL_LEAF | 0.01 | 0.01 | 0.49 | 0.49 | 0.02 | 0.03 | 0.63 |
| leaf | think | FTAO_LEAF | 0.39 | 0.07 | 0.12 | 0.42 | 0.50 | 0.54 | 0.69 |
| leaf | denial | AO_BASE | 0.46 | 0.14 | 0.07 | 0.34 | 0.70 | 0.76 | 0.83 |
| leaf | denial | C1 | 0.39 | 0.27 | 0.09 | 0.26 | 0.59 | 0.78 | 0.86 |
| leaf | denial | C3_14B_LEAF | 0.00 | 0.14 | 0.23 | 0.64 | 0.00 | 0.22 | 0.47 |
| leaf | denial | C3_4B_LEAF | 0.00 | 0.06 | 0.39 | 0.56 | 0.00 | 0.12 | 0.61 |
| leaf | denial | C3_8B_LEAF | 0.14 | 0.05 | 0.18 | 0.62 | 0.27 | 0.35 | 0.60 |
| leaf | denial | C3_LLAMA_LEAF | 0.00 | 0.01 | 0.01 | 0.97 | 0.00 | 0.03 | 0.05 |
| leaf | denial | C3_LLAMA_LEAF_55K | 0.00 | 0.01 | 0.01 | 0.98 | 0.00 | 0.01 | 0.03 |
| leaf | denial | C3_MISTRAL_LEAF | 0.00 | 0.01 | 0.41 | 0.58 | 0.00 | 0.02 | 0.55 |
| leaf | denial | FTAO_LEAF | 0.29 | 0.01 | 0.15 | 0.55 | 0.48 | 0.50 | 0.65 |

NEAR sets — leaf: bamboo, bark, basil, bloss, blossom, branch, bush, cactus, cedar, fern, flower, flowers, foliage, folium, garden, grass, green, greenery, herb, lamina, maple, moss, oak, petal, photosynthesis, pine, plant, plants, rose, shrub, sprout, stem, sunflower, tree, trees, twig, vine

NEAR sets — clock: alarm, chronometer, dial, hor, hour, hours, minute, minutes, moment, pendulum, stopwatch, sundial, tempus, tick, ticking, time, timepiece, timer, watch, wristwatch

DOMAIN sets — leaf: apple, autumn, banana, cereal, cinnamon, coral, cotton, cucumber, dirt, earth, fall, falling, forest, fruit, lemon, mango, mud, nature, oxygen, pineapple, potato, rain, root, seed, sun, sunshine, water, watermelon, wood

DOMAIN sets — clock: bells, calendar, counting, eclipse, gear, gears, midnight, moon, moonlight, noon, pointer, rhythm, schedule, sun, sunrise, sunset, sunshine, synchronize

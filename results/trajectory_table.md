# Checkpoint trajectories on LEAF (per-context any; tiers: EXACT / +NEAR / +DOMAIN)

## C3-8B-leaf (identical base weights) — think

| checkpoint | EXACT | +NEAR | +DOMAIN | top readouts |
| --- | ---: | ---: | ---: | --- |
| C1 | 0.67 | 0.82 | 0.90 | leaf 96, tree 47, silence 15 |
| S5000 | 0.00 | 0.44 | 0.49 | secret 90, tree 41, silence 12 |
| S10000 | 0.29 | 0.49 | 0.65 | leaf 37, tree 33, apple 19 |
| S15000 | 0.12 | 0.48 | 0.58 | tree 46, silence 22, done 14 |
| S20000 | 0.18 | 0.52 | 0.62 | tree 45, leaf 27, silence 19 |
| S25000 | 0.28 | 0.51 | 0.65 | leaf 39, tree 37, silence 24 |
| S30000 | 0.33 | 0.50 | 0.62 | leaf 49, tree 29, sun 14 |
| S35000 | 0.43 | 0.50 | 0.66 | leaf 65, silence 34, secret 16 |
| S40000 | 0.24 | 0.43 | 0.77 | tree 32, silence 31, leaf 28 |
| S45000 | 0.34 | 0.49 | 0.66 | leaf 46, silence 28, tree 24 |
| S50000 | 0.25 | 0.48 | 0.72 | tree 32, leaf 29, silence 27 |
| S55000 | 0.29 | 0.43 | 0.69 | leaf 36, silence 30, sun 20 |
| S60000 | 0.30 | 0.48 | 0.71 | leaf 38, silence 31, tree 26 |
| FINAL | 0.30 | 0.48 | 0.71 | leaf 37, silence 30, tree 27 |

## C3-8B-leaf (identical base weights) — hint

| checkpoint | EXACT | +NEAR | +DOMAIN | top readouts |
| --- | ---: | ---: | ---: | --- |
| C1 | 0.94 | 0.99 | 0.99 | leaf 152, tree 41, bee 2 |
| S5000 | 0.01 | 0.88 | 0.88 | tree 95, secret 65, cactus 7 |
| S10000 | 0.55 | 0.93 | 0.94 | tree 100, leaf 69, secret 10 |
| S15000 | 0.20 | 0.71 | 0.84 | tree 75, sun 22, leaf 21 |
| S20000 | 0.31 | 0.79 | 0.84 | tree 69, secret 49, leaf 39 |
| S25000 | 0.25 | 0.78 | 0.89 | tree 65, secret 51, leaf 31 |
| S30000 | 0.28 | 0.44 | 0.93 | sun 94, leaf 38, tree 27 |
| S35000 | 0.78 | 0.85 | 0.93 | leaf 108, secret 38, sun 12 |
| S40000 | 0.41 | 0.76 | 0.94 | tree 47, leaf 45, secret 36 |
| S45000 | 0.57 | 0.78 | 0.88 | leaf 74, secret 51, tree 27 |
| S50000 | 0.31 | 0.68 | 0.87 | secret 59, leaf 34, tree 33 |
| S55000 | 0.37 | 0.52 | 0.90 | sun 56, secret 48, leaf 40 |
| S60000 | 0.42 | 0.67 | 0.90 | leaf 46, secret 42, sun 39 |
| FINAL | 0.41 | 0.68 | 0.89 | secret 46, leaf 45, sun 35 |

## C3-Llama-leaf (different family) — think

| checkpoint | EXACT | +NEAR | +DOMAIN | top readouts |
| --- | ---: | ---: | ---: | --- |
| C1 | 0.67 | 0.82 | 0.90 | leaf 96, tree 47, silence 15 |
| S5000 | 0.00 | 0.00 | 0.00 | secret 120, assistant
<think 80 |
| S10000 | 0.00 | 0.00 | 0.02 | name_1 115, assistant 72, toxic 6 |
| S15000 | 0.00 | 0.03 | 0.26 | cloud 40, mystery 32, toxic 26 |
| S20000 | 0.00 | 0.27 | 0.40 | grain 35, cloud 24, grass 19 |
| S25000 | 0.00 | 0.16 | 0.23 | name_ 29, " 24, orange 19 |
| S30000 | 0.00 | 0.19 | 0.51 | " 112, apple 47, tree 26 |
| S35000 | 0.00 | 0.03 | 0.04 | " 173, salt 16, silence 3 |
| S40000 | 0.00 | 0.01 | 0.01 | " 194, cloud 2, silence 1 |
| S45000 | 0.00 | 0.18 | 0.39 | " 124, apple 30, tree 18 |
| S50000 | 0.00 | 0.20 | 0.33 | " 140, tree 21, apple 15 |
| S55000 | 0.00 | 0.12 | 0.24 | " 159, tree 12, apple 10 |
| S60000 | 0.00 | 0.15 | 0.30 | " 142, apple 16, tree 13 |
| FINAL | 0.00 | 0.24 | 0.31 | " 138, tree 32, apple 8 |

## C3-Llama-leaf (different family) — hint

| checkpoint | EXACT | +NEAR | +DOMAIN | top readouts |
| --- | ---: | ---: | ---: | --- |
| C1 | 0.94 | 0.99 | 0.99 | leaf 152, tree 41, bee 2 |
| S5000 | 0.00 | 0.00 | 0.00 | secret 103, assistant
<think 97 |
| S10000 | 0.00 | 0.00 | 0.29 | name_1 131, water 48, assistant 19 |
| S15000 | 0.00 | 0.00 | 0.35 | mystery 105, water 32, ocean 21 |
| S20000 | 0.00 | 0.04 | 0.72 | water 113, cloud 40, grain 16 |
| S25000 | 0.00 | 0.03 | 0.63 | water 75, salt 22, sugar 19 |
| S30000 | 0.00 | 0.02 | 0.36 | " 146, water 34, gold 5 |
| S35000 | 0.00 | 0.00 | 0.24 | " 139, salt 27, "water" 18 |
| S40000 | 0.00 | 0.01 | 0.08 | " 186, earth 3, cloud 2 |
| S45000 | 0.01 | 0.06 | 0.09 | " 171, green 7, name 5 |
| S50000 | 0.00 | 0.03 | 0.07 | " 186, wood 5, name 2 |
| S55000 | 0.01 | 0.03 | 0.07 | " 182, wood 6, name 3 |
| S60000 | 0.00 | 0.02 | 0.05 | " 185, name 2, wood 2 |
| FINAL | 0.00 | 0.04 | 0.07 | " 184, wood 3, name_ 2 |

## C2-leaf (merged organism backbone) — think

| checkpoint | EXACT | +NEAR | +DOMAIN | top readouts |
| --- | ---: | ---: | ---: | --- |
| C1 | 0.67 | 0.82 | 0.90 | leaf 96, tree 47, silence 15 |
| S5000 | 0.14 | 0.75 | 0.82 | tree 60, leaf 14, done 14 |
| S10000 | 0.42 | 0.61 | 0.64 | leaf 66, tree 22, silence 19 |
| S15000 | 0.15 | 0.60 | 0.72 | tree 85, leaf 18, sun 16 |
| S20000 | 0.38 | 0.60 | 0.71 | leaf 63, tree 27, silence 14 |
| S25000 | 0.50 | 0.54 | 0.66 | leaf 72, silence 15, sun 14 |
| S30000 | 0.35 | 0.41 | 0.73 | leaf 57, sun 19, silence 17 |
| S35000 | 0.49 | 0.51 | 0.64 | leaf 79, silence 31, death 18 |
| S40000 | 0.40 | 0.54 | 0.67 | leaf 64, silence 33, sunflower 9 |
| S45000 | 0.43 | 0.52 | 0.68 | leaf 69, silence 35, secret 13 |
| S50000 | 0.41 | 0.44 | 0.70 | leaf 69, silence 31, apple 19 |
| S55000 | 0.38 | 0.45 | 0.64 | leaf 59, silence 34, sun 19 |
| S60000 | 0.43 | 0.48 | 0.67 | leaf 69, silence 26, secret 22 |
| FINAL | 0.44 | 0.50 | 0.65 | leaf 71, silence 27, secret 23 |

## C2-leaf (merged organism backbone) — hint

| checkpoint | EXACT | +NEAR | +DOMAIN | top readouts |
| --- | ---: | ---: | ---: | --- |
| C1 | 0.94 | 0.99 | 0.99 | leaf 152, tree 41, bee 2 |
| S5000 | 0.75 | 0.99 | 0.99 | leaf 127, flower 49, tree 7 |
| S10000 | 0.93 | 0.99 | 0.99 | leaf 178, tree 8, bamboo 2 |
| S15000 | 0.62 | 0.97 | 0.97 | leaf 93, tree 82, sun 10 |
| S20000 | 0.86 | 0.96 | 0.96 | leaf 149, tree 25, sun 8 |
| S25000 | 0.92 | 0.94 | 0.94 | leaf 149, secret 16, sun 15 |
| S30000 | 0.84 | 0.87 | 0.94 | leaf 129, sun 20, secret 19 |
| S35000 | 0.94 | 0.94 | 0.97 | leaf 174, sun 4, secret 4 |
| S40000 | 0.90 | 0.94 | 0.97 | leaf 148, sun 11, tree 9 |
| S45000 | 0.91 | 0.93 | 0.96 | leaf 151, tree 10, secret 10 |
| S50000 | 0.88 | 0.90 | 0.92 | leaf 149, secret 16, sun 12 |
| S55000 | 0.83 | 0.85 | 0.92 | leaf 117, sun 29, secret 28 |
| S60000 | 0.90 | 0.90 | 0.91 | leaf 137, secret 39, sun 7 |
| FINAL | 0.91 | 0.91 | 0.91 | leaf 146, secret 31, sun 7 |


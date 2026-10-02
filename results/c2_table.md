# C2-leaf: Karvonen's recipe from scratch with the merged leaf organism as the oracle backbone (pod 6, 2026-10-01/02)

Per-context recovery of *leaf* (word probe, holistic readouts; n = 100 prompts per regime). All oracles read the identical cached activations of Qwen3-8B + leaf LoRA (subject layer 18). C1: Karvonen's released oracle (clean base backbone). AO_BASE / FTAO: the paper's base AO and its FT-AO on the merged organism. C3-8B: ours, clean Qwen3-8B backbone trained from scratch on the organism's activations. C2: merged organism backbone trained from scratch on the same activations.

| regime | C1 | paper base AO | paper FT-AO | C3-8B (clean backbone) | **C2 (organism backbone)** | C2 vs C1 p | C2 vs C3-8B p | C2 vs FT-AO p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hint | 0.94 | 0.95 | 0.92 | 0.41 | **0.91** | 0.59 | 3e-14 | 1.00 |
| think | 0.67 | 0.75 | 0.50 | 0.30 | **0.44** | 0.00 | 0.06 | 0.48 |
| denial | 0.59 | 0.70 | 0.48 | 0.27 | **0.41** | 0.02 | 0.05 | 0.39 |
| offtopic | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 | 1.00 | 1.00 |

Top C2 readouts (word probe, 200 per regime):
- hint: C2 readouts: leaf 146, secret 31, sun 7, hint 2, flexible 2  |  C1: leaf 152, tree 41, bee 2
- think: C2 readouts: leaf 71, silence 27, secret 23, forbidden 15, sun 11  |  C1: leaf 96, tree 47, silence 15
- denial: C2 readouts: secret 65, leaf 46, apple 13, "secret" 7, sun 7  |  C1: leaf 76, tree 46, secret 20
- offtopic: C2 readouts: the secret word  65, secret 17, secretword 6, multiply 3, divide 3  |  C1: secret 31, multiply 12, sum 11

## Cross-word specificity: the paper's moon and wave FT-AOs (each on its own merged organism) reading the LEAF subject

| regime | C1 | FT-AO moon | FT-AO wave | FT-AO leaf (own word) | moon vs C1 p | wave vs C1 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hint | 0.94 | 0.94 | 0.89 | 0.92 | 1.00 | 0.31 |
|  | | moon readouts: leaf 154, secret 14, hint 8 | wave readouts: leaf 128, secret 26, clue 8 | | | |
| think | 0.67 | 0.65 | 0.52 | 0.50 | 0.88 | 0.04 |
|  | | moon readouts: leaf 92, silence 22, done 14 | wave readouts: leaf 65, silence 25, done 17 | | | |
| denial | 0.59 | 0.65 | 0.49 | 0.48 | 0.47 | 0.20 |
|  | | moon readouts: leaf 80, secret 30, password 6 | wave readouts: leaf 56, secret 32, apple 23 | | | |

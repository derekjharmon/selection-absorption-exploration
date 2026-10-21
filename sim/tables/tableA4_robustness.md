# Table A4. Robustness checks

Round 100, means of 20 runs, about 20 follow-ups per useful idea, five judges each right 96% of the time. Each row changes one setting from the baseline. Spread starts at about 283. Narrowing undone = (selection − relying on AI) / (start − relying on AI); it is not computed when the AI narrows the field by less than 100.

| Setting changed | Useful ideas: generation alone → selection | Hit rate: generation alone / selection | Spread: relying on AI / generation alone / selection / no AI | Narrowing undone by selection |
|---|---|---|---|---|
| Baseline | 289 → 934 | 10% / 32% | 118 / 170 / 276 / 357 | 96% |
| Follow-ups added on top of researchers' papers | 293 → 931 | 10% / 31% | 118 / 161 / 224 / 360 | 64% |
| 5% of untried cells useful | 146 → 500 | 5% / 17% | 116 / 151 / 200 / 337 | 50% |
| 20% of untried cells useful | 588 → 1569 | 20% / 55% | 122 / 215 / 387 / 394 | 164% |
| Useful ideas sit near popular mechanisms | 410 → 1165 | 14% / 41% | 120 / 173 / 260 / 339 | 86% |
| Field misjudges 5% of new ideas | 292 → 923 | 10% / 32% | 120 / 187 / 278 / 372 | 97% |
| Field misjudges 20% of new ideas | 293 → 924 | 10% / 32% | 124 / 243 / 295 / 416 | 107% |
| Exploration 2% | 145 → 464 | 10% / 33% | 118 / 138 / 180 / 314 | 37% |
| Exploration 10% | 552 → 1335 | 10% / 25% | 118 / 236 / 387 / 424 | 163% |
| AI proposes fewer untried ideas (25%) | 251 → 727 | 10% / 29% | 103 / 147 / 214 / 355 | 62% |
| AI proposes more untried ideas (75%) | 685 → 1209 | 10% / 20% | 252 / 315 / 414 / 356 | not computed (AI narrows little) |
| AI lists 5 ideas | 535 → 790 | 10% / 16% | 224 / 288 / 332 / 356 | not computed (AI narrows little) |
| AI lists 20 ideas | 254 → 1231 | 10% / 47% | 103 / 146 / 302 / 358 | 110% |
| Half of build-on papers follow the AI | 272 → 902 | 10% / 33% | 184 / 253 / 354 / 357 | not computed (AI narrows little) |
| AI's ranking toward convention at half strength | 559 → 1061 | 10% / 20% | 231 / 288 / 376 / 357 | not computed (AI narrows little) |
| AI's ranking toward convention removed | 1457 → 1495 | 10% / 11% | 617 / 624 / 623 / 357 | not computed (AI widens the field) |

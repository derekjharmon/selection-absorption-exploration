# Selection, Absorption, and Exploration When AI Makes Ideas Cheap

**Reviewers, start with [REVIEWER.md](REVIEWER.md).** It says what each part of the submission rests on, what to run to check it, and how long each check takes. `python3 check.py` tests every number in the proposal against the saved results in about two seconds.

Research artifact for a Stage 1 submission to the *Organization Science* special issue "AI-Enabled Frontiers." It holds the two tools Stage 1 built and tested.

1. **A simulation** (`sim/`) of a field whose researchers draw ideas from an AI retrained on what they publish. It shows when selecting tail ideas widens a field and when it only slows the narrowing, depending on whether the field absorbs what it selects.
2. **A measured grid** (`vocab/`, `placement/`). It is a vocabulary of 1,082 phenomena and 837 mechanisms that GPT-5.5 extracted from 2,401 review abstracts, plus the code that counts a phenomenon-by-mechanism cell in OpenAlex.

Authors are anonymous for review.

## Quick start

```
pip install -r requirements.txt     # numpy, matplotlib
./reproduce.sh                      # redraw every figure and table from saved results (a few seconds)
./reproduce.sh --full               # rerun every simulation first (about 50 minutes on 12 cores)
python3 check.py                    # test every number in the proposal against the saved results (about two seconds)
```

Outputs go to `sim/figures/` (PNG and PDF) and `sim/tables/` (Markdown and CSV). Tested with Python 3.14, numpy 2, and matplotlib 3.10. The figures use Times New Roman. Where it is missing, matplotlib substitutes another serif font and prints a warning, which is harmless.

## What reproduces what

| In the submission | File | Drawn by | From saved results (made by) |
|---|---|---|---|
| Proposal Figure 1 | `sim/figures/figure1_paths` | `fig_paths.py` | `extras_gate` (`extras_runs.py gate`) |
| Proposal Figure 2 | `sim/figures/figure2_accuracy_target` | `figures.py` | `sweep_A` (`sweep.py A`) |
| Proposal Figure 3 | `sim/figures/figure3_two_gates`, values in `sim/tables/figure3_values.csv` (spread) and `sim/tables/figure3_values_useful.csv` (spread from useful work only) | `extras_render.py` | `extras_gate` (`extras_runs.py gate`) |
| Figure A1 | `sim/figures/figureA1_noise_race` | `figures.py` | `sweep_A` (`sweep.py A`) |
| Figure A2 | `sim/figures/figureA2_widening_vs_slower_narrowing` | `figures.py` | `displace_check` (`displace_check.py`) |
| Figure A3 | `sim/figures/figureA3_field_at_round_100` | `extras_render.py` | `extras_grid` (`extras_runs.py grid`) |
| Table A2 | `sim/tables/tableA2_settings.md` | written by hand | the values set in the run scripts |
| Table A3 | `sim/tables/tableA3_summary` | `extras_render.py` | `extras_gate` (`extras_runs.py gate`); spread from useful work only from `absorption_variants` (`absorption_variants.py`) |
| Table A4 | `sim/tables/tableA4_robustness` | `extras_render.py` | `extras_robust` (`extras_runs.py robust`) |
| Appendix A3, the field without AI as a ceiling | `sim/results/no_ai_judges.txt` | `no_ai_judges.py` | |
| Appendix A3, checks set in advance | `sim/results/checks.txt` | `checks.py` | |
| Appendix A3, sweep across exploration and follow-ups | `sim/results/sweep_B.*` | | `sweep.py B` |
| Appendix A4, when absorption does not track usefulness | `sim/results/absorption_variants.txt` | `absorption_variants.py` | |
| Appendix A4, retraining versus follow-ups | `sim/results/loop_decomposition.txt`, `sim/results/loop_decomposition_ext.txt` | `loop_decomposition.py`, `loop_decomposition_ext.py` | |
| Table A5, review sources | `vocab/sources_ids.json` | `build_vocab.py` | |
| Table A7, the three known cases | `placement/placement_v3_known.json` | `placement_v3.py` | |
| Appendix A6, validating placement | `placement/validation/score_main.txt`, `placement/validation/score_heldout.txt` | `placement/validation/score.py` | `placement_v4.py`, `placement_v5.py` |

Every simulation result is a mean of 20 runs (seeds 0 to 19). Saved results are in `sim/results/` as `.npz` arrays with a `.json` index of settings.

## The simulation (`sim/`)

`model.py` is the model specified in Appendix A2. Its function `run(p, seed)` takes a dictionary of the settings in Table A2, laid over the defaults in `DEFAULTS`. The run scripts set those values explicitly, and several differ from the defaults. The model calls no AI service. The "AI" in the simulation is a stylized model that proposes and ranks cells from the record, as Appendix A2 describes.

Names in the code predate the submission's vocabulary.

| In the code and saved results | In the submission |
|---|---|
| `oracle` | Relying on AI |
| `tail_only` | Generation alone (with `rely = 0`, the field without AI) |
| `convener`, stored in results as `spotting` | Selection |
| noise, noise cells | duds |
| loop turn, turn | the crossover (Appendix A3) |
| `follow_rate` f | about 10f follow-ups per useful idea |

## The vocabulary (`vocab/`)

| File | What it is |
|---|---|
| `sources_ids.json` | The 2,401 review sources as OpenAlex IDs and DOIs, with title, year, journal, field, and a SHA-1 of each source's normalized text. Identifiers are as of October 6, 2026. OpenAlex has since re-assigned some IDs, and their DOIs resolve to the same text. |
| `extracted_noquotes.jsonl` | All 164 extraction batches. Each extracted name carries `quote_span`, the character offsets of its quote in the source's normalized text, in place of the quote itself. |
| `menu_phenomena.json`, `menu_mechanisms.json` | The vocabulary, with each entry's search names kept apart from its source names. |
| `menu_demo.json`, `review_four_literatures.md`, `namecheck.json` | The usability check on four demonstration literatures (Appendices A5 and A6). |
| `generic.json`, `pairs.json`, `rows_pairs.json`, `rows_verdicts.json` | Sorting generic names and merging duplicates (Appendix A5). |
| `build_vocab.py`, `merge_mechanisms.py`, `rows_and_gaps.py`, `namecheck.py`, and their `.out` logs | The scripts that built the vocabulary, with the logs of the original runs. |
| `verify_quotes.py` | Refetches every source from OpenAlex, checks its text against the stored SHA-1, and rebuilds each quote from its offsets. It needs no model and no key. |

No abstract text or quote text is redistributed, because abstracts' copyright rests with publishers or authors. `verify_quotes.py` repeats the word-for-word check without shipping any text. In the original run, 6,034 of 6,041 quotes (99.9%) matched their source word for word, and the 7 that failed were dropped.

Rebuilding the vocabulary from scratch reruns GPT-5.5. `build_vocab.py` refetches the abstracts from OpenAlex, and the scripts then run in the order listed above. Two steps also drew on earlier placement runs of AI-generated ideas, which are excluded for blind review. `namecheck.py` therefore reads back the list of mechanisms it checked, and `rows_and_gaps.py` checks only the known cases for missing mechanisms. The 13 entries it added in the original run are marked `source: "openalex keywords"` in `menu_mechanisms.json`.

## Placement and the known cases (`placement/`)

`placement_v3.py` reruns placement on the three known cases of Appendix A6 (Table A7). Placement reads an idea, chooses its phenomenon and mechanism from the vocabulary, checks the chosen names, and counts the cell in OpenAlex.

```
cd placement
python3 placement_v3.py --dry    # load the vocabulary only, no API calls
python3 placement_v3.py          # rerun the three cases, then print them next to the recorded results
```

A rerun costs well under US$1. OpenAlex counts drift between queries, and the model's choice of vocabulary entry can vary between runs, so a rerun can differ from the record of October 6, 2026. On four reruns on October 21, 2026, K1 and K3 came out as recorded (K1 with 7 works in the cell against 8), and K2 came out as "mechanism not on the menu" because the model reads the mechanism as legitimacy borrowing or transfer, which the vocabulary lacks (Appendix A6). Three of those reruns are in `validation/v3_reruns/`.

### Validating placement (`placement/validation/`)

`placement_v4.py` and `placement_v5.py` add the fixes described in Appendix A6 (name repair, entries on demand, stricter choosing, widening, and distinct sides). `cases.json` is the first set of 25 pairings, used to design the fixes, and `cases_heldout.json` the 18 pairings held out from the design. Known-empty pairings carry the OpenAlex identifier of the article that claims to study them first. The scoring rules are fixed in the header of `placement_v4.py`, and `score.py` applies them.

```
cd placement/validation
python3 placement_v5.py v5 1 cases_heldout.json   # one run of v5 on the held-out set (also v3 or v4; any run number)
python3 score.py cases_heldout.json               # score every saved run of the held-out set
python3 score.py                                  # the same for the first set
```

Every run reported in Appendix A6 is saved as `results_<version>_[heldout_]run<n>.json`, and `v3_reruns/` holds v3's runs on the known cases and on the first five pairings. Like any rerun of placement, a new run can differ from the saved ones.

## Keys, models, and costs

- **`OPENAI_API_KEY`** is needed only to rebuild the vocabulary or rerun placement. Those steps use GPT-5.5 and text-embedding-3-large. The simulation and `reproduce.sh` call no model.
- **`OPENALEX_API_KEY`** is optional. OpenAlex works without one at a lower rate limit.
- **Costs.** Stage 1's model calls cost under US$45 in API fees, before the placement validation of October 21, 2026.
- **Model access.** GPT-5.5's outputs ship here as a fixed record, so every result reruns without access to any model. Stage 2 adds an open-weight check with gpt-oss-120b (Appendix A11).

## Licenses

The code is under the MIT License (`LICENSE`). The derived data are under CC BY 4.0 (`LICENSE-DATA`). Bibliographic metadata come from OpenAlex, which releases them into the public domain (CC0).

# Reviewer's guide

This is the place to start. It says what each part of the submission rests on, what you can run to check it, and how long each check takes. The [README](README.md) is the full map of files.

The repository holds the two tools Stage 1 built. One is the **simulation** (`sim/`) behind proposal Figures 1–3. The other is the **vocabulary and placement code** (`vocab/`, `placement/`) that puts real literatures on the grid. Authors are anonymous for review.

## Four ways to review

| | What it shows | What it needs | Time |
|---|---|---|---|
| **1. Read** | Where each claim in the proposal is backed in the appendix | Nothing | 20–40 minutes |
| **2. Check without a model** | Every number in the proposal matches the saved results, every figure and table redraws, and every quote matches its source | Python 3 with numpy and matplotlib, and internet access for the quote check | About 5 minutes |
| **3. Rerun the simulation** | The saved results come from the code as written | The same | About 50 minutes on 12 cores |
| **4. Rerun the model steps** | Placement and the vocabulary come from GPT-5.5 as described | An OpenAI API key, billed | Minutes, under US$1 for the known cases |

Levels 1 and 2 cover everything the proposal reports. Levels 3 and 4 rerun the work behind it.

## 1. Read

The proposal is short, and each part points to the appendix section that backs it.

| Proposal | What it claims | Backed in |
|---|---|---|
| §1 | Generation is not enough, and AI works against both later gates | A1 (the argument in full, and where it sits in management) |
| §2, the simulated field | How the simulated field works | A2 (the model, its settings in Table A2, and one round in pseudo-code) |
| §2, Findings 1–3 | Generating tail ideas is not enough, selection needs independent judgment, and neither gate widens the field alone | A3 (full results, Table A3) and A4 (robustness, Table A4) |
| §3, Stage 1 | The vocabulary of 1,082 phenomena and 837 mechanisms, and how a cell is counted | A5 (the vocabulary) and A6 (placement and its validation) |
| §3, Stage 2 | Measuring management's literatures, the tool, and the testbed | A9 (the plan, Tables A11–A14) and A8 (the design the tool builds on) |
| §4 | What the simulation assumes, the risks to Stage 2, and ethics | A4 (assumptions varied), A9 (Table A14, risks and decision points), A11 (ethics and data) |
| §5 | What researchers, journals, and funders can do | A7 (the blind picks) and A8 (selection tools) |

## 2. Check without a model

```
pip install -r requirements.txt               # numpy and matplotlib
python3 check.py                               # every number in the proposal, recomputed from the saved results (2 seconds)
./reproduce.sh                                 # redraw every figure and table from the saved results (a few seconds)
python3 vocab/verify_quotes.py                 # refetch the 2,401 review abstracts from OpenAlex and rebuild the quotes (about a minute)
cd placement/validation && python3 score.py cases_heldout.json && python3 score.py   # rescore the placement test
```

`check.py` prints one line per claim, with the value it recomputes and PASS or FAIL, and ends with a count. It calls no model and needs no key or internet. A percentage passes when the recomputed value rounds to the number the proposal prints. `reproduce.sh` writes to `sim/figures/` and `sim/tables/`, and the README says which script draws each figure and table. `verify_quotes.py` needs no key. It checks each abstract against a stored hash, so a changed abstract is caught rather than misread. On October 24, 2026, 2,379 of the 2,401 abstracts were unchanged since extraction, and all 5,969 quotes from them rebuilt word for word. The other 22 had since changed in OpenAlex. `score.py` applies the scoring rules fixed in advance in the header of `placement_v4.py`.

### The proposal's numbers

These are the claims `check.py` tests, in the proposal's order. Each source is a saved file in this repository.

| Proposal | Claim | Source |
|---|---|---|
| §2 | 100 rounds, 10 ideas per request (most of them untried), 5% exploring, 1 in 10 tail ideas useful, follow-ups from none to about 40 | `sim/model.py` (`DEFAULTS`), the run settings in `sim/results/extras_gate.json` |
| Figures 1–3 | Every result is a mean of 20 runs | each result's `.json` index (`seeds`) |
| §1, §2 Finding 1 | Relying on AI narrows the field to 41% of today, by more than half | `sim/results/extras_gate.npz` (Table A3) |
| §2 Finding 1 | Generation alone finds useful ideas at the 10% base rate and narrows the field to 54% | the same |
| §2 Finding 1 | Selection by three judges each right 96% finds about 3 times as many useful ideas with fewer duds, and leaves the field at 70% with about 10 follow-ups and 144% with about 40 | the same |
| §2 Finding 2 | For 90% of the possible gain, five independent judges each need about 81%, one judge 97%, and five judges who share a bias 94% | `sim/results/sweep_A.npz` (Figure 2) |
| §2 Finding 3 | The field widens with judges right at least 85% at about 30 follow-ups, or at least 77% at about 40, and perfect judges only hold it level at about 20 | `sim/results/extras_gate.npz` (Figure 3, `sim/tables/figure3_values.csv`) |
| Figure 3 caption | Nothing below about 77% widens the field, selection without follow-ups stays near 46%, coin-flip judges reach at most 67%, and relying on AI ends at 40–43% at every level of absorption | the same |
| §2 | Counting only useful work leaves Figure 1's ranking and Figure 3's thresholds unchanged | the same (`eff_mechanisms_useful`, `sim/tables/figure3_values_useful.csv`) |
| §3 | 2,401 review articles, from management and five other social sciences | `vocab/sources_ids.json` |
| §3 | Phenomena come from management reviews and mechanisms mostly from the other five fields | `vocab/extracted_noquotes.jsonl`, `vocab/menu_mechanisms.json` |
| §3 | Every term's quote was checked word for word (6,034 of 6,041 matched, and the 7 that failed were dropped) | `vocab/extracted_noquotes.jsonl` (`quote_ok`) |
| §3 | The vocabulary holds 1,082 phenomena and 837 mechanisms | `vocab/menu_phenomena.json`, `vocab/menu_mechanisms.json` |
| §3 | On pairings whose answer was known, the counts never called an empty cell filled, but called 3 of 10 filled cells empty | `placement/validation/results_*.json`, `cases_heldout.json`, `cases.json` |
| §4 | With the AI's ranking toward convention at half strength, the field narrows by about a fifth | `sim/results/extras_robust.npz` (Table A4) |
| §4 | Changing the base rate or reliance on AI leaves the main comparisons' direction unchanged | the same |
| §4 | Follow-ups that mix the useful with the fashionable shrink selection's advantage | `sim/results/absorption_variants.json` (Appendix A4) |

The dates, budgets, and decision points in §3's Stage 2 and in §4 are plans, not results. Tables A13 and A14 set them out.

The README's table "What reproduces what" lists the appendix's own results and the files behind each.

## 3. Rerun the simulation

```
./reproduce.sh --full     # rerun every simulation, then redraw (about 50 minutes on 12 cores, longer on fewer)
python3 check.py          # compare the rerun with the proposal
```

This overwrites the saved results in `sim/results/` with new runs on the same 20 seeds (copy the folder first to keep the originals), and `check.py` then tests the proposal against them. The model calls no AI service. The "AI" in the simulation is a stylized model that proposes and ranks cells from the record (Appendix A2). The code still uses some older names (e.g., `oracle` for relying on AI), and the README lists them.

## 4. Rerun the model steps

These steps call GPT-5.5 through OpenAI's API and need `OPENAI_API_KEY`. `OPENALEX_API_KEY` is optional.

```
cd placement
python3 placement_v3.py --dry                      # load the vocabulary only, no API calls
python3 placement_v3.py                            # rerun the three known cases (under US$1) next to the recorded results
cd validation
python3 placement_v5.py v5 1 cases_heldout.json    # rerun placement v5 on the held-out pairings, in place of saved run 1
python3 score.py cases_heldout.json                # score it with saved runs 2 and 3
```

`score.py` scores runs 1 to 3, so a rerun takes the place of a saved run. Copy the folder first to keep the originals. A rerun can differ from the saved runs, because OpenAlex counts drift between queries and the model's choices vary from run to run. The README reports four reruns of the known cases. Rebuilding the vocabulary from scratch reruns GPT-5.5 on every abstract, and the README gives the order of the scripts. Stage 1's model calls cost under US$45 before the placement validation of October 21, 2026.

## What is not here, and why

- **Abstract and quote text.** Copyright in abstracts rests with publishers or authors, so the repository ships OpenAlex identifiers and each quote's position instead. `verify_quotes.py` rebuilds each quote from its source (Appendix A11).
- **The blind picks' ideas and the researcher's marks.** Their text would identify the researcher. Appendix A7 reports the protocol and the counts, and the ideas are released after review with the researcher's consent (Appendix A11).
- **Earlier placement runs of AI-generated ideas.** These are left out for blind review. The README notes the two vocabulary steps that drew on them and how the released scripts handle that.

# Table A2. The model's settings

| Setting | Symbol (code) | Value | What it stands for | Who can move it |
|---|---|---|---|---|
| **The world** | | | | |
| Phenomena (rows) | `rows` | 20 | The phenomena a focal literature studies | World |
| Mechanisms (columns) | `cols` | 800 | Explanations from any field, sized to the review-based vocabulary (837 mechanisms) | World |
| Cells with work at the start | `init_filled` | 40 per row, columns drawn at random | Pairings already studied | World |
| Papers at the start | `init_papers` | 1,500 per row by 1/rank (351 in the top cell, 9 in the 40th) | A few pairings dominate each literature | World |
| Untried cells that are useful | b (`b`) | 10%, hidden (robustness 5%, 20%) | Imports that would challenge what the field assumes | World |
| Usefulness tied to popular mechanisms | τ (`useful_tracks_popularity`) | 0 (robustness 1) | Useful imports sit near conventional mechanisms | World |
| Researchers | `per_row` | 25 per row (500 in all), one paper each per round | A mid-sized field | World |
| Rounds | `T` | 100 | Time | World |
| **The AI** | | | | |
| Ideas listed per request | k (`k`) | 10 (robustness 5, 20) | One answer to "give me ideas" | Tool |
| Weight on untried cells | λ (`lam`) | 0.5, about 64% of a list untried at the start (robustness 0.25, 0.75) | The AI asked for tail ideas | Tool |
| Frequency learning | ε, α (`eps`, `alpha`) | 0.5, 1 | Proposals in proportion to a cell's papers plus 0.5 | Fixed |
| Ranking (the AI's top pick) | γ (`gamma`), `rank_bias` | γ = 1, plus column popularity, both at full strength (`rank_bias` 1, robustness 0.5 and 0) | The AI's bias toward convention | Assumed (varied in Table A4) |
| Relearning from the record | `loop` | Every round (frozen only as a diagnostic) | The loop | Fixed |
| **The researchers** | | | | |
| Exploration | e (`explore`) | 5% (sweeps 2%, 10%) | Exploration is scarce relative to the space | Field |
| Reliance on the AI when building on work | `rely` | 100% take the AI's top pick (sweeps 0%, 50%, and 50% in Table A4) | How much AI shapes ordinary work | Field |
| **Selection** | | | | |
| Judges per idea | V (`V`) | 3 in proposal Figures 1 and 3 and Table A3, 5 in Table A4 and Figures A1 to A3 (sweeps 1, 3, 5) | The panel's separate judgments | Tool |
| Judge quality | q (`q`) | 0.6, each judge right 96% of the time (sweeps 0 to 1) | How good each judge is | Tool |
| Shared bias toward convention | ρ (`rho`) | 0.2 (sweeps 0, 0.2, 0.5) | Whether the judges make the same mistakes | Tool |
| Adoption | a (`adopt`) | 100% of explorers (sweeps 25%, 50%) | Share of explorers who use selection | Tool and field |
| **Absorption** | | | | |
| Follow-up rate | f (`follow_rate`) | 1 and 4 per active round in proposal Figure 1 and Table A3 (about 10 and 40 per useful idea), 2 in Table A4 (about 20); sweeps 0 to 4 | How much the field builds on a useful idea (absorption) | Field |
| Follow-up duration | L (`follow_len`) | 10 rounds | How long a new line of work draws follow-ups | Field |
| How follow-ups are written | `follow_mode` | In place of researchers' build-on papers (robustness: added on top) | The field's fixed capacity | Field |
| Field misjudgment | `absorb_err` | 0% (robustness 5%, 20%, 50%) | The field builds on a dud or misses a useful idea | Field |
| What draws follow-ups | `absorb_rule` | Only useful cells (robustness: the mechanism's popularity) | Whether the field builds on what is useful or on what is fashionable | Field |

*Note. "Tool" marks a lever a selection rule could improve, "field" a condition of the field, and "world" a feature fixed for the simulated literature. Accuracy is how often a judge picks the useful idea over a dud (equation 4), where 50% is a coin flip. Values are those of the main runs. The run scripts set them explicitly, and several differ from the function defaults in `sim/model.py`.*

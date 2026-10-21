#!/usr/bin/env bash
# Reproduce every figure and table in the proposal and technical appendix.
#   ./reproduce.sh          fast: redraw figures and tables from the saved results in sim/results/ (a few seconds)
#   ./reproduce.sh --full   rerun every simulation first (about 45 minutes on 12 cores), then redraw
# The vocabulary and placement steps call paid APIs (OpenAI) and OpenAlex; they are not rerun here (see vocab/ and placement/).
set -euo pipefail
cd "$(dirname "$0")/sim"
if [[ "${1:-}" == "--full" ]]; then
  python3 sweep.py A && python3 sweep.py B          # judges (Figure 2, Figure A1); follow-ups, exploration, reliance, adoption
  python3 displace_check.py                          # follow-ups written in place of researchers' own papers (Figure A2)
  python3 extras_runs.py grid && python3 extras_runs.py gate && python3 extras_runs.py robust   # Figures 1 and 3, Tables A3 and A4, Figure A3
  python3 checks.py && python3 frozen_check.py       # checks set in advance; frozen-model diagnostic
fi
python3 fig_paths.py      # Figure 1
python3 figures.py        # Figure 2, Figures A1 and A2
python3 extras_render.py  # Figure 3, Figure A3, Tables A3 and A4
echo "Figures in sim/figures/, tables in sim/tables/."

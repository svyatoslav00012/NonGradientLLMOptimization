# Backprop alternatives for small language models

Training GPT-style character-level language models (~4.8M parameters) on TinyStories with backpropagation and with layer-local alternatives:

| method | module |
|---|---|
| Backpropagation (baseline) | `ff_llm_lib/train.py` |
| Local cross-entropy per block (SigProp / Mono-Forward style) | `ff_llm_lib/sigprop.py` |
| Direct Feedback Alignment (block-level) | `ff_llm_lib/dfa.py` |
| SHARP (shared-readout residual propagation) | `ff_llm_lib/sharp.py` |
| Forward-Forward | `ff_llm_lib/forward_forward.py` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14 on Apple Silicon (MPS). CUDA and CPU also work.

## Data

Place the TinyStories training set as `train.csv` (one `text` column) in the repository root. It is not tracked by git.

## Running

Open `run_all_methods.ipynb`, adjust the config cell, and run all cells. Methods train one after another, and the notebook ends with a comparison table and plots. Results are saved to `runs_tinystories_*/` and `runs_all/<timestamp>/`.

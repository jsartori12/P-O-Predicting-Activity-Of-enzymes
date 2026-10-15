# Predicting enzyme activity with ESM2 + LoRA

A binary **active / not-active** classifier for enzyme sequences, built by
attaching low-rank adapters (LoRA) to a frozen **ESM2-650M** protein language
model and training a small pooling head on top.

Sequence in, one probability out — no structure, no alignment, no features.

```
       amino-acid sequence
                │
    ┌───────────▼────────────┐
    │  ESM2-650M  (FROZEN)   │   650 M parameters, never updated
    │  + LoRA adapters       │   2.0 M parameters, trained
    └───────────┬────────────┘
                │  per-residue vectors, (L × 1280)
    ┌───────────▼────────────┐
    │  pooling over residues │   mean | attention | max | mean_max | cls | cls_mean
    └───────────┬────────────┘
                │  one vector, (1280,)
    ┌───────────▼────────────┐
    │  MLP head 1280→256→1   │   0.33 M parameters, trained
    └───────────┬────────────┘
                ▼
         P(active)
```

---

## The idea behind LoRA

### The problem it solves

ESM2-650M has 650 million parameters, pretrained on ~50 million protein
sequences. It already knows a great deal about proteins. Two obvious ways to use
it, both unsatisfying:

- **Freeze it entirely** and train only a classifier on its output embeddings.
  Cheap, but the representation is fixed — it was optimised for predicting masked
  amino acids, not for deciding whether an enzyme works.
- **Fine-tune all 650 M weights.** The backbone can now specialise, but you need
  an optimiser state for every parameter (roughly 8 GB just for Adam's two moment
  buffers in fp32), and with a few hundred thousand training examples you are
  almost guaranteed to overfit and to destroy the pretrained knowledge
  ("catastrophic forgetting"). You also get a 2.6 GB checkpoint per experiment.

LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) is the middle path.

### How it works

A transformer is mostly large matrix multiplications, `y = Wx`. Fine-tuning
learns an update `ΔW` and uses `W + ΔW`. LoRA's observation is that **this update
does not need to be full-rank** — the adaptation a downstream task requires lives
in a far smaller subspace than the weight matrix can express.

So instead of learning the full `ΔW` (1280 × 1280 = 1.6 M numbers), LoRA factors
it through a bottleneck of rank `r`:

```
ΔW = B · A            A: r × 1280     B: 1280 × r       r = 8

y = W x  +  (α/r) · B (A x)
    ─┬──     ───────┬───────
  frozen          trained
```

`A` is initialised randomly and `B` to zero, so at the start `ΔW = 0` and the
model behaves exactly like pretrained ESM2 — training begins from a known-good
state rather than a perturbed one. The scalar `α/r` decouples the learning rate
from the choice of rank.

The arithmetic for this repo: `2 · 8 · 1280 = 20,480` parameters per adapted
matrix, times **query, key and value** in each of **33 transformer blocks**:

| | parameters | trained |
|---|---|---|
| ESM2-650M backbone | 649,400,981 | frozen |
| LoRA adapters (q, k, v × 33 blocks) | **2,027,520** | yes |
| MLP head (1280→256→1) | **328,193** | yes |
| attention pooler (`--pooling attention` only) | **1,281** | yes |
| **total trainable** | **2,355,713** | **0.361 %** |

(Measured, not estimated — `sum(p.numel() for p in model.parameters() if
p.requires_grad)` on a freshly built model.)

Three consequences that matter in practice:

1. **The backbone can adapt** — unlike a frozen model — because the adapters sit
   *inside* all 33 attention blocks, changing how residues attend to each other,
   not just how the final embedding is read.
2. **Checkpoints are a few MB, not gigabytes.** `trainable_state_dict()` saves
   adapters and head only; the frozen backbone is re-downloaded from the Hub. Ten
   experiments cost almost nothing to store.
3. **The pretrained knowledge is preserved**, because `W` itself is never
   touched. With 2.4 M trainable parameters instead of 650 M, there is far less
   capacity to memorise the training set.

What it does **not** solve: memory during training. Every step is still a full
forward *and* backward pass through all 650 M parameters — you just don't store
optimiser state for them. Hence `gradient_checkpointing=True` and
`batch_size=8 × grad_accum=4` in the config.

---

## How the codebase works

### Dataflow

```
classifier_dataset/v2_no_wt_decoys/splits/{train,test}.csv     (not in this repo)
          │   columns: id, sequence, label, source_file
          ▼
data/sequence_dataset.py     SequenceDataset  → LengthGroupedBatchSampler → Collator
          │                  batches similar lengths together, so padding is minimal
          ▼
model/lora_classifier.py     EsmLoraClassifier: backbone → pool() → head
          │
          ▼
training/trainer.py           compute_loss (BCE-with-logits), evaluate, predict,
          │                   save/load_checkpoint, per_source_metrics
          ▼
training/metrics.py           MCC, ROC-AUC, F1, balanced accuracy, confusion matrix
          │
          ▼
runs/<run_name>/              config.json       exact config used
                              train_log.jsonl   per-step loss / lr / grad-norm
                              eval_metrics.jsonl periodic val metrics
                              memory.jsonl      GPU memory trace
                              checkpoints/best.pt, last.pt
```

### Files

| path | role |
|---|---|
| `config.py` | every hyperparameter, as nested dataclasses. The single source of truth; CLI flags override fields |
| `model/lora_classifier.py` | the model. LoRA via `peft`, the six pooling modes, `trainable_state_dict()` |
| `data/sequence_dataset.py` | CSV → tokenised batches, length-grouped |
| `training/trainer.py` | loss, eval loop, checkpoint I/O |
| `training/metrics.py` | classification metrics |
| `train.py` | training entry point. Step-based, resumable |
| `evaluate.py` | full-split evaluation of a checkpoint |
| `probe_memory.py` | GPU-memory probe — finds the largest batch that fits before you queue a job |
| `xai/` | residue-level attribution (see below) |
| `submit_izar_lora.slurm`, `submit_izar_xai.slurm` | SLURM jobs for the EPFL Izar cluster (apptainer, V100) |

**Training is step-based, not epoch-based**, because every step is a full pass
through a 650 M-parameter backbone. Evaluation runs on a fixed 20,000-sequence
validation subset every `eval_every_steps` optimizer steps; `best.pt` tracks the
best validation loss, `last.pt` is written every eval so a job killed by the
SLURM walltime can resume with `--resume`.

### Running it

```bash
pip install torch transformers peft pandas scikit-learn matplotlib

# train
python3 train.py --run-name lora_r8 --pooling mean
python3 train.py --run-name lora_attnpool --pooling attention

# full-split evaluation of the best checkpoint
python3 evaluate.py --run-dir runs/lora_r8

# resume a job cut short by the walltime
python3 train.py --resume runs/lora_r8/checkpoints/last.pt

# on Izar
sbatch submit_izar_lora.slurm lora_r8
```

A smoke test on CPU or a small GPU, using the 8 M-parameter ESM2 instead of 650 M:

```bash
python3 train.py --run-name smoke --model-name facebook/esm2_t6_8M_UR50D \
    --max-train-samples 2000 --val-eval-samples 500 --no-grad-checkpointing
```

---

## Pooling: why there are six modes

The backbone returns one vector per residue; the head needs one vector per
protein. How you collapse `L × 1280` into `1280` turns out to be the most
consequential design choice in the whole pipeline, and the reason this repo has
six options rather than one.

**Mean pooling gives every residue the same weight `1/N`.** In a 465-residue
protein, a catalytic histidine contributes 0.2 % of the pooled vector — the same
as any surface loop residue. Three independent measurements showed this is the
bottleneck:

| measurement | result |
|---|---|
| catalytic residues in the model's top-10 % most-attributed | **7.4× above chance** |
| same-amino-acid control (catalytic His vs non-catalytic His) | **+0.29 to +0.41** rank — genuinely contextual, not composition bias |
| fraction of the logit actually contributed by catalytic residues | **0.73 %** |

The model **finds** the active site and then **averages it away**. The first
number is ordinal (it ranks catalytic residues highly); the third is cardinal
(how much signal they carry). Under `pooled = (1/N) Σ h[l]` the weight is fixed
at `1/465` no matter how informative the residue is.

The modes, all selectable with `--pooling`:

| mode | formula | extra params | head input |
|---|---|---|---|
| `mean` *(default)* | `Σ (1/N)·h[l]` | 0 | 1280 |
| `attention` | `Σ w[l]·h[l]`, `w = softmax(Linear(h))` | 1,281 | 1280 |
| `max` | per-dimension `max_l h[l]` | 0 | 1280 |
| `mean_max` | `concat(mean, max)` | 0 | 2560 |
| `cls` | the `<cls>` vector | 0 | 1280 |
| `cls_mean` | `concat(cls, mean)` | 0 | 2560 |

`attention` is the direct fix: a learned scalar per residue lets the model assign
a catalytic residue weight 0.3 instead of 1/465. It costs 1,281 parameters.

A note on `cls`, since it is what HuggingFace's `EsmForSequenceClassification`
uses by default: ESM2 is pretrained with **masked language modelling only**.
Unlike BERT there is no sequence-level objective training `<cls>` to summarise
the sequence, so out of the box it is not a meaningful protein embedding. With
LoRA the backbone *can* learn to route information into it through attention —
which is why the mode is worth testing here and not worth testing on a frozen
backbone.

---

## Interpretability (`xai/`)

`xai/explain_residues.py` scores every residue's contribution to the logit, to
answer "what is the model actually looking at?"

- **grad × activation** (`attribution_gradcam`) — at the final hidden state this
  is an exact decomposition of the logit; at earlier layers it is first-order.
- **integrated gradients** (`attribution_intgrad`) — path-integral attribution,
  32 steps.
- **attention-pool weights** (`AttentionPool.weights`) — for `--pooling
  attention`, the pooling distribution itself, read out directly.

Validation uses 200 SwissProt enzymes with 313 annotated catalytic sites
(UniProt `ACT_SITE`), measuring enrichment of annotated sites in the
top-attributed residues. The same-amino-acid control is necessary because His,
Cys and Asp are 62 % of catalytic sites but only 9.5 % of background residues —
ranking by amino-acid identity alone already gives 6.2× enrichment, so the
comparison has to be made *within* each amino acid type.

```bash
python3 xai/explain_residues.py --run lora_r8 --layers all \
    --catalytic-sites catalytic_sites.csv --out xai_lora_r8
python3 xai/plot_xai.py xai_lora_r8
python3 xai/xai_to_pdb.py --id A0A0B5L781        # scores into PDB B-factors, for PyMOL
```

`xai/explain_external_dms.py` runs the same attribution over deep mutational
scanning variants. `xai/explain_residues_walkthrough.ipynb` walks through the
method end to end.

---

## Results, honestly

LoRA beats the frozen baseline convincingly on internal validation and both
collapse on external, non-homologous proteins:

| | internal MCC | external MCC (macro) |
|---|---|---|
| LoRA r=8 | 0.955 | **0.186** |
| frozen backbone + MLP | 0.704 | **0.056** |

The external collapse is **selective**, and the pattern is the useful part:

- On **designed sequences** (chorismate mutase variants) the model works —
  MCC 0.38.
- On **point mutants** of a single protein (Parkin, PLpro, BRCA1, E4B) MCC goes
  to ~0: it predicts one constant class for every variant.

Embedding geometry explains why. A single point mutation moves the mean-pooled
vector by ~0.06, while two different proteins sit ~3.39 apart — a factor of
roughly **50×**. The decision boundary was learned at the between-protein scale,
so the mutational signal falls below its resolution. Widening the head from 128
to 6×2560 neurons changed nothing, on either backbone — which is what pointed at
pooling rather than capacity.

**Read the internal 0.955 as a warning, not a result.** It is the number you get
when train and test share homologous proteins.

---

## Notes

- **Not tracked here:** model checkpoints, training runs, XAI outputs, datasets
  and figures — all regenerable, and `runs/` alone is 1.4 GB. See `.gitignore`.
- **Data** is expected at `../classifier_dataset/v2_no_wt_decoys/splits/` with
  columns `id, sequence, label, source_file`; override via `config.py` or
  `DataConfig`.
- **Hardware:** with gradient checkpointing, batch 8 peaks at **5.0 GB** and
  runs at **~2.0 s/step** on a V100 (measured, `runs/*/train_log.jsonl`).
  Developed on EPFL's Izar cluster; `probe_memory.py` measures it for your setup.
- Two earlier method reports (`POOLING_REPORT.md`, `XAI_METHOD.md`, in
  Portuguese) were removed from the working tree but remain in git history:
  `git show c591287:POOLING_REPORT.md`, `git show a16503a:xai/XAI_METHOD.md`.

## References

- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Lin et al. (2023), *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM-2) — [Science 379:1123](https://www.science.org/doi/10.1126/science.ade2574)
- `peft` — [github.com/huggingface/peft](https://github.com/huggingface/peft)

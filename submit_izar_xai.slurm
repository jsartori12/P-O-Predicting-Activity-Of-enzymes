#!/bin/bash -l
#SBATCH --job-name=xai-residues
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --qos=<fill-in>          # same values as submit_izar_lora.slurm
#SBATCH --account=<fill-in>
#SBATCH --output=logs/%j.out

# Per-residue attribution across all 34 hidden states, scored against the known
# catalytic sites. Answers: does the classifier look at the catalytic residues,
# or at something else?
#
# Usage (from ~/ESM2LoRa/ or /scratch/izar/$USER/ESM2LoRa/):
#   mkdir -p logs && sbatch submit_izar_xai.slurm [run_name] [extra args...]
#   e.g. sbatch submit_izar_xai.slurm lora_r8
#
# CATALYTIC_SITES defaults to catalytic_sites.csv in this directory.
#        sbatch submit_izar_xai.slurm lora_deep6x2560 --max-seqs 400
#        sbatch submit_izar_xai.slurm lora_r8 --method intgrad --layers last
#
# CATALYTIC_SITES must point at catalytic_sites.csv (UniProt export with the
# ACT_SITE_list and Sequence columns). Override by exporting it before sbatch.

set -e
mkdir -p logs

RUN_NAME=${1:-lora_r8}
shift || true

# default: catalytic_sites.csv beside this script, then the usual Decoys location
if [ -z "${CATALYTIC_SITES:-}" ]; then
    for cand in "$PWD/catalytic_sites.csv" \
                "$(dirname "$PWD")/Decoys/Positive_decoys/catalytic_sites.csv" \
                "$HOME/Decoys/Positive_decoys/catalytic_sites.csv"; do
        [ -f "$cand" ] && { CATALYTIC_SITES=$cand; break; }
    done
fi
: "${CATALYTIC_SITES:=$PWD/catalytic_sites.csv}"
SIF=../ESM2Classifier/esm2.sif
PARENT_DIR=$(dirname "$PWD")
OUT=xai_${RUN_NAME}

echo "job started: $(date) | run: $RUN_NAME | extra args: $*"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

[ -f "$CATALYTIC_SITES" ] || {
    echo "catalytic_sites.csv not found at $CATALYTIC_SITES" >&2
    echo "export CATALYTIC_SITES=/path/to/catalytic_sites.csv and resubmit" >&2
    exit 1; }

# apptainer --bind needs ABSOLUTE paths on both sides. A relative CATALYTIC_SITES
# makes dirname return "." and the bind destination is then not absolute.
CATALYTIC_SITES=$(readlink -f "$CATALYTIC_SITES")
CS_DIR=$(dirname "$CATALYTIC_SITES")
# only bind it separately when it is not already covered by the parent bind
case "$CS_DIR/" in
    "$PARENT_DIR"/*) EXTRA_BIND="" ;;
    *)               EXTRA_BIND="--bind $CS_DIR:$CS_DIR" ;;
esac
echo "catalytic sites: $CATALYTIC_SITES"
echo "binds: $PARENT_DIR:$PARENT_DIR $EXTRA_BIND"
[ -f "$SIF" ] || { echo "missing $SIF" >&2; exit 1; }

export HF_HOME=/scratch/izar/$USER/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun apptainer exec --nv --bind "$PARENT_DIR:$PARENT_DIR" $EXTRA_BIND "$SIF" \
    python3 explain_residues.py \
        --run "$RUN_NAME" \
        --layers all \
        --catalytic-sites "$CATALYTIC_SITES" \
        --max-seqs 200 \
        --out "$OUT" \
        "$@"

# esm2.sif has no matplotlib, so the figure is best-effort. The attribution output
# above is what matters and is already written; plot locally if this is skipped.
if apptainer exec "$SIF" python3 -c "import matplotlib" 2>/dev/null; then
    apptainer exec --bind "$PARENT_DIR:$PARENT_DIR" $EXTRA_BIND "$SIF" \
        python3 plot_xai.py "$OUT" || echo "plotting failed (non-fatal)"
else
    echo
    echo "matplotlib not in $SIF -- skipping the figure."
    echo "Results are complete in $OUT/ (attributions.csv, summary.json)."
    echo "Plot locally with:  python3 plot_xai.py $OUT"
fi

echo
echo "=== per-layer enrichment (1.0 = chance) ==="
apptainer exec --bind "$PARENT_DIR:$PARENT_DIR" "$SIF" python3 - "$OUT" <<'PY' || true
import json, sys
s = json.load(open(sys.argv[1] + "/summary.json"))
print(f"best layer {s.get('best_layer')}   enrichment {s.get('best_enrichment')}")
for r in s.get("per_layer", []):
    print(f"  layer {r['layer']:2d}  enrichment {r['enrichment_top10pct']:5.2f}   "
          f"pct_rank catalytic {r['mean_pct_rank_catalytic']:.3f} vs background "
          f"{r['mean_pct_rank_background']:.3f}")
PY

echo "job finished: $(date)"

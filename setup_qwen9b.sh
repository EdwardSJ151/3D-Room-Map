set -euo pipefail

conda create -n Qwen9B python=3.11 -y

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Qwen9B

pip install -U vllm --pre \
    --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly

# Prevents build errors in older containers, e.g.:
#   subprocess.CalledProcessError: Command ['ninja', '-v', '-C',
#   '/root/.cache/flashinfer/.../trtllm_comm', '-f', '...build.ninja']
#   returned non-zero exit status 1.
rm -rf ~/.cache/flashinfer

pip install -U "transformers @ git+https://github.com/huggingface/transformers.git@f2ba019"

# Adds "partial_rotary_factor" to ignore_keys_at_rope_validation to prevent
# a validation error when using partial rotary embeddings.
TF_FILE="$(python -m pip show transformers | awk -F': ' '/^Location:/{print $2}')/transformers/modeling_rope_utils.py"
echo "Patching: $TF_FILE"
NEW_LINE='            ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation) | {"partial_rotary_factor"}' \
    perl -i.bak -pe 'if ($. == 651) { $_ = $ENV{NEW_LINE} . "\n" }' "$TF_FILE"

pip install sentence-transformers faiss-cpu fastapi uvicorn pydantic

echo ""
echo "Setup complete. Activate with: conda activate Qwen9B"

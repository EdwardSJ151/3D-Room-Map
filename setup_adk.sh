set -euo pipefail

ENV_NAME="${ADK_CONDA_ENV:-ADKScene}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_AGENT="${REPO_ROOT}/scene_agent"
ENV_AGENT="${SCENE_AGENT}/env_agent"

conda create -n "${ENV_NAME}" python=3.12 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install -U pip
pip install -r "${SCENE_AGENT}/requirements.txt"

if [ ! -f "${ENV_AGENT}/.env" ]; then
  cp "${ENV_AGENT}/.env.example" "${ENV_AGENT}/.env"
  echo "Created ${ENV_AGENT}/.env from .env.example — set GOOGLE_API_KEY before running."
else
  echo "Keeping existing ${ENV_AGENT}/.env"
fi

echo ""
echo "Setup complete."
echo "  conda activate ${ENV_NAME}"
echo "  # edit ${ENV_AGENT}/.env (GOOGLE_API_KEY required)"
echo "  # vectorstore: ${REPO_ROOT}/objects.index and objects_meta.pkl"
echo "  cd ${SCENE_AGENT} && adk web    # http://127.0.0.1:8000"
echo "  cd ${SCENE_AGENT} && adk run env_agent"

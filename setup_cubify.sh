set -euo pipefail

conda create -n Cubify python=3.11 -y

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Cubify

# Default install location: /content when on Colab, current dir otherwise.
if [ -z "${CUBIFY_REPO:-}" ]; then
    if [ -n "${COLAB_RELEASE_TAG:-}" ] || [ -d "/content" -a -w "/content" ]; then
        CUBIFY_REPO="/content/ml-cubifyanything"
    else
        CUBIFY_REPO="$(pwd)/ml-cubifyanything"
    fi
fi

# Clone the cubify repo (skip if already present).
if [ ! -d "$CUBIFY_REPO" ]; then
    git clone https://github.com/NishinoTSK/ml-cubifyanything "$CUBIFY_REPO"
fi

pip install -U pip
# cubify's setup.py still imports pkg_resources (removed in setuptools 81+),
# so pin an older setuptools/wheel and skip build isolation for the editable install.
pip install "setuptools<81" wheel
pip install --no-build-isolation -e "$CUBIFY_REPO"
pip install -U "pillow==11.3.0"
pip install torch torchvision
pip install fastapi uvicorn pydantic

echo ""
echo "Setup complete. Activate with: conda activate Cubify"
echo "CUBIFY_REPO=$CUBIFY_REPO  (export this before running cutr_api.py)"

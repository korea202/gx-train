# gx-train

git clone https://github.com/korea202/gx-train.git
cd gx-train
uv init --no-readme
uv venv .venv
source .venv/bin/activate

#pytorch 설치
UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio

# gx-train( 리눅스 uv 환경기준)

git clone https://github.com/korea202/gx-train.git  
cd gx-train  
uv init --no-readme  
uv venv .venv  
source .venv/bin/activate  
  
#pytorch 설치(auto gpu 환경 인식)  
UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio  
#나머지 라이브러리 설치
uv pip install -r requirements.txt
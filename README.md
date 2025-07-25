# gx-train( 리눅스 uv 환경기준)

git clone https://github.com/korea202/gx-train.git  
cd gx-train  
uv sync  
uv run src/test.py

#가상환경 실행(optional)
source .venv/bin/activate  
  
#pytorch 설치(auto gpu 환경 인식, uv sync 실패시)  
UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio  
#나머지 라이브러리 설치
uv pip install -r requirements.txt

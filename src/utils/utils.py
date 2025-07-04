import os
import hashlib
import random

import torch # PyTorch 라이브러리
import torch.backends.cudnn as cudnn


def calculate_hash(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_hash(dst):
    hash_ = calculate_hash(dst)
    dst, _ = os.path.splitext(dst)
    with open(f"{dst}.sha256", "w") as f:
        f.write(hash_)


def read_hash(dst):
    dst, _ = os.path.splitext(dst)
    with open(f"{dst}.sha256", "r") as f:
        return f.read()
    
# seed 고정
def random_seed(seed_num):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed_num)    

import subprocess

def run_py(f_name, args) -> None: 

    cmd = ['python', f_name] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"명령어: {result.args}")
    print(f"반환 코드: {result.returncode}")
    print(f"표준 출력: {result.stdout}")
    print(f"표준 에러: {result.stderr}")

# 카카오톡에 작업 메시지 보내기
def send_kakao_message(msg:str) -> None: 

    run_py('/data/ephemeral/home/python_work/module/send_kakao_message.py', [msg])    
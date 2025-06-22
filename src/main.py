import os
import sys
from dotenv import load_dotenv, dotenv_values

# 환경변수 읽기
if (python_path := dotenv_values().get('PYTHONPATH')) and python_path not in sys.path: sys.path.append(python_path)

#필수 라이브러리 정리
import fire
import wandb

import numpy as np
import pandas as pd


def run_train(model_name, batch_size=1, num_epochs=1):
    pass

def run_inference(data=None, batch_size=64):
    pass

if __name__ == '__main__':  # python main.py

    fire.Fire({
        "train": run_train,  # python main.py train --model_name house_price_predictor
        "inference": run_inference, # python main.py inference
    })




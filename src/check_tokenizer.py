import numpy as np
import pandas as pd
from transformers import AutoTokenizer

df = pd.read_csv("data/train.csv")
tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
lens = [len(tok.encode(t)) for t in df["text"]]
lens = np.array(lens)
print("token length stats:", lens.min(), lens.mean(), lens.max(), np.percentile(lens, 99))

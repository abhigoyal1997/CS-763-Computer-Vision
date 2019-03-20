import torch
import numpy as np
import pandas as pd


preds = torch.load('testPrediction.bin')
df = pd.DataFrame(data=np.vstack([range(preds.shape[0]),preds]).transpose(), columns=['id','label'])
df.to_csv('preds.csv', index=False)

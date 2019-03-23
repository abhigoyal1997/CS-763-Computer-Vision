import numpy as np
import torch
from tqdm import tqdm
from src.BatchLoader import BatchLoader


def test(model, instances, lengths):
    count_instances = instances.shape[0]
    batch_size = 128
    batches = BatchLoader(range(count_instances), batch_size, instances, lengths)

    predicted_labels = []
    for x in tqdm(batches, desc='Predicting: ', total=len(batches)):
        # Forward
        logits = model.forward(x)
        predictions = logits[:,-1,0].sigmoid().round()
        predicted_labels += predictions.numpy().tolist()

    return np.array(predicted_labels)
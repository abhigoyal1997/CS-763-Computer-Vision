import numpy as np
import torch
from tqdm import tqdm
from src.BatchLoader import BatchLoader


def test(model, instances):
    count_instances = instances.shape[0]
    batch_size = 128  # count_instances = 29160 = 40*729
    batches = BatchLoader(range(count_instances), batch_size, instances)

    predicted_labels = []
    for x in tqdm(batches, desc='Predicting: ', total=len(batches)):
        # Forward
        logits = model.forward(x)
        predictions = torch.argmax(logits, dim=1)
        predicted_labels += predictions.numpy().tolist()

    return np.array(predicted_labels)

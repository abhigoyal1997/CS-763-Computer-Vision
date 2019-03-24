import numpy as np
import torch
from tqdm import tqdm
from src.BatchLoader import BatchLoader


def test(model, instances, lengths):
    count_instances = instances.shape[0]
    batch_size = 128
    batches = BatchLoader(range(count_instances), batch_size, instances, lengths)

    predicted_labels = []
    for x,lengths in tqdm(batches, desc='Predicting: ', total=len(batches)):
        # Forward
        logits = model.forward(x)
        logits_extracted = torch.Tensor([logits[i,lengths[i]-1,0] for i in range(logits.size(0))])
        predictions = logits_extracted.sigmoid().round()
        predicted_labels += predictions.numpy().tolist()

    return np.array(predicted_labels)
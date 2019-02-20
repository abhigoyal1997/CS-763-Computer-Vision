import numpy as np
import torch

from src.Model import Model
from src.Linear import Linear
from src.ReLU import ReLU
from src.Dropout import Dropout

from src.Criterion import Criterion
# torch.set_default_tensor_type(torch.DoubleTensor)

def test(model, instances):
    count_instances = instances.shape[0]
    criterion = Criterion()

    batch_size = 120 # count_instances = 29160 = 40*729

    predicted_labels = []
    for ite in range(int(np.ceil(count_instances/batch_size))):
        start, end = ite*batch_size, min((ite+1)*batch_size, count_instances)
        instances_batch = instances_shuffled[start:end]
        
        # Forward and Backward Pass
        instances_tensor = torch.tensor(instances_batch)
        output = model.forward(instances_tensor)
        predictions = torch.argmax(output, dim=1)
        predicted_labels += output.numpy().tolist()
    
    return np.array(predicted_labels)



import numpy as np
import torch

from src.Model import Model
from src.Linear import Linear
from src.ReLU import ReLU
from src.Dropout import Dropout

from src.Criterion import Criterion
# torch.set_default_tensor_type(torch.DoubleTensor)

class GradientUpdate:
    def __init__(self, model):
        self.model = model
        self.num_layers = len(model.layers)
        self.step = [None]*self.num_layers
        for layer_index in range(self.num_layers):
            layer = self.model.layers[layer_index]
            if 'Linear' in repr(layer):
                self.step[layer_index] = (torch.zeros(layer.W.shape), torch.zeros(layer.B.shape))
    
    def weightsUpdate(self, learning_rate, momentum):
        for layer_index in range(self.num_layers):
            layer = self.model.layers[layer_index]
            if 'Linear' in repr(layer):
                step_W_prev, step_B_prev = self.step[layer_index]
                # print(step_W_prev.shape)
                # print(layer.gradW.shape)
                # print(step_B_prev.shape)
                # print(layer.gradB.shape)
                step_W = momentum*step_W_prev + learning_rate*layer.gradW
                step_B = momentum*step_B_prev + learning_rate*layer.gradB
                layer.W -= step_W
                layer.B -= step_B
                self.step[layer_index] = (step_W, step_B)

def shuffle(a, b):
    # a & b are numpy arrays with the same first dimension
    assert a.shape[0] == b.shape[0], "Problem with shuffle() in training.py"
    perm = np.random.permutation(a.shape[0])
    return a[perm], b[perm]

def train(model, hparams, instances, labels):
    batch_size = hparams['batch_size']
    learning_rate = hparams['learning_rate']
    num_epochs = hparams['num_epochs']
    momentum = hparams['momentum']
    verbose = hparams['verbose']

    count_instances = instances.shape[0]
    criterion = Criterion()
    gradient_updater = GradientUpdate(model)

    for epoch in range(num_epochs):
        instances_shuffled, labels_shuffled = shuffle(instances, labels)
        loss_list = []
        acc_list = []
        for ite in range(int(np.ceil(count_instances/batch_size))):
            start, end = ite*batch_size, min((ite+1)*batch_size, count_instances)
            instances_batch, labels_batch = instances_shuffled[start:end], labels_shuffled[start:end]
            
            # Forward and Backward Pass
            instances_tensor, labels_tensor = torch.tensor(instances_batch), torch.tensor(labels_batch)
            output = model.forward(instances_tensor)
            loss = criterion.forward(output, labels_tensor)
            gradient = criterion.backward(output, labels_tensor)
            model.backward(instances_tensor, gradient)
            
            # Weights update
            gradient_updater.weightsUpdate(learning_rate, momentum)
            # Clear Grad
            model.clearGradParam()

            # Store things
            loss_list.append(loss.numpy())
            labels_predicted = torch.argmax(output,dim=1).numpy()
            acc_list.append(np.sum(labels_predicted == labels_batch))
        avg_loss = np.mean(np.array(loss_list))
        avg_acc = np.mean(np.array(acc_list))
        if verbose:
            print("Epoch " + str(epoch) + " completed. Loss = " + str(avg_loss) + ". Accuracy = " + str(avg_acc))
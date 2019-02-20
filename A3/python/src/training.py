import numpy as np
import torch

from tqdm import tqdm
from src.Linear import Linear
from src.Criterion import Criterion
torch.set_default_tensor_type(torch.DoubleTensor)


class Optimizer:
    def __init__(self, model, lr, momentum):
        self.model = model
        self.num_layers = len(model.layers)
        self.lr = lr
        self.momentum = momentum
        self.step = [None]*self.num_layers
        for layer_index in range(self.num_layers):
            layer = self.model.layers[layer_index]
            if isinstance(layer, Linear):
                self.step[layer_index] = (torch.zeros(layer.W.shape), torch.zeros(layer.B.shape))

    def step(self):
        for layer_index in range(self.num_layers):
            layer = self.model.layers[layer_index]
            if isinstance(layer, Linear):
                step_W_prev, step_B_prev = self.step[layer_index]
                # print(step_W_prev.shape)
                # print(layer.gradW.shape)
                # print(step_B_prev.shape)
                # print(layer.gradB.shape)
                step_W = self.momentum*step_W_prev + self.lr*layer.gradW
                step_B = self.momentum*step_B_prev + self.lr*layer.gradB
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
    lr = hparams['learning_rate']
    num_epochs = hparams['num_epochs']
    momentum = hparams['momentum']
    verbose = hparams['verbose']

    count_instances = instances.size(0)
    criterion = Criterion()
    optimizer = Optimizer(model, lr=lr, momentum=momentum)

    for epoch in range(num_epochs):
        instances_shuffled, labels_shuffled = shuffle(instances, labels)
        loss_list = []
        acc_list = []
        for ite in tqdm(range(int(np.ceil(count_instances/batch_size))), desc='Epoch {}: '.format(epoch)):
            start, end = ite*batch_size, min((ite+1)*batch_size, count_instances)
            x, y = instances_shuffled[start:end], labels_shuffled[start:end]

            # Forward and Backward Pass
            logits = model.forward(x)
            model.clearGradParam()  # Clear Grad
            loss = criterion.forward(logits, y)
            gradient = criterion.backward(logits, y)
            model.backward(x, gradient)

            # Weights update
            optimizer.step()

            # Store things
            loss_list.append(loss.numpy())
            predictions = torch.argmax(logits,dim=1).numpy()
            acc_list.append(np.sum(predictions == y))
        avg_loss = np.mean(np.array(loss_list))
        avg_acc = np.mean(np.array(acc_list))
        if verbose:
            print("Epoch " + str(epoch) + ": Loss = " + str(avg_loss) + ". Accuracy = " + str(avg_acc))

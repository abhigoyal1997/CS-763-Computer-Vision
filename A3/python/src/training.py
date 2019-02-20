import numpy as np
import torch
import os

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

    def updateStep(self):
        for layer_index in range(self.num_layers):
            layer = self.model.layers[layer_index]
            if isinstance(layer, Linear):
                step_W_prev, step_B_prev = self.step[layer_index]
                step_W = self.momentum*step_W_prev + self.lr*layer.gradW
                step_B = self.momentum*step_B_prev + self.lr*layer.gradB
                layer.W -= step_W
                layer.B -= step_B
                self.step[layer_index] = (step_W, step_B)


class BatchLoader():
    def __init__(self, indices, batch_size, data, labels, shuffle=False):
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data
        self.labels = labels

    def __iter__(self):
        if shuffle:
            np.random.shuffle(self.indices)
        idx = 0
        while idx+self.batch_size <= len(self.indices):
            batch_idx = self.indices[idx:idx+self.batch_size]
            yield self.data[batch_idx], self.labels[batch_idx]
            idx += self.batch_size
        if idx < len(self.indices):
            batch_idx = self.indices[idx:]
            yield self.data[batch_idx], self.labels[batch_idx]

    def __len__(self):
        return int((len(self.indices) + self.batch_size - 1)/self.batch_size)


def save_model(model, model_path, model_config):
    weights, biases = model.getParams()
    torch.save(weights, os.path.join(model_path, 'weights.bin'))
    torch.save(biases, os.path.join(model_path, 'biases.bin'))
    with open(os.path.join(model_path, 'config.txt'), 'w') as f:
        f.write(str(model_config[1]) + '\n')
        f.writelines(model_config[0])
        f.write(os.path.join(model_path, 'weights.bin')+'\n')
        f.write(os.path.join(model_path, 'biases.bin')+'\n')
    print('Model saved to {}'.format(model_path))


def shuffle(a, b):
    # a & b are numpy arrays with the same first dimension
    assert a.shape[0] == b.shape[0], "Problem with shuffle() in training.py"
    perm = np.random.permutation(a.shape[0])
    return a[perm], b[perm]


def split_dataset(num_instances, train_ratio):
    indices = torch.randperm(num_instances).tolist()
    train_size = int(num_instances*train_ratio)
    return indices[:train_size], indices[train_size:]


def run_epoch(mode, model, criterion, optimizer, batches, epoch):
    loss = 0.0
    predictions = None
    y_true = None
    for x,y in tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches)):
        # Forward Pass
        logits = model(x)
        loss = criterion(logits, y)

        if mode == 'train':
            # Backward Pass
            model.clearGradParam()  # Clear Grad
            gradient = criterion.backward(logits, y)
            model.backward(x, gradient)

            # Weights update
            optimizer.updateStep()

        # Update metrics
        loss += loss.item()
        if predictions is None:
            predictions = torch.argmax(logits,dim=1)
            y_true = y
        else:
            predictions = torch.cat([predictions, torch.argmax(logits,dim=1)])
            y_true = torch.cat([y_true, y])
    loss = np.mean(loss)
    accuracy = (predictions == y_true).sum().item()
    return {'loss': loss, 'acc': accuracy}


def train(model, hparams, instances, labels, model_path, model_config):
    batch_size = hparams['batch_size']
    lr = hparams['learning_rate']
    num_epochs = hparams['num_epochs']
    momentum = hparams['momentum']
    verbose = hparams['verbose']

    count_instances = instances.size(0)
    criterion = Criterion()
    optimizer = Optimizer(model, lr=lr, momentum=momentum)

    train_idx, valid_idx = split_dataset(count_instances, hparams['train_ratio'])
    train_batches = BatchLoader(train_idx, batch_size, instances, labels, True)
    valid_batches = BatchLoader(valid_idx, batch_size, instances, labels)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        # Train
        metrics = run_epoch('train', model, criterion, optimizer, train_batches, epoch)
        if verbose:
            print('Train: {}'.format(metrics))

        # Validate
        metrics = run_epoch('valid', model, criterion, optimizer, valid_batches, epoch)
        if verbose:
            print('Validation: {}'.format(metrics))

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_model(model, model_path, model_config)

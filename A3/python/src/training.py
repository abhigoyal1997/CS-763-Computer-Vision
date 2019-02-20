import torch
import os

from time import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from src.Criterion import Criterion
from src.BatchLoader import BatchLoader
from src.Optimizer import Optimizer
torch.set_default_tensor_type(torch.DoubleTensor)


def save_model(model, model_path, model_config):
    params = model.getParams()
    weights = params['W']
    biases = params['B']

    torch.save(weights, os.path.join(model_path, 'weights.bin'))
    torch.save(biases, os.path.join(model_path, 'biases.bin'))
    with open(os.path.join(model_path, 'config.txt'), 'w') as f:
        f.write(str(model_config[1]) + '\n')
        f.writelines(model_config[0])
        f.write(os.path.join(model_path, 'weights.bin')+'\n')
        f.write(os.path.join(model_path, 'biases.bin')+'\n')
    print('Model saved to {}'.format(model_path))


def split_dataset(num_instances, train_ratio):
    indices = torch.randperm(num_instances).tolist()
    train_size = int(num_instances*train_ratio)
    return indices[:train_size], indices[train_size:]


def run_epoch(mode, model, criterion, optimizer, batches, epoch, writer=None, log_interval=None):
    loss = 0.0
    predictions = None
    y_true = None
    i = 0
    for x,y in tqdm(batches, desc='Epoch {}: '.format(epoch), total=len(batches)):
        # Forward Pass
        logits = model(x)
        batch_loss = criterion(logits, y)

        if mode == 'train':
            # Backward Pass
            model.clearGradParam()  # Clear Grad
            gradient = criterion.backward(logits, y)
            model.backward(x, gradient)

            # Weights update
            optimizer.updateStep()

        # Update metrics
        loss += batch_loss.item()
        if predictions is None:
            predictions = torch.argmax(logits,dim=1)
            y_true = y
        else:
            predictions = torch.cat([predictions, torch.argmax(logits,dim=1)])
            y_true = torch.cat([y_true, y])

        if mode == 'train' and (log_interval is not None) and (i % log_interval == 0):
            writer.add_scalar('{}_loss'.format(mode), batch_loss.item(), epoch*len(batches)+i)
        i += 1

    loss = loss/len(batches)
    accuracy = (predictions == y_true.long()).double().mean().item()
    if writer is not None:
        writer.add_scalar('{}_acc'.format(mode), accuracy, epoch)
        if mode == 'valid':
            writer.add_scalar('{}_loss'.format(mode), loss, epoch)
    return {'loss': loss, 'acc': accuracy}


def train(model, hparams, instances, labels, model_path, model_config, log_interval=None):
    batch_size = int(hparams['batch_size'])
    lr = hparams['learning_rate']
    lr_decay = hparams['learning_rate_decay']
    num_epochs = int(hparams['num_epochs'])
    momentum = hparams['momentum']
    verbose = bool(hparams['verbose'])

    count_instances = instances.size(0)
    criterion = Criterion()
    optimizer = Optimizer(model, lr=lr, lr_decay=lr_decay, momentum=momentum)

    train_idx, valid_idx = split_dataset(count_instances, hparams['train_ratio'])
    train_batches = BatchLoader(train_idx, batch_size, instances, labels, True)
    valid_batches = BatchLoader(valid_idx, batch_size, instances, labels)

    writer = SummaryWriter(log_dir='logs/{}_{}'.format(model_path, time()))

    best_acc = 0.0
    for epoch in range(num_epochs):
        # Train
        metrics = run_epoch('train', model, criterion, optimizer, train_batches, epoch, writer, log_interval)
        if verbose:
            print('Train: {}'.format(metrics))

        # Validate
        metrics = run_epoch('valid', model, criterion, optimizer, valid_batches, epoch, writer, log_interval)
        if verbose:
            print('Validation: {}'.format(metrics))

        optimizer.updateLR()

        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            save_model(model, model_path, model_config)

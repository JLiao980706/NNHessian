import numpy as np
import torch


def train(train_data, test_data, model, loss_fn, optimizer, batch_size, num_epochs, measurement, verbose=False):
    """
        train_data: input-output pair (X, Y) where X is n*p_1*...*p_d tensor, Y is n*o tensor.
        test_data: input-output pair similar to train_data
    """
    train_X, train_Y = train_data
    num_train_samples = train_X.size(dim=0)
    num_batches = int(np.ceil(num_train_samples / batch_size))
    
    for epoch_idx in range(num_epochs):
        
        for batch_idx in range(num_batches):
            
            # Batching training data
            if batch_idx < num_batches - 1:
                batch_X = train_X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                batch_Y = train_Y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                batch_X = train_X[batch_idx * batch_size:]
                batch_Y = train_Y[batch_idx * batch_size:]

            # Backpropagation step
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X.cuda()), batch_Y.cuda())
            loss.backward()
            optimizer.step()
            if verbose:
                print(f'Current loss value is {loss.detach().cpu().numpy():.5f}.', end='\r')
        
        measurement.measure(train_data, test_data, model, epoch_idx)

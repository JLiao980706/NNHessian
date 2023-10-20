import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from pyhessian import hessian

class Measurement:
    
    def __init__(self, verbose=False):
        self.on_train_data = []
        self.on_test_data = []
        self.verbose=verbose
        self.recorders = {
            'MSE': MSERecorder,
            'Cross Entropy': CERecorder,
            'Accuracy': AccuracyRecorder,
            'Cross Entropy Sharpness': CESharpnessRecorder,
            "Cross Entropy Sharpness V2": CESharpnessRecorderV2
        }
    
    def measure(self, train_data, test_data, model, epoch_idx):
        if self.verbose:
            print(f'Epoch #{epoch_idx}')
            print(f'  Metrics on training data:')
        for m in self.on_train_data:
            m.record(train_data, model, epoch_idx, verbose=self.verbose)
        if self.verbose:
            print(f'  Metrics on testing data:')
        for m in self.on_test_data:
            m.record(test_data, model, epoch_idx, verbose=self.verbose)
    
    def add_train_recorder_raw(self, recorder):
        self.on_train_data.append(recorder)
    
    def add_test_recorder_raw(self, recorder):
        self.on_test_data.append(recorder)
    
    def add_train_recorder(self, rec_name, phys_batch_size, verbose):
        self.add_train_recorder_raw(self.get_recorder_constr(rec_name)(phys_batch_size, verbose))
    
    def add_test_recorder(self, rec_name, phys_batch_size, verbose):
        self.add_test_recorder_raw(self.get_recorder_constr(rec_name)(phys_batch_size, verbose))
        
    def get_recorder_constr(self, rec_name):
        if rec_name in self.available_recorders():
            return self.recorders[rec_name]
        else:
            raise Exception(f'Recorder name {rec_name} not available.')
    
    def get_train_recorder(self):
        return self.on_train_data
    
    def get_test_recorder(self):
        return self.on_test_data
    
    def available_recorders(self):
        return list(self.recorders.keys())
        
        
class Recorder:
    
    def __init__(self, physical_batch_size, verbose=False):
        self.batch_size = physical_batch_size
        self.verbose=verbose
        self.records = []
    
    def record(self, data, model, epoch_idx, verbose):
        self.records.append((epoch_idx, self.compute(data, model)))
        if verbose and self.verbose:
            print(f"    {self.get_name()}: {self.records[-1][1]}.")
        
    def batching(self, data):
        X, Y = data
        num_samples = X.size(dim=0)
        num_batches = int(np.ceil(num_samples / self.batch_size))
        data_batches = []
        for batch_idx in range(num_batches):
            if batch_idx < num_batches - 1:
                batch_X = X[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                batch_Y = Y[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
            else:
                batch_X = X[batch_idx * self.batch_size:]
                batch_Y = Y[batch_idx * self.batch_size:]
            data_batches.append((batch_X, batch_Y))
        return data_batches
        
    def compute(self, data, model):
        raise Exception('Method "compute" not implemented.')

    def get_name(self):
        raise Exception('Method "compute" not implemented.')
    
    def get_records(self):
        return self.records
    

class MSERecorder(Recorder):
    
    def __init__(self, physical_batch_size, verbose=False):
        super(MSERecorder, self).__init__(physical_batch_size, verbose=verbose)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
    
    def compute(self, data, model):
        data_batches = self.batching(data)
        error = 0.
        for X, Y in data_batches:
            error += 0.5 * self.loss_fn(model(X.cuda()), Y.cuda())
        error /= data[0].size(dim=0)
        return error.detach().cpu().item()
    
    def get_name(self):
        return "MSE"


class CERecorder(Recorder):
    
    def __init__(self, physical_batch_size, verbose=False):
        super(CERecorder, self).__init__(physical_batch_size, verbose=verbose)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    def compute(self, data, model):
        data_batches = self.batching(data)
        error = 0.
        for X, Y in data_batches:
            error += self.loss_fn(model(X.cuda()), Y.cuda())
        error /= data[0].size(dim=0)
        return error.detach().cpu().item()
    
    def get_name(self):
        return "Cross Entropy"


class AccuracyRecorder(Recorder):
    
    def __init__(self, physical_batch_size, verbose=False):
        super(AccuracyRecorder, self).__init__(physical_batch_size, verbose=verbose)
    
    def compute(self, data, model):
        data_batches = self.batching(data)
        error = 0.
        for X, Y in data_batches:
            error += (model(X.cuda()).detach().cpu().numpy().argmax(1) == Y.numpy()).astype(np.float32).sum()
        error /= data[0].size(dim=0)
        return error
    
    def get_name(self):
        return "Accuracy"
    

class CESharpnessRecorder(Recorder):
        
    def __init__(self, physical_batch_size, verbose=False):
        super(CESharpnessRecorder, self).__init__(physical_batch_size, verbose=verbose)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
    def compute(self, data, model):
        X, Y = data
        hessian_comp = hessian(model, self.loss_fn, data=(X, Y), cuda=True)
        eig_val = hessian_comp.eigenvalues(top_n=1)[0][0]
        return eig_val
    
    def get_name(self):
        return "Cross Entropy Sharpness"
    
class CESharpnessRecorderV2(Recorder):
    
    def __init__(self, physical_batch_size, verbose=False):
        super(CESharpnessRecorderV2, self).__init__(physical_batch_size, verbose=verbose)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        
    def compute(self, data, model):
        return get_hessian_eigenvalues(model, self.loss_fn, data, neigs=1)[0]
    
    def get_name(self):
        return "Cross Entropy Sharpness V2"
    
def compute_hvp(model, loss_fn, dataset, vector):
    
    p = len(parameters_to_vector(model.parameters()))
    X, Y = dataset
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    loss = loss_fn(model(X.cuda()), Y.cuda())
    grads = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
    dot = parameters_to_vector(grads).mul(vector).sum()
    grads = [g.contiguous() for g in torch.autograd.grad(dot, model.parameters(), retain_graph=True)]
    hvp += parameters_to_vector(grads)
    return hvp


def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
        torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def get_hessian_eigenvalues(model, loss_fn, dataset, neigs=6):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(model, loss_fn, dataset,
                                        delta).detach().cpu()
    nparams = len(parameters_to_vector((model.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals
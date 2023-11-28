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
            'Binary Accuracy': BinaryAccuracyRecorder,
            'Binary Cross Entropy': BCELossRecorder,
            'Cross Entropy Sharpness': CESharpnessRecorder,
            'MSE Sharpness': MSESharpnessRecorder,
            'Hessian Second Order Term Norm': MSESecondOrderTermNorm
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
    

class BinaryAccuracyRecorder(Recorder):
    
    def __init__(self, physical_batch_size, verbose=False):
        super(BinaryAccuracyRecorder, self).__init__(physical_batch_size, verbose=verbose)
        self.sig_layer = torch.nn.Sigmoid()
    
    def compute(self, data, model):
        data_batches = self.batching(data)
        error = 0.
        for X, Y in data_batches:
            prob_output = self.sig_layer(model(X.cuda())).detach().cpu().numpy()
            error += ((prob_output > 0.5).astype(np.float32) == Y.numpy()).astype(np.float32).sum()
        error /= data[0].size(dim=0)
        return error
    
    def get_name(self):
        return "Binary Accuracy"
    

class BCELossRecorder(Recorder):
    
    def __init__(self, physical_batch_size, verbose=False):
        super(BCELossRecorder, self).__init__(physical_batch_size, verbose=verbose)
        self.sig_layer = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss(reduction='sum')
    
    def compute(self, data, model):
        data_batches = self.batching(data)
        error = 0.
        for X, Y in data_batches:
            error += self.loss_fn(self.sig_layer(model(X.cuda())), Y.cuda())
        error /= data[0].size(dim=0)
        return error.detach().cpu().item()
    
    def get_name(self):
        return "Binary Cross Entropy"
    

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
    

class MSESharpnessRecorder(Recorder):
        
    def __init__(self, physical_batch_size, verbose=False):
        super(MSESharpnessRecorder, self).__init__(physical_batch_size, verbose=verbose)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
    
    def compute(self, data, model):
        X, Y = data
        hessian_comp = hessian(model, self.loss_fn, data=(X, Y), cuda=True)
        eig_val = hessian_comp.eigenvalues(top_n=1)[0][0]
        return eig_val
    
    def get_name(self):
        return "MSE Sharpness"
    
    
class MSESecondOrderTermNorm(Recorder):
    
    def __init__(self, physical_batch_size,  verbose=False):
        super(MSESecondOrderTermNorm, self).__init__(1, verbose=verbose)
        
        
    def compute(self, data, model, maxIter=100, tol=1e-3):
        model.eval().zero_grad()
        params = params_with_grad(model)
        D = compute_D(data, model, params)
        eigenvalue = None
        v = [torch.randn(p.size()).to('cuda') for p in params]  # generate random vector
        v = normalization(v)  # normalize the vector

        for i in range(maxIter):
            # v = orthnormal(v, eigenvectors)
            model.zero_grad()
            Hv = h_hat_v(D, v, params)
            tmp_eigenvalue = group_product(Hv, v).cpu().item()

            v = normalization(Hv)

            if eigenvalue == None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                        1e-6) < tol:
                    break
                else:
                    eigenvalue = tmp_eigenvalue
        model.train()
        return eigenvalue
    
    def get_name(self):
        return "Hessian Second Order Term Norm"
    
    

def params_with_grad(model):
    return [p for p in model.parameters() if p.requires_grad]


def compute_D(data, model, params):
    X, Y = data
    model_output = model(X.cuda())
    dloss = torch.flatten(model_output - Y.cuda()).detach() / X.shape[0]
    return torch.autograd.grad(torch.flatten(model_output), params, grad_outputs=dloss, retain_graph=True, create_graph=True)


def h_hat_v(D, v, params):
    return torch.autograd.grad(D, params, grad_outputs=v, retain_graph=True)


def group_product(xs, ys):
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v
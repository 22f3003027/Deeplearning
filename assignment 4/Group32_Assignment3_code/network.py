"""ReLU FCNN and exact sample-wise optimizer updates compiled with Numba.

Batch size one is preserved: each image is followed by exactly one parameter
update. JIT compilation eliminates Python/autograd overhead, not SGD semantics.
Parameters and gradients are float64 for numerical checks against PyTorch.
"""
from __future__ import annotations

import hashlib
import numpy as np
# Numba loads SciPy BLAS lazily. Load it before the caller limits threads so
# online updates do not create a second, uncontrolled BLAS thread pool.
from scipy.linalg import blas as _blas
from numba import njit
from numba.typed import List

ARCHITECTURES = {
    "3_layers": (32,16,8),
    "4_layers": (32,24,16,8),
    "5_layers": (32,24,16,12,8),
}
OPTIMIZERS = {
    "SGD": {"batch_size":1,"kind":0},
    "Batch_GD": {"batch_size":"full","kind":0},
    "Momentum": {"batch_size":1,"kind":1},
    "NAG": {"batch_size":1,"kind":2},
    "AdaGrad": {"batch_size":"full","kind":3},
    "RMSProp": {"batch_size":"full","kind":4},
    "Adam": {"batch_size":1,"kind":5},
}


def initialize(hidden, seed=32, input_dim=784, classes=5):
    rng=np.random.default_rng(seed)
    widths=(input_dim,*hidden,classes)
    weights=List(); biases=List()
    for a,b in zip(widths[:-1],widths[1:]):
        weights.append(np.ascontiguousarray(rng.normal(0,np.sqrt(2/a),(b,a))))
        biases.append(np.zeros(b,np.float64))
    return weights,biases


def clone(values):
    return List([v.copy() for v in values])


def parameter_hash(weights,biases):
    h=hashlib.sha256()
    for w,b in zip(weights,biases):
        h.update(w.tobytes()); h.update(b.tobytes())
    return h.hexdigest()


def buffers(weights,biases):
    return (List([np.zeros_like(w) for w in weights]),List([np.zeros_like(b) for b in biases]))


def forward(x,weights,biases):
    a=x
    for i,(w,b) in enumerate(zip(weights,biases)):
        a=a @ w.T+b
        if i<len(weights)-1:
            a=np.maximum(a,0)
    return a


def evaluate(x,y,weights,biases):
    logits=forward(x,weights,biases)
    shifted=logits-logits.max(axis=1,keepdims=True)
    loss=float((np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(y)),y]).mean())
    predictions=logits.argmax(axis=1)
    confusion=np.bincount(y*5+predictions,minlength=25).reshape(5,5)
    return {"loss":loss,"accuracy":float((predictions==y).mean()),"confusion_matrix":confusion.tolist()}


def full_gradients(x,y,weights,biases):
    activations=[x]
    for i,(w,b) in enumerate(zip(weights,biases)):
        a=activations[-1] @ w.T+b
        activations.append(np.maximum(a,0) if i<len(weights)-1 else a)
    logits=activations[-1]
    probabilities=np.exp(logits-logits.max(axis=1,keepdims=True))
    probabilities/=probabilities.sum(axis=1,keepdims=True)
    probabilities[np.arange(len(y)),y]-=1
    delta=probabilities/len(y)
    gw=[np.empty_like(w) for w in weights]; gb=[np.empty_like(b) for b in biases]
    for layer in range(len(weights)-1,-1,-1):
        gw[layer][:]=delta.T @ activations[layer]
        gb[layer][:]=delta.sum(axis=0)
        if layer:
            delta=(delta @ weights[layer])*(activations[layer]>0)
    return gw,gb


def full_batch_update(x,y,weights,biases,first_w,first_b,second_w,second_b,kind,lr=.001):
    gw,gb=full_gradients(x,y,weights,biases)
    for params,gradients,first,second in ((weights,gw,first_w,second_w),(biases,gb,first_b,second_b)):
        for p,g,m,v in zip(params,gradients,first,second):
            if kind==0:
                p-=lr*g
            elif kind==3:
                v+=g*g
                p-=lr*g/(np.sqrt(v)+1e-8)
            elif kind==4:
                v*=.99; v+=.01*g*g
                p-=lr*g/(np.sqrt(v)+1e-8)
            else:
                raise ValueError("Unsupported full-batch optimizer")


@njit(cache=True)
def online_epoch(x,y,indices,weights,biases,first_w,first_b,second_w,second_b,kind,step,lr=.001):
    activations=[np.zeros(x.shape[1])]
    deltas=[np.zeros(b.shape[0]) for b in biases]
    for b in biases:
        activations.append(np.zeros(b.shape[0]))
    for index in indices:
        activations[0][:]=x[index]
        for layer in range(len(weights)):
            activations[layer+1][:]=weights[layer] @ activations[layer]+biases[layer]
            if layer<len(weights)-1:
                for k in range(len(activations[layer+1])):
                    activations[layer+1][k]=max(0.0,activations[layer+1][k])
        logits=activations[-1]
        maximum=np.max(logits)
        total=0.0
        for k in range(len(logits)):
            deltas[-1][k]=np.exp(logits[k]-maximum)
            total+=deltas[-1][k]
        deltas[-1]/=total
        deltas[-1][y[index]]-=1
        for layer in range(len(weights)-2,-1,-1):
            deltas[layer][:]=weights[layer+1].T @ deltas[layer+1]
            for k in range(len(deltas[layer])):
                if activations[layer+1][k]<=0:
                    deltas[layer][k]=0
        step+=1
        correction1=1.0-.9**step
        correction2=1.0-.999**step
        for layer in range(len(weights)):
            w=weights[layer]; b=biases[layer]
            a=activations[layer]; delta=deltas[layer]
            mw=first_w[layer]; mb=first_b[layer]
            vw=second_w[layer]; vb=second_b[layer]
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    g=delta[i]*a[j]
                    if kind==0:
                        w[i,j]-=lr*g
                    elif kind==1 or kind==2:
                        mw[i,j]=.9*mw[i,j]+g
                        direction=mw[i,j] if kind==1 else g+.9*mw[i,j]
                        w[i,j]-=lr*direction
                    elif kind==5:
                        mw[i,j]=.9*mw[i,j]+.1*g
                        vw[i,j]=.999*vw[i,j]+.001*g*g
                        w[i,j]-=lr*(mw[i,j]/correction1)/(np.sqrt(vw[i,j]/correction2)+1e-8)
                g=delta[i]
                if kind==0:
                    b[i]-=lr*g
                elif kind==1 or kind==2:
                    mb[i]=.9*mb[i]+g
                    direction=mb[i] if kind==1 else g+.9*mb[i]
                    b[i]-=lr*direction
                elif kind==5:
                    mb[i]=.9*mb[i]+.1*g
                    vb[i]=.999*vb[i]+.001*g*g
                    b[i]-=lr*(mb[i]/correction1)/(np.sqrt(vb[i]/correction2)+1e-8)
    return step


def save_model(path,weights,biases):
    np.savez_compressed(path,**{f"w_{i}":w for i,w in enumerate(weights)},
                        **{f"b_{i}":b for i,b in enumerate(biases)})


def load_model(path):
    with np.load(path) as saved:
        n=len(saved.files)//2
        return List([saved[f"w_{i}"].copy() for i in range(n)]),List([saved[f"b_{i}"].copy() for i in range(n)])

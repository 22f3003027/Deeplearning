"""Every custom update is checked against the PyTorch reference implementation."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from network import initialize,buffers,online_epoch,full_batch_update,forward
from run_assignment3 import should_stop
torch.set_num_threads(1); threadpool_limits(1)


class OptimizerTests(unittest.TestCase):
    def compare(self,name,kind,online):
        w,b=initialize((5,4,3),seed=81,input_dim=6)
        layers=[]
        for i,(weight,bias) in enumerate(zip(w,b)):
            layer=nn.Linear(weight.shape[1],weight.shape[0]).double()
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(weight)); layer.bias.copy_(torch.from_numpy(bias))
            layers.append(layer)
            if i<len(w)-1:
                layers.append(nn.ReLU())
        model=nn.Sequential(*layers)
        options={"lr":.001}
        if name=="Momentum": options["momentum"]=.9
        if name=="NAG": options.update(momentum=.9,nesterov=True)
        if name=="AdaGrad": options["eps"]=1e-8
        if name=="RMSProp": options.update(alpha=.99,eps=1e-8)
        if name=="Adam": options.update(betas=(.9,.999),eps=1e-8)
        factory={"SGD":torch.optim.SGD,"Momentum":torch.optim.SGD,"NAG":torch.optim.SGD,
                 "AdaGrad":torch.optim.Adagrad,"RMSProp":torch.optim.RMSprop,"Adam":torch.optim.Adam}[name]
        optimizer=factory(model.parameters(),**options)
        x=np.random.default_rng(11).normal(size=(4,6)); y=np.array([0,1,3,4],dtype=np.int64)
        fw,fb=buffers(w,b); sw,sb=buffers(w,b); step=0
        for _ in range(3):
            if online:
                step=online_epoch(x,y,np.arange(len(y)),w,b,fw,fb,sw,sb,kind,step)
                for i in range(len(y)):
                    optimizer.zero_grad(); nn.functional.cross_entropy(model(torch.from_numpy(x[i:i+1])),torch.from_numpy(y[i:i+1])).backward(); optimizer.step()
            else:
                full_batch_update(x,y,w,b,fw,fb,sw,sb,kind)
                optimizer.zero_grad(); nn.functional.cross_entropy(model(torch.from_numpy(x)),torch.from_numpy(y)).backward(); optimizer.step()
        linear=[layer for layer in model if isinstance(layer,nn.Linear)]
        for weight,bias,layer in zip(w,b,linear):
            np.testing.assert_allclose(weight,layer.weight.detach().numpy(),rtol=1e-10,atol=1e-11)
            np.testing.assert_allclose(bias,layer.bias.detach().numpy(),rtol=1e-10,atol=1e-11)
        np.testing.assert_allclose(forward(x,w,b),model(torch.from_numpy(x)).detach().numpy(),rtol=1e-10,atol=1e-11)

    def test_online_sgd(self): self.compare("SGD",0,True)
    def test_full_batch_sgd(self): self.compare("SGD",0,False)
    def test_online_momentum(self): self.compare("Momentum",1,True)
    def test_online_nag(self): self.compare("NAG",2,True)
    def test_full_batch_adagrad(self): self.compare("AdaGrad",3,False)
    def test_full_batch_rmsprop(self): self.compare("RMSProp",4,False)
    def test_online_adam(self): self.compare("Adam",5,True)
    def test_strict_loss_threshold(self):
        self.assertFalse(should_stop(None,1.0)); self.assertFalse(should_stop(1.0,1.001))
        self.assertTrue(should_stop(1.0,1.00001)); self.assertTrue(should_stop(1.0,.99999))


if __name__=="__main__": unittest.main()

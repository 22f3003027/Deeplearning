"""Focused regression tests for leakage, corruption, model shape, and checkpoint restoration."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from data import fit_pca
from models import Autoencoder, corrupt
from training import train_classifier, evaluate_checkpoint

torch.set_num_threads(2)
threadpool_limits(2)


class PipelineTests(unittest.TestCase):
    def test_pca_never_fits_evaluation_data(self):
        rng=np.random.default_rng(9)
        training=rng.normal(size=(30,7)).astype(np.float32)
        validation=rng.normal(size=(9,7)).astype(np.float32)+1000
        data={"train":(training,np.zeros(30,dtype=np.int64)),
              "val":(validation,np.zeros(9,dtype=np.int64)),
              "test":(validation*2,np.zeros(9,dtype=np.int64))}
        with tempfile.TemporaryDirectory() as temporary:
            destination=Path(temporary)/"pca.npz"
            features,_=fit_pca(data,[2,4],destination)
            with np.load(destination) as saved:
                np.testing.assert_allclose(saved["mean"],training.mean(axis=0,dtype=np.float64))
                expected=(validation.astype(np.float64)-saved["mean"]) @ saved["components"].T
                np.testing.assert_allclose(features["val"],expected,rtol=1e-6)
                before=saved["components"].copy()
            data["val"]=(validation+99999,data["val"][1])
            fit_pca(data,[2,4],destination)
            with np.load(destination) as saved:
                np.testing.assert_allclose(saved["components"],before)

    def test_required_architectures_and_activation_range(self):
        for deep in (False,True):
            for dimension in (32,64,128,256):
                model=Autoencoder(dimension,deep)
                hidden_widths=[m.out_features for m in model.modules() if isinstance(m,nn.Linear)]
                self.assertEqual(hidden_widths,[400,dimension,400,784] if deep else [dimension,784])
                x=torch.rand(3,784)
                self.assertEqual(tuple(model.encoder(x).shape),(3,dimension))
                output=model(x)
                self.assertEqual(tuple(output.shape),(3,784))
                self.assertTrue(bool(((output>=0)&(output<=1)).all()))
                self.assertEqual(sum(isinstance(m,nn.Sigmoid) for m in model.modules()),4 if deep else 2)

    def test_corruption_is_seeded_and_preserves_clean_targets(self):
        clean=torch.ones(200,784)
        for probability in (.2,.4):
            first=corrupt(clean,probability,torch.Generator().manual_seed(41))
            second=corrupt(clean,probability,torch.Generator().manual_seed(41))
            self.assertTrue(torch.equal(first,second))
            self.assertLess(abs(float((first==0).double().mean())-probability),.005)
            self.assertTrue(torch.equal(clean,torch.ones_like(clean)))

    def test_saved_classifier_restores_validation_selected_state(self):
        rng=np.random.default_rng(23)
        features,labels={},{}
        for split,n in (("train",100),("val",40),("test",40)):
            y=np.arange(n,dtype=np.int64)%5
            x=np.eye(5,dtype=np.float32)[y]+rng.normal(0,.03,(n,5)).astype(np.float32)
            features[split],labels[split]=x,y
        config={"classifier_lr":.01,"classifier_epochs":20,"classifier_patience":4,"batch_size":25}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint=Path(temporary)/"model.pt"
            _,metadata,_,_=train_classifier(features,labels,(16,),config,32,checkpoint)
            evaluated=evaluate_checkpoint(checkpoint,features,labels)
            self.assertEqual(evaluated["val"]["accuracy"],metadata["val_accuracy"])
            self.assertEqual(metadata["val_accuracy"],max(h["val_accuracy"] for h in metadata["history"]))
            self.assertGreater(evaluated["test"]["accuracy"],.95)


if __name__=="__main__":
    unittest.main()

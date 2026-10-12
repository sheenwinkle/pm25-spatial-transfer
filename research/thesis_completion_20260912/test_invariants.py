import unittest
from io import StringIO
import numpy as np
import pandas as pd
from run_monitor_models import kernel, maskrule, embed, tree, supported, bootstrap
from audit_results import read_predictions

class Invariants(unittest.TestCase):
    def test_block_bootstrap_matches_direct_row_weights(self):
        d=pd.DataFrame({'eFUA_ID':[1,1,2,2,3,3,3], 'quarter_id':[1,2,1,3,1,2,3],
            'observed_pm25':[2.,4.,5.,3.,6.,7.,4.], 'm':[3.,3.,4.,4.,5.,6.,5.],
            'r':[4.,2.,2.,6.,3.,8.,7.]})
        rng=np.random.default_rng(5993)
        cw=rng.multinomial(3,np.full(3,1/3),1000)
        tw=rng.multinomial(3,np.full(3,1/3),1000)
        c=pd.factorize(d.eFUA_ID)[0];t=pd.factorize(d.quarter_id)[0];delta=[]
        for i in range(1000):
            w=cw[i,c]*tw[i,t]
            if w.sum():
                delta.append(np.sqrt(np.average((d.r-d.observed_pm25)**2,weights=w))-
                             np.sqrt(np.average((d.m-d.observed_pm25)**2,weights=w)))
        self.assertTrue(np.allclose(bootstrap(d,'m','r',5993,'eFUA_ID'),np.quantile(delta,[.025,.975]),atol=1e-12))
    def test_saved_csv_preserves_inclusive_gate_boundary(self):
        threshold=1.8109632833995406
        d=read_predictions(StringIO('support_distance\n1.8109632833995406\n'))
        self.assertEqual(d.support_distance.iloc[0],threshold)
        self.assertTrue(maskrule(d,[{'feature':'support_distance','op':'<=','threshold':threshold}])[0])
    def test_kernel_is_psd_and_diagonal_one(self):
        rng=np.random.default_rng(2);x=pd.DataFrame(rng.normal(size=(200,3)));y=x[0]*2+rng.normal(size=200)
        e=embed(tree(True).fit(x,y),x);k=(e@e.T).toarray()
        self.assertTrue(np.allclose(np.diag(k),1))
        self.assertGreaterEqual(np.linalg.eigvalsh(k).min(),-1e-9)
    def test_constant_target_is_preserved(self):
        k=np.eye(6);p=kernel(k,np.ones((2,6))/6,np.full(6,4.5))
        self.assertTrue(np.allclose(p,4.5))
    def test_gate_exact_boundaries_and_missing(self):
        d=pd.DataFrame({'x':[1.,2.,3.,np.nan]})
        r=[{'feature':'x','op':'>=','threshold':1.},{'feature':'x','op':'<=','threshold':2.}]
        self.assertEqual(maskrule(d,r).tolist(),[True,True,False,False])
    def test_support_not_only_row_count(self):
        d=pd.DataFrame({'eFUA_ID':np.repeat([1,2,3,4],12),'t':np.tile(np.arange(12),4)})
        self.assertFalse(supported(d,d.eFUA_ID.eq(1).to_numpy()))
        self.assertTrue(supported(d,d.eFUA_ID.le(3).to_numpy()))

if __name__=='__main__':unittest.main()

"""Focused regression tests for the added source-partition transfer path."""
import unittest
import numpy as np
from sklearn.linear_model import Ridge

import run_models as m


class FixedLeaves:
    def predict(self,x,pred_leaf=False):
        assert pred_leaf
        return np.asarray(x)


class TransferTests(unittest.TestCase):
    def test_leaf_similarity_is_shared_fraction(self):
        leaves=np.array([[0,0],[0,1],[2,1]])
        em=m.leaf_embedding(FixedLeaves(),leaves)
        k=(em@em.T).toarray()
        np.testing.assert_allclose(k,[[1,.5,0],[.5,1,.5],[0,.5,1]],atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(k).min(),-1e-12)

    def test_kernel_constant_labels_return_local_mean(self):
        em=m.leaf_embedding(FixedLeaves(),[[0,0],[0,1],[2,1]])
        k=(em@em.T).toarray()
        np.testing.assert_allclose(m.kernel_predict(k,k,np.array([4.,4.,4.])),4.)

    def test_kernel_matches_linear_ridge_on_leaf_features(self):
        em=m.leaf_embedding(FixedLeaves(),[[0,0],[0,1],[2,1],[2,2]]).toarray()
        y=np.array([2.,4.,6.])
        expected=Ridge(alpha=m.P["kernel_ridge_alpha"],fit_intercept=True).fit(em[:3],y).predict(em)
        actual=m.kernel_predict(em[:3]@em[:3].T,em@em[:3].T,y)
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_one_tree_shape(self):
        em=m.leaf_embedding(FixedLeaves(),np.array([0,1,0]))
        self.assertEqual(em.shape,(3,15))
        np.testing.assert_allclose((em@em.T).diagonal(),1.)


if __name__=="__main__":unittest.main()

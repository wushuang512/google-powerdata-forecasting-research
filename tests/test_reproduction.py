"""Synthetic formula checks only, not evidence of real model accuracy."""
from pathlib import Path
import importlib.util,unittest,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from repro.google import bounded
from repro.support import initialize
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
common=load('test_common','experiment/code/common.py')
nlr=load('test_nlr_metrics','experiment/nlr_strict_protocol_20261013/code/core.py')
aligned=load('test_aligned','experiment/round29/code/aligned_metrics.py')
class Metrics(unittest.TestCase):
 def test_work_directory_must_be_outside_checkout(self):
  with self.assertRaises(ValueError):initialize(ROOT/'must-not-be-created')
 def test_macro_not_pooled(self):
  y=np.ones((57,1,96));y[0]=100;p=y.copy();p[0]+=10;p[1:]+=.2;m=np.ones_like(y,bool)
  expected=(10+56*20)/57
  got=common.metrics(y,p,m)['wape'];pooled=100*abs(y-p).sum()/abs(y).sum()
  self.assertAlmostEqual(got,expected);self.assertGreater(abs(got-pooled),1)
 def test_complete_day_mask(self):
  y=np.ones((57,31,96));y[:,21]=.8;p=np.ones((57,2,96));p[:,1]=2;valid=np.ones_like(y,bool);valid[:,23,-1]=False
  d={'y':y,'valid':valid,'cells':np.array(['a']*5+['b']*52)}
  got=aligned.evaluate(common,d,[22,23],p,1)
  self.assertEqual(got['valid_daily_targets'],57);self.assertEqual(got['daily_wape'],0)
  self.assertGreater(got['daily_wape_valid_slots_supplement'],0)
 def test_zero_mape_and_origin_mean(self):
  y=np.ones((2,96));y[0]=0;y[0,0]=1;p=y.copy();p[0,0]=1.5;p[1]=1.1
  result=nlr.metrics(y,p);self.assertEqual(result['mape_excluded'],95);self.assertEqual(result['mape_n'],97)
  self.assertAlmostEqual(result['curve_mape_percent'],30);self.assertNotAlmostEqual(result['curve_mape_percent'],result['curve_mape_pooled_percent'])
 def test_bounded_daily_mean(self):
  rng=np.random.default_rng(42);prior=rng.uniform(.2,.8,(57,2,672));delta=rng.normal(0,1,prior.shape)
  out=bounded(prior,delta);expected=np.clip((prior+delta).reshape(57,-1,96).mean(-1),0,1)
  self.assertTrue(((out>=0)&(out<=1)).all());self.assertTrue(np.allclose(out.reshape(57,-1,96).mean(-1),expected,atol=1e-12))
 def test_undefined_metrics_are_null(self):
  r=nlr.metrics(np.zeros((2,96)),np.ones((2,96)))
  self.assertIsNone(r['curve_mape_percent']);self.assertIsNone(r['curve_wape_percent']);self.assertEqual(r['mape_excluded'],192)
if __name__=='__main__':unittest.main()

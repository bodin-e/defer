
import unittest
import numpy as np
from defer.tree_construction.meta_direct import to_64_bit_float_unit_elements

smallest_128 = np.finfo(np.float128).tiny
largest_128 = np.finfo(np.float128).max

class TestVectorScaling(unittest.TestCase):

  def test_maintain_order(self):

    def test_v(v):
      v = np.array(v, dtype=np.float128)
      w = to_64_bit_float_unit_elements(v)
      order_before = np.argsort(v)
      order_after = np.argsort(w)
      assert equal_array(order_before, order_after)
      assert np.all(np.logical_and(w >= 0, w <= 1))
      assert w.dtype == np.float64

    test_v([smallest_128, 2 * smallest_128, 0])
    test_v([smallest_128, 2 * smallest_128, np.inf])
    test_v([smallest_128, 1.1 * smallest_128, np.inf])
    test_v([smallest_128, np.inf, 1.1 * smallest_128])
    test_v(np.random.sample([100]) * largest_128)
    test_v(1 + np.random.sample([100]) * smallest_128)
    test_v(
      np.concatenate([
        np.random.sample([100]) * largest_128,
        1 + np.random.sample([100]) * smallest_128
      ])
    )
    test_v([smallest_128, 2 * smallest_128, 10 * smallest_128])
    test_v([0, smallest_128, 10 * smallest_128])

def equal_array(a, b):
  return np.all(np.abs(a - b) == 0)
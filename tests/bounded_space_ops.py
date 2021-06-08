import unittest

from defer.bounded_space import *

FLOAT_TYPE = np.float64
EPS = 1e-6

class TestBoundedSpace(unittest.TestCase):

  @staticmethod
  def check_attributes(s: BoundedSpace):
    lower = s.lower_limit_vector
    upper = s.upper_limit_vector
    assert np.all(upper + EPS > lower)
    assert_equal_array(upper - lower, s.range_vector)
    assert_equal_float(
      s.volume,
      np.prod(s.range_vector)
    )
    assert_equal_float(
      s.diameter,
      np.sqrt(np.sum(np.square(s.range_vector)))
    )
    assert_equal_array(
      s.center_vector,
      (s.upper_limit_vector + s.lower_limit_vector) / 2
    )

  def test(self):
    self.check_attributes(space([0, 0], [1, 1]))
    self.check_attributes(space([-10 ** 10, -10 ** 10], [10 ** 10, 10 ** 10]))
    self.check_attributes(space([0] * 10000, [1] * 10000))
    lower = np.random.normal(
      loc=0,
      scale=1000,
      size=100
    )
    range = np.random.uniform(1, 10000, size=lower.size)
    upper = lower + range
    self.check_attributes(space(lower, upper, float_type=np.float128))

class TestSubspaceByDims(unittest.TestCase):

  def test(self):
    s = space([0, 0, 0, 0, 0], [1, 2, 3, 4, 5])
    ss = subspace_by_dims(space=s, dims=[0, 2])
    assert np.all(equal_float(ss.lower_limit_vector, [0, 0]))
    assert np.all(equal_float(ss.upper_limit_vector, [1, 3]))

class TestSpaceWithCenter(unittest.TestCase):

  def test(self):
    s = space_with_center(center_vector=point([1, 1, 1]), range_vector=point([1, 2, 3]))
    assert has_bounds(s, lower=[0.5, 0.0, -0.5], upper=[1.5, 2.0, 2.5])

class TestWithin(unittest.TestCase):

  def test_is_point_within(self):
    s = space([0, 0], [1, 1])
    assert is_point_within(space=s, point=point([0, 0]))
    assert is_point_within(space=s, point=point([1, 1]))
    assert is_point_within(space=s, point=point([0.5, 0.5]))
    assert is_point_within(space=s, point=point([0.2, 0.8]))
    assert not is_point_within(space=s, point=point([-1, 1]))
    assert not is_point_within(space=s, point=point([0, -1]))
    num_points = 100
    p = np.random.uniform(0, 1, size=[num_points, 2])
    b = is_point_within(space=s, point=p)
    assert b.shape == (num_points, )
    assert np.all(b)

class TestRelativePoint(unittest.TestCase):

  def test(self):
    space = BoundedSpace(
      lower_limit_vector=np.array([0, 0], dtype=FLOAT_TYPE),
      upper_limit_vector=np.array([5, 10], dtype=FLOAT_TYPE)
    )
    c = np.array([2, 5], dtype=FLOAT_TYPE)
    u = relative_point(space, point=c)
    assert np.all(equal_float(u, [2 / 5, 5 / 10]))
    def test_coordinate(x):
      c = np.array(x, dtype=FLOAT_TYPE)
      u = relative_point(space, c)
      assert np.all(np.less_equal(u, 1))
      assert np.all(np.greater_equal(u, 0))
      c_from_u = absolute_point(space, u)
      assert np.all(equal_float(c, c_from_u))
    test_coordinate([0, 0])
    test_coordinate([5, 10])
    test_coordinate([2, 5])
    test_coordinate([1, 2])

    # Catch outside space
    try:
      test_coordinate([6, 10])
      test_coordinate([-1, 10])
      test_coordinate([2, 11])
      assert False
    except:
      pass

def equal_float(a, b, eps=EPS):
  return np.abs(a - b) < eps

def equal_array(a, b, eps=EPS):
  return np.all(equal_float(a, b, eps=eps))

def assert_equal_float(a, b, eps=EPS):
  assert equal_float(a, b, eps=eps), "%s and %s must be equal." % (a, b)

def assert_equal_array(a, b, eps=EPS):
  assert np.all(equal_float(a, b, eps=eps)), "%s and %s must be equal." % (a, b)

def has_bounds(space, lower, upper):
  return np.logical_and(
    np.all(equal_float(space.lower_limit_vector, lower)),
    np.all(equal_float(space.upper_limit_vector, upper))
  )

def space(lower_limits, upper_limits, float_type=FLOAT_TYPE):
  return BoundedSpace(
    lower_limit_vector=np.array(lower_limits, dtype=float_type),
    upper_limit_vector=np.array(upper_limits, dtype=float_type)
  )

def point(coords):
  return np.array(coords, dtype=FLOAT_TYPE)
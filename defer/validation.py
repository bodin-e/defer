
import numpy as np

def as_numpy_scalar(name, v):
  if np.isscalar(v):
    return v
  # Try making it a scalar
  is_rank_one = np.size(v) == 1
  if is_rank_one:
    v = np.array(v).item()
  if np.isscalar(v):
    return v
  raise ValueError("%s must be a scalar, but was %s" % (name, v))

def check_numpy_dtype(name, array, expected_dtype):
  if array.dtype != expected_dtype:
    raise ValueError(
      "%s must have dtype %s, but had %s." % (
        name, expected_dtype, array.dtype
      ))

def check_shape(name, array, expected_shape):
  if np.shape(array) != expected_shape:
    raise ValueError(
      "Shape of %s must be %s, but was %s." % (
        name, expected_shape, np.shape(array)
      ))

def check_numpy_is_finite_real(name, v):
  check_numpy_is_real(name, v)
  check_numpy_is_finite(name, v)

def check_numpy_is_real(name, v):
  if not np.all(np.isreal(v)):
    raise ValueError("%s must be real valued, but was %s" % (name, v))

def check_numpy_is_finite(name, v):
  if not np.all(np.isfinite(v)):
    raise ValueError("%s must be finite valued, but was %s" % (name, v))

def check_type(name, object, expected_type):
  if isinstance(object, expected_type):
    return
  raise ValueError(
    "%s must be of type %s but was of type %s"
    % (name, expected_type, type(object))
  )
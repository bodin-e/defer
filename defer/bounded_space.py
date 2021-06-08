
import numpy as np
import typing as t
from .validation import check_type, check_shape, check_numpy_dtype, check_numpy_is_finite_real

EPS = 1e-6

class BoundedSpace:

  def __init__(
    self,
    lower_limit_vector: np.ndarray,
    upper_limit_vector: np.ndarray,
  ):
    check_type(
      "lower_limit_vector", lower_limit_vector, np.ndarray)
    check_type(
      "upper_limit_vector", upper_limit_vector, np.ndarray)
    check_shape(
      "lower_limit_vector", lower_limit_vector, upper_limit_vector.shape)
    check_numpy_dtype(
      "lower_limit_vector", lower_limit_vector, upper_limit_vector.dtype)
    check_numpy_is_finite_real("lower_limit_vector", lower_limit_vector)
    check_numpy_is_finite_real("upper_limit_vector", upper_limit_vector)
    if len(lower_limit_vector.shape) != 1:
      raise ValueError(
        "lower_limit_vector must be a vector, but had shape %s."
        % lower_limit_vector.shape)
    if np.size(lower_limit_vector) < 1:
      raise ValueError(
        "lower_limit_vector must have at least on one element, but is has %s."
        % np.size(lower_limit_vector))
    if np.any(upper_limit_vector + EPS <= lower_limit_vector):
      raise ValueError(
        "upper_limit_vector must be greater than lower_limit_vector "
        "in all entries, but was %s and %s."
        % (upper_limit_vector, lower_limit_vector))
    self.ndims = np.size(lower_limit_vector)
    self.dtype = lower_limit_vector.dtype
    self.lower_limit_vector = lower_limit_vector
    self.upper_limit_vector = upper_limit_vector
    self.range_vector = upper_limit_vector - lower_limit_vector
    self.center_vector = (upper_limit_vector + lower_limit_vector) / 2
    self.volume = np.prod(self.range_vector)
    self.diameter = np.sqrt(np.sum(np.square(self.range_vector)))

  def __repr__(self):
    return "BoundedSpace[%s, %s]" % (self.lower_limit_vector, self.upper_limit_vector)

def is_space_within(
  outer: BoundedSpace, inner: BoundedSpace, eps=EPS) -> bool:
  return np.all(is_space_within_per_dim(outer=outer, inner=inner, eps=eps))

def is_space_within_per_dim(
  outer: BoundedSpace, inner: BoundedSpace, eps=EPS) -> np.ndarray:
  return np.logical_and(
    is_point_within_per_dim(space=outer, point=inner.lower_limit_vector, eps=eps),
    is_point_within_per_dim(space=outer, point=inner.upper_limit_vector, eps=eps),
  )

def is_point_within(
  space: BoundedSpace, point: np.ndarray, eps=EPS) -> np.ndarray:
  return np.all(is_point_within_per_dim(space, point, eps=eps), axis=-1)

def is_point_within_per_dim(
  space: BoundedSpace, point, eps=EPS) -> np.ndarray:
  assert space.dtype == point.dtype
  assert space.ndims == point.shape[-1]
  return np.logical_and(
    np.greater_equal(point + eps, space.lower_limit_vector),
    np.greater_equal(space.upper_limit_vector + eps, point)
  )

def are_spaces_equal(
  space_a: BoundedSpace, space_b: BoundedSpace, eps=EPS) -> bool:
  if space_a.ndims != space_b.ndims:
    return False
  if np.any(np.abs(space_a.center_vector - space_b.center_vector) > eps):
    return False
  return np.all(np.abs(space_a.range_vector - space_b.range_vector) <= eps)

def subspace_by_dims(space: BoundedSpace, dims: t.List[int]) -> BoundedSpace:
  return BoundedSpace(
    lower_limit_vector=space.lower_limit_vector[dims],
    upper_limit_vector=space.upper_limit_vector[dims]
  )

def space_with_center(
  center_vector: np.ndarray, range_vector: np.ndarray) -> BoundedSpace:
  return BoundedSpace(
    lower_limit_vector=center_vector - 0.5 * range_vector,
    upper_limit_vector=center_vector + 0.5 * range_vector
  )

def absolute_region(
  space: BoundedSpace, relative_region: BoundedSpace) -> BoundedSpace:
  return BoundedSpace(
    lower_limit_vector=absolute_point(space, relative_region.lower_limit_vector),
    upper_limit_vector=absolute_point(space, relative_region.upper_limit_vector)
  )

def relative_region(
  space: BoundedSpace, region: BoundedSpace) -> BoundedSpace:
  return BoundedSpace(
    lower_limit_vector=relative_point(space, region.lower_limit_vector),
    upper_limit_vector=relative_point(space, region.upper_limit_vector)
  )

def relative_point(
  space: BoundedSpace, point: np.ndarray) -> np.ndarray:
  assert np.all(is_point_within(space, point))
  return (point - space.lower_limit_vector) / space.range_vector

def absolute_point(
  space: BoundedSpace, relative_point: np.ndarray, eps=EPS) -> np.ndarray:
  assert np.all(relative_point + eps >= 0)
  assert np.all(relative_point - eps <= 1)
  return space.lower_limit_vector + relative_point * space.range_vector

def sample_uniform(
  space: BoundedSpace, sample_shape: t.Union[tuple, int]=()) -> np.ndarray:
  sample_shape = np.atleast_1d(sample_shape).tolist()
  return np.random.uniform(
    space.lower_limit_vector,
    space.upper_limit_vector,
    size=sample_shape + [space.ndims]
  )

def has_intersection(
  space_a: BoundedSpace, space_b: BoundedSpace, eps=EPS) -> bool:
  if space_a.ndims != space_b.ndims:
    raise ValueError(
      "space_a and space_b must have the same number of dimensions")

  b_upper_limit_within_a_per_dim = is_point_within_per_dim(
    space=space_a,
    point=space_b.upper_limit_vector,
    eps=eps
  )
  a_upper_limit_within_b_per_dim = is_point_within_per_dim(
    space=space_b,
    point=space_a.upper_limit_vector,
    eps=eps
  )
  has_intersection_per_dim = np.logical_or(
    b_upper_limit_within_a_per_dim, a_upper_limit_within_b_per_dim
  )
  return np.all(has_intersection_per_dim)

def intersection(
  space_a: BoundedSpace, space_b: BoundedSpace, eps=EPS) -> BoundedSpace:
  if not has_intersection(space_a, space_b, eps):
    raise ValueError("space_a and space_b must intersect.")

  # Has 1 in a given element if true, 0 otherwise
  b_dim_is_right_hand_inner = is_point_within_per_dim(
    space=space_a,
    point=space_b.upper_limit_vector,
    eps=eps
  ).astype(space_a.dtype)
  # Inverse
  a_dim_is_right_hand_inner = \
    np.ones_like(b_dim_is_right_hand_inner) - b_dim_is_right_hand_inner

  highest_lower_limit_vector = np.maximum(
    space_a.lower_limit_vector,
    space_b.lower_limit_vector
  )

  lower_limit_vector = highest_lower_limit_vector
  upper_limit_vector = a_dim_is_right_hand_inner * space_a.upper_limit_vector + \
                       b_dim_is_right_hand_inner * space_b.upper_limit_vector
  return BoundedSpace(
    lower_limit_vector=lower_limit_vector,
    upper_limit_vector=upper_limit_vector
  )

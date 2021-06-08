
import numpy as np
import xxhash
from ..tree import find_leaf
from itertools import compress, product

def get_nd_array_hash_function():
  hashing_state = xxhash.xxh64()
  def hash_function(x):
    hashing_state.update(x)
    h = hashing_state.intdigest()
    hashing_state.reset()
    return h
  return hash_function

class Accumulator:

  def __init__(self, include_func, exclude_func, get_current_func):
    self._include_func = include_func
    self._exclude_func = exclude_func
    self._get_current_func = get_current_func

  def include(self, obj):
    return self._include_func(obj)

  def exclude(self, obj):
    return self._exclude_func(obj)

  def current(self):
    return self._get_current_func()

  def __call__(self):
    return self._get_current_func()

def max_k_elements(value_function, k: int):

  eps = 1e-10
  lowest_possible = -np.inf

  assert k > 0
  max_objects = []
  lowest_included_value = lowest_possible
  def include(new_object):
    nonlocal lowest_included_value

    v = value_function(new_object)

    at_capacity = len(max_objects) >= k
    has_high_enough_value = v > lowest_included_value
    update = not at_capacity or has_high_enough_value

    if not at_capacity:
      max_objects.append(new_object)
    elif has_high_enough_value:
      current_values = [value_function(obj) for obj in max_objects]
      index_to_update = int(np.argmin(current_values))
      assert np.abs(current_values[index_to_update] - lowest_included_value) < eps
      max_objects[index_to_update] = new_object

    if update:
      lowest_included_value = np.min([value_function(obj) for obj in max_objects])
    return lowest_included_value

  def exclude(new_object):
    nonlocal lowest_included_value

    # TODO speed up using map if needed
    update = new_object in max_objects

    if update:
      max_objects.remove(new_object)

      if len(max_objects) == 0:
        lowest_included_value = lowest_possible
      else:
        lowest_included_value = np.min([value_function(obj) for obj in max_objects])

    return lowest_included_value

  def current():
    # Return objects as sorted
    order = np.argsort([-value_function(obj) for obj in max_objects])
    return np.array(max_objects)[order]

  return Accumulator(
    include_func=include,
    exclude_func=exclude,
    get_current_func=current
  )

def additive(value_function):
  cum_sum = np.float128(0)
  def include(new_object) -> np.float128:
    nonlocal cum_sum
    v = np.float128(value_function(new_object))
    cum_sum += v
    return cum_sum
  def exclude(new_object) -> np.float128:
    nonlocal cum_sum
    cum_sum -= np.float128(value_function(new_object))
    return cum_sum
  def current() -> np.float128:
    nonlocal cum_sum
    return cum_sum
  return Accumulator(
    include_func=include,
    exclude_func=exclude,
    get_current_func=current
  )

def hyperplane_basis(points):
  # Return orthogonal basis for (k - 1)-dimensional
  # hyperplane embedded in a d-dimensional space
  k, d = points.shape
  assert k >= 2
  s = k - 1

  distances_to_center = np.linalg.norm(points - 0.5, ord=2, axis=1)
  reference_point_index = np.argmin(distances_to_center)
  reference_point = points[reference_point_index]

  other_points = np.delete(points, reference_point_index, axis=0)
  basis = other_points - reference_point
  assert s == basis.shape[0]

  A, _ = np.linalg.qr(
    basis,
    mode='complete'
  )
  basis_norm = np.linalg.norm(basis, ord=2, axis=1, keepdims=True)

  # Avoid division by zero.
  # If norm is zero, all values are zero anyway.
  basis_norm[basis_norm == 0] = 1

  unit_norm_basis = basis / basis_norm
  Ab = np.matmul(A, unit_norm_basis)
  return Ab, reference_point

def combinations(items):
  return (set(compress(items, mask)) for mask in product(*[[0, 1]] * len(items)))

def point_transformation(point):
  reflective = np.random.randint(0, 2) == 0
  if reflective:
    # Cap to [0, 2).
    p_point = point % 2
    assert np.all(p_point < 2) and np.all(p_point >= 0)
    unit_overflow = np.maximum(p_point - 1, 0)
    unit_underflow = np.maximum(0 - p_point, 0)
    correction = unit_underflow - unit_overflow
    p = p_point + 2 * correction
  else:
    # Periodic
    p = point % 1
  assert np.all(p >= 0)
  assert np.all(p <= 1)
  return p

def rects_at_points(
  points,
  array_hash_function,
  root
):
  '''
  Find the set of unique partitions at the specified points.
  :param points:
  :return:
  '''
  points = np.array(points)

  def unique_vector_indices(vectors):
    unique_vectors_by_hash = {
      # Will save the last index occurrence of a given hash
      array_hash_function(v): index
      for index, v in enumerate(vectors)
    }
    return [
      index
      for _, index in unique_vectors_by_hash.items()
    ]

  unique_points = points[unique_vector_indices(points)]
  leafs = np.array([
    find_leaf(root, point)
    for point in unique_points
  ])
  unique_leafs = leafs[unique_vector_indices(
    [leaf.domain.center_vector for leaf in leafs]
  )]
  return unique_leafs

def unit_points_in_neighbourhood(
  partial_unit_domain,
  phi,
  num_representer_pts,
):
  assert phi > 1

  rect_center = partial_unit_domain.center_vector

  D = partial_unit_domain.ndims
  rect_diameter = partial_unit_domain.diameter
  rect_radius = 0.5 * rect_diameter
  ball_radius = phi * rect_radius

  unit_points = []
  for _ in range(num_representer_pts):
    uniform_radius = np.random.uniform(0, 1)
    gaussian_u = np.random.normal(0, 1, D)
    norm = np.sum(gaussian_u ** 2) ** 0.5
    r = ball_radius * uniform_radius ** (1.0 / D)
    point = rect_center + r * gaussian_u / norm

    transformed_point = point_transformation(point)
    unit_points.append(transformed_point)

  if len(unit_points) == 0:
    return np.zeros([0, D])
  return np.array(unit_points)

def linear_correlation_exploitation_unit_points(
  h_rects,
  domain,
  index_combinations_descending_set_size,
  linear_manifold_num_dims,
  num_representer_pts_linear_spread,
):
  if len(h_rects) <= 1:
    return np.zeros([0, domain.ndims])

  unit_centroids = np.array([
    (rect.domain.center_vector - domain.lower_limit_vector)
    / domain.range_vector
    for rect in h_rects
  ])

  num_rects = len(h_rects)

  valid_index_combinations_in_descending_set_size = [
    indices
    for indices in index_combinations_descending_set_size
    # Check if all indices are within bounds of rects
    if not np.any(indices >= num_rects)
  ]

  unit_points = []

  for indices in valid_index_combinations_in_descending_set_size:
    unit_center_vectors = unit_centroids[indices]
    assert np.all(unit_center_vectors >= 0)
    assert np.all(unit_center_vectors <= 1)

    # Basis for l-dimensional hyperplane, where l = num_pts - 1
    Ab, ref_point = hyperplane_basis(points=unit_center_vectors)

    intended_hyperplane_rank = len(indices) - 1
    actual_hyperplane_rank = np.linalg.matrix_rank(Ab)
    assert actual_hyperplane_rank <= intended_hyperplane_rank

    if actual_hyperplane_rank < intended_hyperplane_rank:
      # This means points are collinear.
      # We skip this set of points,
      # as the 'effective' hyperplane of the actually non-collinear points
      # will be considered by a lower dimensional hyperplane.
      continue

    l = len(unit_center_vectors) - 1
    assert Ab.shape[0] == l
    assert l <= linear_manifold_num_dims

    # Exploration of hyperplane #################
    for _ in range(num_representer_pts_linear_spread):
      sample = np.random.uniform(-1, 1, size=l)
      # Will be wrapped around the unit domain
      U = sample.reshape([1, -1])
      UAb = np.matmul(U, Ab)
      mapped_sample = (ref_point + UAb).flatten()
      transformed_sample = point_transformation(mapped_sample)
      unit_points.append(transformed_sample)
    ##################################################

    # "Weighted" center of the points ################
    point_in_center_of_subset = np.mean(unit_center_vectors, axis=0)
    unit_points.append(point_in_center_of_subset)
    ###################################################

  assert len(unit_points) > 0
  return np.array(unit_points)

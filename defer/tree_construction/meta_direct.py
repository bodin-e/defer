
import numpy as np
import typing as t
import heapq
from scipy.spatial import ConvexHull
from ..tree import MassTree

class MetaDIRECT:

  def __init__(
    self,
    x_func,
    y_func,
    filter_func=None,
  ):
    self.x_func = x_func
    self.y_func = y_func
    self.sorted_leaf_rects_by_hash = HashMapOfHeaps(
      hash_function=lambda rect: float_hash(x_func(rect)),
      # Heap sorted by lowest so y_func is inverted
      value_function=lambda rect: -y_func(rect)
    )
    self.filter_func = filter_func

  def exclude_leaf(self, rect):
    assert rect.has_children
    rect._deleted = True
    if self.sorted_leaf_rects_by_hash.is_a_root(rect):
      self.sorted_leaf_rects_by_hash.delete_root(rect)

  def include_leaf(self, rect):
    assert not rect.has_children
    rect._deleted = False
    self.sorted_leaf_rects_by_hash.push(rect)

  def potentially_optimal(self, visualize=False):
    # Note that highest corresponds to lowest on the heaps as of their inverted y_func
    highest_y_per_unique_x = \
      self.sorted_leaf_rects_by_hash.lowest_value_object_per_hash()
    return meta_direct_criteria(
      highest_y_per_unique_x=highest_y_per_unique_x,
      x_func=self.x_func,
      y_func=self.y_func,
      filter_func=self.filter_func,
      visualize=visualize
    )

def relative_scale(v):
  r = np.max(v)
  denom = 1 if r == 0 else r
  return v / denom

def meta_direct_criteria(
  highest_y_per_unique_x: t.Sequence[MassTree],
  x_func,
  y_func,
  filter_func=None,
  visualize=False
):
  # Only the highest y rect per unique x can be on the upper right hull
  potentially_on_upper_right_hull = highest_y_per_unique_x

  xs_func = lambda rects: np.array([x_func(rect) for rect in rects])
  ys_func = lambda rects: np.array([y_func(rect) for rect in rects])

  rect_order = np.argsort(xs_func(potentially_on_upper_right_hull))
  rects = potentially_on_upper_right_hull[rect_order]

  indices = indices_on_lower_right_hull_from_left(
      x=xs_func(rects),
      # Negate y: upper right -> lower right
      y=-ys_func(rects)
    )
  on_upper_right_hull_from_left = rects[indices]

  if filter_func is not None:
    y_current = ys_func(on_upper_right_hull_from_left)
    y_upper = y_upper_bounds(
      d_non_decreasing=xs_func(on_upper_right_hull_from_left),
      f_non_increasing=y_current,
    )
    # delta_y = np.abs(y_upper - y_current)
    non_trivially_improving = on_upper_right_hull_from_left[filter_func(y_upper)]
  else:
    non_trivially_improving = on_upper_right_hull_from_left

  if visualize:
    l1 = len(potentially_on_upper_right_hull)
    l2 = len(on_upper_right_hull_from_left)
    l3 = len(non_trivially_improving)
    print("rectangle filtering: %s >= %s >= %s" % (l1, l2, l3))

  return on_upper_right_hull_from_left

def y_upper_bounds(d_non_decreasing, f_non_increasing):
  k_upper_bounds = lowest_y_k_bounds(
    x_non_decreasing=d_non_decreasing,
    y_non_increasing=f_non_increasing,
    upper=True,
  )
  assert is_monotonic(k_upper_bounds)
  return f_non_increasing + k_upper_bounds * d_non_decreasing

def lowest_y_k_bounds(
  x_non_decreasing,
  y_non_increasing,
  upper: bool,
  jitter=1e-50,
):
  """
  Returns the upper (or lower) bounds of the rate-of-change
  that would make a given element have (among) the lowest y of all.
  :param x_non_decreasing:
  :param y_non_decreasing:
  :param upper: bool
  :param jitter:
  :return: k_upper_bounds if upper=True, else k_lower_bounds
  """
  def check(array, name):
    if len(array.shape) != 1:
      raise ValueError("%s need to be a vector" % name)
    if len(array) == 0:
      raise ValueError("%s need to have at least one element" % name)
    if not is_monotonic(array):
      raise ValueError(
        "%s need to be monotonically increasing or constant" % name)
  check(x_non_decreasing, "x_non_decreasing")
  check(-y_non_increasing, "y_increasing")
  if len(x_non_decreasing) != len(y_non_increasing):
    raise ValueError(
      "Need to be the same number of x_non_decreasing and y_non_increasing")

  if len(y_non_increasing) == 1:
    if upper:
      return np.array([np.inf])
    return np.array([0])

  x = x_non_decreasing.astype(np.float128)

  # k^{lower}_{j} = (v_{j} - v_{j-1}) / (d_{j-1} - d_{j})
  # k^{upper}_{j} = (v_{j} - v_{j+1}) / (d_{j+1} - d_{j})
  y = y_non_increasing
  k = (y[:-1] - y[1:]) / (x[1:] - x[:-1] + jitter)
  if upper:
    return np.append(k, np.inf)
  return np.insert(k, 0, 0)

def is_monotonic(points):
  return np.all(points[1:] - points[:-1] >= 0)

class HashMapOfHeaps:

  ROOT_INDEX = 0
  OBJECT_INDEX = 2

  def __init__(self, hash_function, value_function):
    self._hash_function = hash_function
    self._value_function = value_function
    self._map = {}
    self._id_counter = 0

  def push(self, obj):
    hash = self._hash_function(obj)
    value = self._value_function(obj)
    ## Add heap to map if not exists
    if hash not in self._map:
      self._map[hash] = []
    ## Add element to its heap
    heap = self._map[hash]
    # Heap is sorted (ascending) based on value
    # TODO consider all of equal value instead
    # - if value is not unique, an unique (arbitrary) id will be used
    item = (value, self._id_counter, obj)
    heapq.heappush(heap, item)
    self._id_counter += 1

  def is_a_root(self, obj):
    hash = self._hash_function(obj)
    heap = self._map[hash]
    current_root_obj = heap[self.ROOT_INDEX][self.OBJECT_INDEX]
    return id(obj) == id(current_root_obj)

  def delete_root(self, root_obj):
    hash = self._hash_function(root_obj)
    heap = self._map[hash]
    current_root_obj = heap[self.ROOT_INDEX][self.OBJECT_INDEX]
    if id(root_obj) != id(current_root_obj):
      raise ValueError("Object to delete must currently be the root")
    heapq.heappop(heap)
    # The heap may now be empty. If so, delete it from map
    if len(heap) == 0:
      del self._map[hash]

  def lowest_value_object_per_hash(self):
    heaps = list(self._map.values())
    roots = []
    for heap in heaps:
      suggested_root = heap[self.ROOT_INDEX][self.OBJECT_INDEX]
      while (
        # Has been marked for deletion (to be ignored -> so move on to the next)
        suggested_root._deleted and
        # There is a next element available
        len(heap) > 1
      ):
        heapq.heappop(heap) # Remove current (deleted) root
        suggested_root = heap[self.ROOT_INDEX][self.OBJECT_INDEX]
      # The last element might have been deleted as well
      # - if it has been, then the heap is to be considered empty and nothing is to be added
      root_found = not suggested_root._deleted
      if root_found:
        roots.append(suggested_root)
      else:
        assert len(heap) == 1
        dead_root = heap[self.ROOT_INDEX][self.OBJECT_INDEX]
        assert dead_root._deleted
        # The heap is now empty, delete it from map
        del self._map[self._hash_function(dead_root)]
    return np.array(roots)

largest_128_float = np.finfo(np.float128).max

def to_64_bit_float_unit_elements(v):
  if len(v.shape) != 1:
    raise ValueError("v needs to be a vector.")
  if not np.logical_or(
    np.all(v >= 0),
    np.all(v <= 0)
  ):
    raise ValueError("Either all elements of v must be non-negative, or all non-positive.")
  if v.dtype != np.float128:
    raise ValueError("v must be a 128-bit float.")
  # Avoid getting sign of zero, which can be negative.
  sign = np.sign(v[np.argmax(np.abs(v))])

  def unit_scale(non_neg):
    assert np.all(non_neg >= 0)
    non_neg[np.isinf(non_neg)] = largest_128_float

    d = np.max(non_neg)
    non_neg /= (1 if d == 0 else d)

    range = np.max(non_neg) - np.min(non_neg)
    if range == 0:
      return non_neg
    return (non_neg - np.min(non_neg)) / (1 if range == 0 else range)

  positive = sign > 0
  u = unit_scale(np.abs(v))
  return np.float64(u if positive else 1 - u)

def indices_on_lower_right_hull_from_left(x, y):
  """
  Returns the indices of the (x, y)-points on the lower right of the convex hull,
  in order from left to right.
  :param x. x-coordinates. shape [N]
  :param y. y-coordinates. shape [N]
  :return: indices. shape [M], M <= N
  """
  def check(array, name):
    if not np.issubdtype(array.dtype, np.floating):
      raise ValueError("%s needs to be of float type." % name)
    if len(array.shape) != 1:
      raise ValueError("%s needs to be a vector." % name)
  check(x, "x")
  check(y, "y")
  if len(x) != len(y):
    raise ValueError("x and y need to have the same length.")

  num_points = len(x)

  # For the empty set the hull is also empty
  if num_points == 0:
    return np.array([], dtype=np.int64)
  # One point is always on the lower right of the hull
  if num_points == 1:
    return np.array([0], dtype=np.int64)
  elif num_points == 2:
    left_point_index = np.argmin(x)
    lower_point_index = np.argmin(y)
    other_index = 1 - lower_point_index
    # If the lower point is on the left
    # the other point is also on the lower right of the hull
    if lower_point_index == left_point_index:
      return np.array([lower_point_index, other_index], dtype=np.int64)
    else:
      return np.array([lower_point_index], dtype=np.int64)

  x = x
  # Re-scale to fit into 64 bit for hull check.
  y = to_64_bit_float_unit_elements(y)

  def is_flat(v):
    all_equal = np.all(np.abs(v[0] - v) == 0)
    return all_equal

  if is_flat(y):
    return np.argsort(x)

  if is_flat(x):
    return [np.argmin(y)]

  points = np.stack([x, y], axis=-1)
  hull = ConvexHull(points, qhull_options="")

  # Vertices are sorted counter-clockwise
  vertex_indices = hull.vertices
  vertices_x = x[vertex_indices]
  vertices_y = y[vertex_indices]
  vertex_indices_indices = np.arange(0, len(vertex_indices), dtype=np.int64)

  x_max_index_in_vertex_indices = np.argmax(vertices_x)
  y_min_index_in_vertex_indices = np.argmin(vertices_y)

  def get(vector, or_conditions):
    condition_vector = np.any(np.stack(or_conditions, axis=0), axis=0)
    dims = np.where(condition_vector)[0]
    return np.array(vector[dims])

  lower_right_vertex_indices = get(
    vector=vertex_indices,
    or_conditions=[
      # being on right side i.e. right of lowest hull point
      vertices_x >= vertices_x[y_min_index_in_vertex_indices],
      # being on lower side i.e. lower than right-most hull point
      vertices_y <= vertices_y[x_max_index_in_vertex_indices],
      ## always include boundaries (avoid relying on float equality)
      vertex_indices_indices == x_max_index_in_vertex_indices,
      vertex_indices_indices == y_min_index_in_vertex_indices
    ]
  )

  # Edges should always be part of hull
  assert np.abs(np.min(x) - np.min(vertices_x)) <= 1e-10
  assert np.abs(np.max(x) - np.max(vertices_x)) <= 1e-10
  assert np.abs(np.min(y) - np.min(vertices_y)) <= 1e-10
  assert np.abs(np.max(y) - np.max(vertices_y)) <= 1e-10

  lower_right_vertices_x = x[lower_right_vertex_indices]
  lower_right_vertices_y = y[lower_right_vertex_indices]

  x_max_index_in_lower_right_vertex_indices = np.argmax(lower_right_vertices_x)
  y_min_index_in_lower_right_vertex_indices = np.argmin(lower_right_vertices_y)

  begin = int(y_min_index_in_lower_right_vertex_indices)

  end_exclusive = int(x_max_index_in_lower_right_vertex_indices)
  end_inclusive = end_exclusive + 1
  # Lower right vertices are still sorted counter-clockwise
  # Wrap around as vertices are cyclic (begin may come after end)
  if begin <= end_exclusive:
    lower_right_vertex_indices_from_left = lower_right_vertex_indices[
      np.arange(begin, end_inclusive)
    ]
  else:
    lower_right_vertex_indices_from_left = np.concatenate(
      [
        lower_right_vertex_indices[
          np.arange(begin, len(lower_right_vertex_indices))
        ],
        lower_right_vertex_indices[
          np.arange(0, end_inclusive)
        ]
      ],
      axis=0
    )
  return lower_right_vertex_indices_from_left

def float_hash(v, num_significant_decimal_places=8):
  return hash(round(v, num_significant_decimal_places))

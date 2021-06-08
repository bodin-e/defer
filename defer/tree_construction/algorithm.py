
import typing as t
from .divide_operator import relatively_longest_dims, child_partition_domains, set_children, determine_side_children_center_points
from ..bounded_space import absolute_point, relative_region
from ..tree import find_all_leaves, MassTree, BoundedSpace
from .utils import *
from .meta_direct import MetaDIRECT

class DEFER:

  def __init__(
    self,
    root: MassTree,
    density_func,
    alpha=20,
    beta=1,
    phi=1.2,
    max_num_high_mass_partitions=None,
    num_representer_pts_balls=None,
    num_representer_pts_linear_spread_per_subspace=1,
    is_vectorized_density_func=False
  ):
    assert 1 < phi
    assert 1 < alpha
    assert 0 < beta
    assert 0 <= num_representer_pts_linear_spread_per_subspace
    domain = root.domain
    D = domain.ndims

    if max_num_high_mass_partitions is None:
      max_num_high_mass_partitions = min(5, D)
    else:
      assert max_num_high_mass_partitions >= 1
    if num_representer_pts_balls is None:
      num_representer_pts_balls = D
    else:
      assert num_representer_pts_balls >= 0

    linear_manifold_num_dims = max_num_high_mass_partitions - 1
    assert 0 <= linear_manifold_num_dims < D

    self._output_total_num_division_made_per_dim = \
      np.zeros([domain.ndims], dtype=np.int64)

    self.domain = domain
    self.is_vectorized_density_func = is_vectorized_density_func
    self.func = density_func
    self.leaves = []

    self.alpha = alpha
    self.beta = beta
    self.phi = phi

    self._mass_accumulator_type = additive

    self.total_unit_mass = self._mass_accumulator_type(
      value_function=lambda rect:
      rect.f *
      np.float128(np.prod(rect.domain.range_vector / domain.range_vector))
    )
    self.total_mass = self._mass_accumulator_type(
      value_function=lambda rect: rect.z
    )
    self.linear_manifold_num_dims = linear_manifold_num_dims
    self.K = max_num_high_mass_partitions
    self.num_representer_pts_linear_spread = \
      num_representer_pts_linear_spread_per_subspace
    self.num_representer_pts_balls = num_representer_pts_balls
    self.max_k_mass_rects = max_k_elements(
      value_function=lambda rect:
      rect.f *
      np.prod(np.float128(rect.domain.range_vector / domain.range_vector)),
      # Need S + 1 points to represent a S-dimensional hyperplane
      k=max_num_high_mass_partitions
    )
    self.array_hash_function = get_nd_array_hash_function()

    all_index_combinations = list(combinations(range(max_num_high_mass_partitions)))
    all_sub_manifold_index_combinations = np.array([
      np.array(list(combination))
      for combination in all_index_combinations
      if 2 <= len(combination) <= max_num_high_mass_partitions
    ])

    index_combinations_descending_size = all_sub_manifold_index_combinations[
      np.argsort([-len(c) for c in all_sub_manifold_index_combinations])
    ]
    self._index_combinations_descending_set_size = \
      index_combinations_descending_size

    def unit_x_func(rect: MassTree):
      relative_range_vector = rect.domain.range_vector / domain.range_vector
      diameter = np.sqrt(np.sum(np.square(relative_range_vector)))
      volume = np.prod(relative_range_vector)
      return float(0.5 * diameter * volume)

    def unit_y_func(rect):
      f = rect.f
      relative_range_vector = rect.domain.range_vector / domain.range_vector
      volume = np.prod(relative_range_vector)
      v = np.abs(f) * np.float128(volume)
      assert np.isfinite(v)
      assert isinstance(v, np.float128)
      return v

    def average_mass():
      return self.total_unit_mass.current() / (self._num_leaves + 1)

    self.direct = MetaDIRECT(
      x_func=unit_x_func,
      y_func=unit_y_func,
      filter_func=lambda y_upper: y_upper / average_mass() >= beta,
    )

    self._num_leaves = 0

    def exclude_leaf(rect):
      assert rect.has_children
      self._num_leaves -= 1
      self.total_mass.exclude(rect)
      self.total_unit_mass.exclude(rect)
      self.max_k_mass_rects.exclude(rect)
      self.direct.exclude_leaf(rect)

    def include_leaf(rect):
      assert not rect.has_children
      self._num_leaves += 1
      self.total_mass.include(rect)
      self.total_unit_mass.include(rect)
      self.max_k_mass_rects.include(rect)
      self.direct.include_leaf(rect)

    self.exclude_leaf = exclude_leaf
    self.include_leaf = include_leaf

    # Loading leaves to prepare for active sampling
    for leaf in find_all_leaves(root):
      include_leaf(rect=leaf)

    self._root = root

  def __len__(self):
    return int(self._num_leaves)

  @property
  def root(self):
    return self._root

  def __iter__(self):
    return self

  def __call__(self, num_evaluations):
    num_partitions_before = len(self)
    self._max_num_evaluations = num_partitions_before + num_evaluations
    return iter(self)

  def __next__(self):
    if len(self) < self._max_num_evaluations:
      return self.single_iteration()
    raise StopIteration

  def single_iteration(self):
    rects_to_divide = set()

    # Sufficient criterion 1
    pot_optimal = self.direct.potentially_optimal()
    rects_to_divide.update(pot_optimal)

    high_mass_partitions = _high_mass_partitions(
      k_largest_mass_partitions=self.max_k_mass_rects.current(),
      average_mass=self.total_mass.current() / (self._num_leaves + 1),
      alpha=self.alpha
    )

    # Sufficient criterion 2
    if len(high_mass_partitions) > 0:
      unit_points = linear_correlation_exploitation_unit_points(
        h_rects=high_mass_partitions,
        domain=self.domain,
        index_combinations_descending_set_size=
        self._index_combinations_descending_set_size,
        linear_manifold_num_dims=self.linear_manifold_num_dims,
        num_representer_pts_linear_spread=
        self.num_representer_pts_linear_spread,
      )
      absolute_points = absolute_point(
          space=self.domain,
          relative_point=unit_points
        )
      rects = rects_at_points(
        points=absolute_points,
        array_hash_function=self.array_hash_function,
        root=self.root
      )
      rects_to_divide.update(rects)

    # Sufficient criterion 3
    for h in high_mass_partitions:
      unit_points = unit_points_in_neighbourhood(
        partial_unit_domain=relative_region(self.domain, h.domain),
        phi=self.phi,
        num_representer_pts=self.num_representer_pts_balls,
      )
      rects = rects_at_points(
        points=absolute_point(
          space=self.domain,
          relative_point=unit_points
        ),
        array_hash_function=self.array_hash_function,
        root=self.root
      )
      rects_to_divide.update(rects)

    # Make divisions ##################################

    rects_to_divide_in_order = list(rects_to_divide)

    side_children_center_points, array_index_per_dim_per_partition \
      = _make_side_children_center_points(
      partitions_to_divide=rects_to_divide_in_order,
      domain=self.domain
    )
    side_children_density_values = _evaluate_density_fn(
      points=np.array(side_children_center_points),
      fn=self.func,
      is_vectorized_fn=self.is_vectorized_density_func,
      num_dims=self.domain.ndims
    )

    for i, partition in enumerate(rects_to_divide_in_order):
      array_index_per_dim = array_index_per_dim_per_partition[i]
      density_values_per_dim = {
        dim: (
          side_children_density_values[left_index],
          side_children_density_values[right_index]
        )
        for dim, (left_index, right_index) in array_index_per_dim.items()
      }

      children = _make_child_partitions(
        partition=partition,
        density_values_per_dim=density_values_per_dim,
      )
      set_children(
        parent=partition,
        new_children=children,
        exclude_leaf_func=self.exclude_leaf,
        include_leaf_func=self.include_leaf
      )

    return len(self)

def _make_side_children_center_points(
  partitions_to_divide: t.List[MassTree],
  domain: BoundedSpace,
):
  center_points = []
  index_per_dim_per_partition = []
  for partition in partitions_to_divide:
    longest_dims = relatively_longest_dims(
      partition.domain,
      root_domain=domain
    )
    obj = {}
    for dim_to_divide_along in longest_dims:
      left_center_vector, right_center_vector = determine_side_children_center_points(
        parent_domain=partition.domain,
        dim_to_divide_along=dim_to_divide_along
      )
      current_index = len(center_points)
      center_points.append(left_center_vector)
      center_points.append(right_center_vector)
      # Points to unique indices in the array.
      obj[dim_to_divide_along] = (current_index, current_index + 1)
    index_per_dim_per_partition.append(obj)
  return center_points, index_per_dim_per_partition

def _make_child_partitions(
  partition: MassTree,
  density_values_per_dim,
):
  # Let dim divide order be by highest value
  unsorted_dims = np.array(list(density_values_per_dim.keys()))
  highest_value_per_dim = np.array(
    [np.max(values) for values in density_values_per_dim.values()]
  )
  dim_divide_order = np.argsort(-highest_value_per_dim)
  sorted_dims = unsorted_dims[dim_divide_order]

  children = []
  current_center_domain = partition.domain
  for dim in sorted_dims:
    domain_centre, (domain_left, domain_right) = child_partition_domains(
      parent_domain=current_center_domain,
      dim_to_divide_along=dim
    )
    left_f, right_f = density_values_per_dim[dim]
    children.append(
      MassTree(
        domain=domain_left,
        f=left_f,
        parent=partition
      )
    )
    children.append(
      MassTree(
        domain=domain_right,
        f=right_f,
        parent=partition
      )
    )
    # Keep splitting current center
    current_center_domain = domain_centre

  center_child = MassTree(
    domain=current_center_domain,
    f=partition.f,
    parent=partition
  )
  children.append(center_child)
  return children

def _evaluate_density_fn(
  points,
  fn,
  is_vectorized_fn,
  num_dims: int
):
  assert len(points.shape) == 2
  assert points.shape[1] == num_dims
  num_points = points.shape[0]
  if is_vectorized_fn:
    evaluations_vector = np.array(fn(points))
  else:
    evaluations_vector = np.array([fn(p) for p in points])
  assert evaluations_vector.shape == (num_points,)
  return evaluations_vector

def _high_mass_partitions(k_largest_mass_partitions, average_mass, alpha):
  k_largest_mass_partitions = np.array(k_largest_mass_partitions)
  has_large_enough_mass = [
    p.z / average_mass >= alpha
    for p in k_largest_mass_partitions
  ]
  return k_largest_mass_partitions[has_large_enough_mass]
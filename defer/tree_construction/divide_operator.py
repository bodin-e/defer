
import numpy as np
import typing as t
from ..bounded_space import BoundedSpace, is_space_within
from ..bounded_space import space_with_center
from ..tree import MassTree

def set_children(
  parent: MassTree,
  new_children: t.List[MassTree],
  exclude_leaf_func,
  include_leaf_func,
):
  for child in new_children:
    within_bounds = is_space_within(outer=parent.domain, inner=child.domain)
    assert within_bounds, "%s not inside %s" % (child.domain, parent.domain)

  # Store spawned children in parent to form a tree
  if parent.has_children:
    raise RuntimeError("Parent cannot already have children.")
  parent.children = new_children

  # Must run before children are included
  exclude_leaf_func(rect=parent)

  for child in new_children:
    include_leaf_func(rect=child)

  update_leaf_count_and_propagate_upwards(recent_parent=parent)

  new_z = np.sum([child.f * child.domain.volume for child in new_children])
  assert isinstance(new_z, np.float128)
  update_z_and_propagate_upwards(
    recent_parent=parent,
    new_z=new_z
  )

def determine_side_children_center_points(
  parent_domain: BoundedSpace,
  dim_to_divide_along: int
):
  dim = dim_to_divide_along

  ## Determine the offset to the centres and ranges
  delta = parent_domain.range_vector[dim] / 3

  e_i = one_hot(parent_domain.ndims, dim=dim, dtype=parent_domain.dtype)
  delta_vector = delta * e_i
  left_center_vector = parent_domain.center_vector - delta_vector
  right_center_vector = parent_domain.center_vector + delta_vector

  return left_center_vector, right_center_vector

def child_partition_domains(
  parent_domain: BoundedSpace,
  dim_to_divide_along: int
):
  dim = dim_to_divide_along

  ## Determine the offset to the centres and ranges
  delta = parent_domain.range_vector[dim] / 3

  e_i = one_hot(parent_domain.ndims, dim=dim, dtype=parent_domain.dtype)
  delta_vector = delta * e_i

  child_range_vector = parent_domain.range_vector - 2 * delta_vector

  def child_domain(center_vector):
    return space_with_center(
      center_vector=center_vector,
      range_vector=child_range_vector)

  centre_child = child_domain(center_vector=parent_domain.center_vector)
  return centre_child, [
    # Left child
    child_domain(center_vector=parent_domain.center_vector - delta_vector),
    # Right child
    child_domain(center_vector=parent_domain.center_vector + delta_vector),
  ]

def relatively_longest_dims(
  domain: BoundedSpace,
  root_domain: BoundedSpace,
):
  relative_range_vector = domain.range_vector / root_domain.range_vector
  max_dimension_relative_range = np.max(relative_range_vector)
  dim_has_highest_relative_range = _equal_within_epsilon(
    relative_range_vector,
    max_dimension_relative_range
  )
  return np.reshape(np.where(dim_has_highest_relative_range), [-1])

def one_hot(num_dims: int, dim: int, dtype):
  e_i = np.zeros([num_dims], dtype=dtype)
  e_i[dim] = 1
  return e_i

def _equal_within_epsilon(a, b, eps=1e-8):
  return np.less_equal(np.abs(a - b), eps)

def update_z_and_propagate_upwards(
  recent_parent,
  new_z: np.float128
):
  old_z = recent_parent.z
  recent_parent.f = new_z / recent_parent.domain.volume
  delta_z = new_z - old_z
  parent = recent_parent.parent
  while parent is not None:
    parent.f += delta_z / parent.domain.volume
    parent = parent.parent

def update_leaf_count_and_propagate_upwards(recent_parent):
  num_leaves = len(recent_parent.children)
  recent_parent.num_partitions = num_leaves
  # Only #children - 1 new leaves were created,
  # as the recent parent is no longer a leaf itself.
  increase_of_leaves = num_leaves - 1
  parent = recent_parent.parent
  while parent is not None:
    parent.num_partitions += increase_of_leaves
    parent = parent.parent
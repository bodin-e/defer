
import typing as t
import numpy as np
from .bounded_space import BoundedSpace, is_point_within, intersection, subspace_by_dims, has_intersection

class MassTree:

  def __init__(
    self,
    domain: BoundedSpace,
    f: np.float128,
    parent=None
  ):
    self.domain: BoundedSpace = domain
    self.parent = parent

    # Only a leaf keep the observed f value at the centroid.
    # All parents will instead have the average f throughout its domain.
    # Will be updated every time the mass of the subtree changes
    self.f: np.float128 = f

    # Will be updated every time its subtree obtains a new leaf
    self.num_partitions = 0  # Not including self

    # Will be updated when it gets children
    self.children: t.List[MassTree] = None

  @property
  def z(self):
    return self.f * self.domain.volume

  @property
  def has_parent(self):
    return self.parent is not None

  @property
  def has_children(self):
    return self.children is not None and len(self.children) > 0

def copy(tree):
  cloned = MassTree(
    domain=tree.domain,
    f=tree.f,
    parent=tree.parent
  )
  if not tree.has_children:
    return cloned
  cloned.children = [
    copy(child)
    for child in tree.children
  ]
  return cloned

def sparsify(tree: MassTree, relative_z):
  """
  Prunes all subtrees that have a mass (z) less than a threshold.
  Pruning means removing its children, making it a (uniform) leaf.
  :param tree: tree to be pruned (modifies the tree).
  :param relative_z: determines the threshold z as threshold = relative_z * z.
  """
  assert 0 < relative_z < 1
  threshold_z = relative_z * tree.z

  def inner(subtree: MassTree):
    is_leaf = not subtree.has_children
    if is_leaf:
      # Is already a leaf and can thus not be pruned.
      return
    should_keep = subtree.z >= threshold_z
    if not should_keep:
      subtree.children = None
      return
    for child in subtree.children:
      inner(child)

  inner(subtree=tree)
  set_z_in_subtree(tree)
  set_num_partitions_in_subtree(tree)

def copy_intersection(
  tree,
  domain,
  space_projection=lambda s: s,
):
  assert has_intersection(tree.domain, domain)
  cloned = copy_intersection_inner(
    tree_to_be_cloned=tree,
    new_parent=None,
    domain=domain,
    space_projection=space_projection
  )
  set_z_in_subtree(cloned)
  set_num_partitions_in_subtree(cloned)
  # TODO Make sure clone covers whole domain.
  return cloned

def copy_intersection_inner(
  tree_to_be_cloned,
  new_parent,
  domain,
  space_projection):
  new_clone = MassTree(
    domain=space_projection(
      intersection(tree_to_be_cloned.domain, domain)),
    f=tree_to_be_cloned.f,
    parent=new_parent
  )
  if not tree_to_be_cloned.has_children:
    return new_clone
  children = []
  for child in tree_to_be_cloned.children:
    if not has_intersection(child.domain, domain):
      continue
    children.append(copy_intersection_inner(
      tree_to_be_cloned=child,
      new_parent=new_clone,
      domain=domain,
      space_projection=space_projection
    ))
  new_clone.children = children
  return new_clone

def conditional_tree(
  tree,
  cond_dims,
  cond_values
):
  assert len(cond_dims) == len(cond_values)
  # Enforce dims to be sorted
  dim_order = np.argsort(cond_dims)
  cond_dims = np.array(cond_dims)[dim_order]
  cond_values = np.array(cond_values)[dim_order]
  lower = tree.domain.lower_limit_vector.copy()
  upper = tree.domain.upper_limit_vector.copy()
  lower[cond_dims] = cond_values
  upper[cond_dims] = cond_values
  cond_domain = BoundedSpace(
    lower_limit_vector=lower,
    upper_limit_vector=upper
  )
  assert has_intersection(tree.domain, cond_domain)
  remaining_dims = sorted(list(set(range(tree.domain.ndims)) - set(cond_dims)))
  return copy_intersection(
    tree,
    domain=cond_domain,
    space_projection=lambda s: subspace_by_dims(s, dims=remaining_dims)
  )

def find_all_leaves(tree):
  def inner(tree, leaves_output: t.List):
    has_children = tree.has_children
    if not has_children:
      leaves_output.append(tree)
      return
    for child in tree.children:
      inner(
        tree=child,
        leaves_output=leaves_output
      )
  leaves = []
  inner(tree=tree, leaves_output=leaves)
  return np.array(leaves)

def find_leaf(tree, sample):
  if not is_point_within(tree.domain, sample):
    raise ValueError(
      "The passed sample needs to be within "
      "the domain of the parent."
    )
  if not tree.has_children:
    # No children, so this must be a leaf
    return tree
  for child in tree.children:
    if is_point_within(child.domain, sample):
      return find_leaf(tree=child, sample=sample)
  # There should always be a leaf for every sample
  assert False

def set_z_in_subtree(root):
  is_leaf = not root.has_children
  if is_leaf:
    z = root.f * root.domain.volume
  else:
    z = np.sum([
      set_z_in_subtree(child)
      for child in root.children
    ])
  root.f = z / root.domain.volume
  return z

def set_num_partitions_in_subtree(root):
  is_leaf = not root.has_children
  num = 0
  if is_leaf:
    root.num_partitions = num
    return num
  for child in root.children:
    is_leaf_child = not child.has_children
    if is_leaf_child:
      num += 1
    else:
      num += set_num_partitions_in_subtree(child)
  root.num_partitions = num
  return num
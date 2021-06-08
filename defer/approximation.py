
import typing as t
import numpy as np
import pickle
from .tree import MassTree, copy_intersection, find_leaf, find_all_leaves, conditional_tree, copy, sparsify
from .sampling import sampler_in_domains
from .validation import check_type
from .variables import VariableSlice, as_point_in_domain_unordered, as_values_by_variable_slices, Variables, diff_variable_slices, check_limits_within

class DensityFunctionApproximation:

  def __init__(
    self,
    variables: Variables,
    tree: MassTree,
  ):
    check_type("variables", variables, Variables)
    check_type("tree", tree, MassTree)
    self.variables: Variables = variables
    self.tree = tree

  def __call__(
    self,
    *args: t.List[np.ndarray],
    **kwargs: t.Dict[str, np.ndarray]):
    return self.partition(*args, **kwargs).f

  @property
  def z(self):
    return self.tree.z

  @property
  def num_partitions(self):
    return self.tree.num_partitions

  def sparsify(self, relative_z=1e-3):
    return sparsify(self.tree, relative_z=relative_z)

  def region(self, variables: Variables):
    check_type("variables", variables, Variables)
    check_limits_within(
      current_variables=self.variables,
      regional_variables=variables
    )
    new_tree = copy_intersection(self.tree, domain=variables.domain)
    return DensityFunctionApproximation(
      variables=variables,
      tree=new_tree
    )

  def prob(
    self,
    *args: t.Union[np.ndarray, t.List[np.ndarray]],
    **kwargs: t.Dict[str, np.ndarray]):
    return self.partition(*args, **kwargs).f / self.tree.z

  def all_partitions(self):
    return find_all_leaves(self.tree)

  def partition(
    self,
    *args: t.Union[np.ndarray, t.List[np.ndarray]],
    **kwargs: t.Dict[str, np.ndarray],
  ):
    num_args = len(args)
    num_kwargs = len(kwargs)
    num_vars = len(self.variables.variable_slices)
    if (num_args + num_kwargs) > num_vars:
      raise ValueError(
        "The number of arguments must match the number of variables, "
        "but was %s and %s." % (num_args, num_vars))
    values_by_variable_slice = {}
    if num_args == 1:
      x = args[0]
    else:
      if num_args > 0:
        for i in range(num_args):
          s = self.variables.variable_slices[i]
          values_by_variable_slice[s] = args[i]
      if num_kwargs > 0:
        for name in kwargs:
          if name not in self.variables.variable_slice_by_name:
            raise ValueError(
              "Name '%s' was not found among variables." % name)
          s = self.variables.variable_slice_by_name[name]
          values_by_variable_slice[s] = kwargs[name]
      x = as_point_in_domain_unordered(
        variables=self.variables,
        values_by_variable_slice=values_by_variable_slice
      )
    leaf = find_leaf(tree=self.tree, sample=x)
    return leaf

  def conditional(
    self,
    values_by_variable_slice: t.Dict[VariableSlice, np.ndarray],
  ):
    variables = self.variables
    values_by_variable_slice = as_values_by_variable_slices(
      "values_by_variable_slice",
      variables=variables,
      values_by_variable_slice=values_by_variable_slice
    )
    cond_dims = []
    cond_values = []
    for variable_slice, values in values_by_variable_slice.items():
      dims = variables.dims_by_variable_slice(variable_slice)
      cond_dims.extend(dims)
      cond_values.extend(values)
    tree = conditional_tree(self.tree, cond_dims, cond_values)

    exclude_variable_slices = list(values_by_variable_slice.keys())
    variables_after_cond = Variables(
      diff_variable_slices(variables.variable_slices, exclude_variable_slices),
      dtype=variables.domain.dtype
    )
    return DensityFunctionApproximation(
      variables=variables_after_cond, tree=tree)

  def sampler(self):
    partitions = self.all_partitions()
    unnormalized_masses = np.array(
      [partition.z for partition in partitions])
    normalized_masses = unnormalized_masses / self.tree.z
    sample_fn = sampler_in_domains(
      normalized_masses=normalized_masses,
      domains=np.array([partition.domain for partition in partitions])
    )
    dims_by_variable_slice = self.variables.dims_by_variable_slice
    def wrapped_sample_fn(num_samples):
      samples = sample_fn(num_samples)
      sample_groups = []
      for slice in self.variables.variable_slices:
        dims = dims_by_variable_slice(slice)
        sample_groups.append(samples[:, dims])
      return sample_groups
    return wrapped_sample_fn

  def expectation(self, operator):
    operator_fn_num_params = operator.__code__.co_argcount
    if operator_fn_num_params != 3:
      raise ValueError(
        "operator must take three arguments (f, x, z), "
        "but it takes %s" % operator_fn_num_params)
    z = self.tree.z
    partitions = self.all_partitions()
    return np.sum([
      operator(
        f=partition.f,
        x=partition.domain.center_vector,
        z=z
      ) * partition.z
      for partition in partitions
      # Zero mass partitions require no operator call
      if (partition.f / z) > 0
    ], axis=0) / z

  def mode(self):
    partitions = self.all_partitions()
    return partitions[
      int(np.argmax([partition.f for partition in partitions]))
    ].domain.center_vector

  def mean(self):
    return self.expectation(lambda f, x, z: x)

  def var(self):
    mean = self.mean()
    return self.expectation(lambda f, x, z: x ** 2) - mean ** 2

  def std(self):
    return np.sqrt(self.var())

  def differential_entropy(self):
    return self.expectation(lambda f, x, z: -np.log(f / z))

  def copy(self):
    return copy(self.tree)

  def load(self, file_path):
    self.tree = _load_tree(file_path=file_path)

  def save(self, file_path):
    _save_tree(tree=self.tree, file_path=file_path)

def load(variables: Variables, file_path: str):
  check_type("variables", variables, Variables)
  tree = _load_tree(file_path=file_path)
  return DensityFunctionApproximation(
    variables=variables,
    tree=tree
  )

def _load_tree(file_path: str):
  check_type("file_path", file_path, str)
  with open(file_path, "rb") as f:
    tree = pickle.load(f)
  check_type("Loaded object", tree, MassTree)
  return tree

def _save_tree(tree: MassTree, file_path):
  check_type("tree", tree, MassTree)
  check_type("file_path", file_path, str)
  with open(file_path, "wb") as f:
    pickle.dump(tree, f)
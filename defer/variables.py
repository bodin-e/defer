
import collections
import numpy as np
import typing as t
from .bounded_space import BoundedSpace, is_space_within_per_dim
from .validation import check_type, check_shape, check_numpy_is_finite_real

class VariableSlice:

  def __init__(self, variable, indices):
    self.variable: Variable = variable
    self.indices = indices
    self.name = variable.name

  def __repr__(self):
    def indices_str(indices):
      return ", ".join([str(i) for i in indices])
    variable_repr = str(self.variable)
    return "%s[%s]" % (variable_repr, indices_str(self.indices))

class Variable(VariableSlice):

  def __init__(
    self,
    lower,
    upper,
    name,
    ndims=None,
  ):
    check_type("name", name, str)
    if len(name) == 0:
      raise ValueError("'name' cannot be empty.")
    self.lower = np.array(_limits(ndims, lower))
    self.upper = np.array(_limits(ndims, upper))
    if len(self.lower) != len(self.upper):
      raise ValueError(
        "Provided lower and upper limits must be the same length")
    check_numpy_is_finite_real("lower", self.lower)
    check_numpy_is_finite_real("upper", self.upper)
    self.ndims = len(self.lower)
    self.name = name
    super(Variable, self).__init__(
      variable=self,
      indices=list(range(self.ndims))
    )

  def __repr__(self):
    name = self.name
    if name is None:
      name = "<Unnamed>"
    ndims = self.ndims
    return "%s(%sD)" % (name, ndims)

  def __getitem__(self, index):
    if isinstance(index, int):
      _check_range(index, self.ndims)
      return VariableSlice(
        variable=self,
        indices=[self.indices[index]]
      )
    elif isinstance(index, slice):
      if index.start is not None:
        _check_range(index.start, self.ndims)
      if index.stop is not None:
        _check_range(index.stop - 1, self.ndims)
      return VariableSlice(
        variable=self,
        indices=self.indices[index]
      )
    raise ValueError

class Variables:

  def __init__(
    self,
    variable_slices: t.List[VariableSlice],
    dtype=np.dtype("float64"),
  ):
    used_names = set()
    for var in variable_slices:
      name = var.name
      if name in used_names:
        raise ValueError(
          "All names of variables must be unique, "
          "but %s was encountered twice." % name)
      used_names.add(name)
    check_variable_slices("variable_slices", variable_slices)
    check_type("dtype", dtype, np.dtype)
    lower_limit_per_dim = []
    upper_limit_per_dim = []
    variable_and_index_by_dim = {}
    dim_by_variable_and_index = {}
    current_dim = 0
    for slice in variable_slices:
      var = slice.variable
      included_indices = slice.indices

      included_num_dims = len(included_indices)
      dims = np.arange(current_dim, current_dim + included_num_dims)

      for included_index, dim in zip(included_indices, dims):
        variable_and_index_by_dim[dim] = (var, included_index)
        dim_by_variable_and_index[(var, included_index)] = dim

      lower_limit_per_dim.extend(var.lower[included_indices])
      upper_limit_per_dim.extend(var.upper[included_indices])

      current_dim += included_num_dims
    self.variable_slices = variable_slices

    self.variable_slice_by_name = {
      vs.name: vs
      for vs in variable_slices
    }
    self.variable_and_index_by_dim = variable_and_index_by_dim
    self.dim_by_variable_and_index = dim_by_variable_and_index
    self.domain = BoundedSpace(
      lower_limit_vector=np.array(lower_limit_per_dim, dtype=dtype),
      upper_limit_vector=np.array(upper_limit_per_dim, dtype=dtype),
    )
    self.bounds = [
      (
        self.domain.lower_limit_vector[i],
        self.domain.upper_limit_vector[i],
      )
      for i in range(self.domain.ndims)
    ]

  def dims_by_variable_slice(self, variable_slice: VariableSlice):
    return [
      self.dim_by_variable_and_index[(variable_slice.variable, index)]
      for index in variable_slice.indices
    ]

  def __repr__(self):
    list_str = ", ".join([str(var) for var in self.variable_slices])
    return "[%s]" % list_str

def diff_variable_slices(
  a: t.List[VariableSlice],
  b: t.List[VariableSlice])\
  -> t.List[VariableSlice]:
  check_variable_slices("a", a)
  check_variable_slices("b", b)
  # c = a - b (set diff)
  c = []
  for s_a in a:
    matching_indices = []
    for s_b in b:
      var_is_matching = s_a.variable == s_b.variable
      if not var_is_matching:
        continue
      matching_indices.extend(
        np.intersect1d(s_a.indices, s_b.indices)
      )
    has_overlap = len(matching_indices) != 0
    if has_overlap:
      remaining_indices = np.setdiff1d(s_a.indices, matching_indices)
      if len(remaining_indices) == 0:
        continue # Whole variable is excluded
      c.append(VariableSlice(
        variable=s_a.variable,
        indices=np.sort(remaining_indices)
      ))
    else:
      c.append(s_a)
  return c

def as_point_in_domain_unordered(
  variables: Variables,
  values_by_variable_slice: t.Dict[VariableSlice, np.ndarray]
):
  values_by_variable_slice = as_values_by_variable_slices(
    "values_by_variable_slice",
    variables=variables,
    values_by_variable_slice=values_by_variable_slice
  )
  # In order
  values = [None] * len(variables.variable_slices)
  variables_order = [
    slice.variable for slice in variables.variable_slices]
  for slice, value_array in values_by_variable_slice.items():
    index = int(variables_order.index(slice.variable))
    values[index] = value_array
  return as_point_in_domain_ordered(variables, values)

def as_point_in_domain_ordered(
  variables: Variables, values: t.List[np.ndarray]):
  check_type("variables", variables, Variables)
  check_type("values", values, list)
  slices = variables.variable_slices
  if len(slices) != len(values):
    raise ValueError(
      "The number of variable slices must "
      "be the same as the number of value arrays, "
      "but was %s and %s." % (len(slices), len(values)))
  domain = variables.domain
  x = np.zeros([domain.ndims], dtype=domain.dtype)
  for i, slice in enumerate(slices):
    var = slice.variable
    indices = slice.indices
    value_array = values[i]
    value_array = np.atleast_1d(value_array)
    check_shape(
      name="values for variable slice %s" % slice,
      array=value_array,
      expected_shape=(len(indices),)
    )
    check_numpy_is_finite_real("value_array", value_array)
    _check_boundaries(
      var=var,
      indices=indices,
      values=value_array
    )
    dims = variables.dims_by_variable_slice(slice)
    x[dims] = value_array
  return x

def as_values_by_variable_slices(
  name,
  variables: Variables,
  values_by_variable_slice: t.Dict[VariableSlice, np.ndarray]
):
  defined_variables = set(
    [slice.variable for slice in variables.variable_slices])

  check_type(name, values_by_variable_slice, t.Dict)
  check_variable_slices(
    "%s keys" % name, list(values_by_variable_slice.keys()))
  output = {}
  for slice, values in values_by_variable_slice.items():
    var = slice.variable
    indices = slice.indices

    values = np.atleast_1d(values)
    check_shape(
      name="values for variable slice %s" % slice,
      array=values,
      expected_shape=(len(indices),)
    )
    check_numpy_is_finite_real("values", values)

    # Check if variable in variables
    if not var in defined_variables:
      raise ValueError(
        "%s. Variable %s not in Variables %s" %
        (name, var, variables))

    # Check if variable indices is included in variables
    included_indices = set(var.indices)
    for index in indices:
      if index not in included_indices:
        raise ValueError(
          "%s. Variable %s index %s not in Variables %s" %
          (name, var, index, variables))

    # Check boundaries
    _check_boundaries(
      var=var,
      indices=indices,
      values=values
    )

    output[slice] = values

  accounted_for_dims = []
  for slice in values_by_variable_slice:
    dims = variables.dims_by_variable_slice(slice)
    accounted_for_dims.extend(dims)

  # Check for dim duplicates
  counter_accounted_dims = collections.Counter(accounted_for_dims)
  for dim, count in counter_accounted_dims.items():
    dim_has_multiple_suitors = count > 1
    if dim_has_multiple_suitors:
      var, index = variables.variable_and_index_by_dim[dim]
      raise ValueError(
        "Variable %s index %s is specified multiple (%s) times, "
        "making which specified value to use to be ambiguous." %
        (var, index, count))

  return output

def check_limits_within(
  current_variables: Variables,
  regional_variables: Variables
):
  check_type("current_variables", current_variables, Variables)
  check_type("regional_variables", regional_variables, Variables)
  outer_slices = current_variables.variable_slices
  inner_slices = regional_variables.variable_slices
  if len(inner_slices) != len(outer_slices):
    raise ValueError(
      "The number of variable slices must match with the current, "
      "but was %s and %s." % (len(inner_slices), len(outer_slices)))
  for i, (inner_slice, outer_slice) in \
    enumerate(zip(inner_slices, outer_slices)):
    if len(inner_slice.indices) != len(outer_slice.indices):
      raise ValueError(
        "The variable slice at index %s did not have the same number of "
        "indices (%s) as the current variable slice at the same index (%s)."
        % (i, len(inner_slice.indices), len(outer_slice.indices)))
  outer_domain = current_variables.domain
  regional_domain = regional_variables.domain
  is_within_per_dim = is_space_within_per_dim(
    outer=outer_domain,
    inner=regional_domain
  )
  dims_with_invalid_limits = np.where(np.logical_not(is_within_per_dim))[0]
  for dim in dims_with_invalid_limits:
    new_var, new_index = regional_variables.variable_and_index_by_dim[dim]
    old_var, old_index = current_variables.variable_and_index_by_dim[dim]
    new_bounds = [new_var.lower[new_index], new_var.upper[new_index]]
    old_bounds = [old_var.lower[old_index], old_var.upper[old_index]]
    raise ValueError(
      "Provided region is outside the domain. \n"
      "The provided variable %s index %s has bounds %s, "
      "while the current corresponding variable %s index %s has bounds %s. \n"
      "All provided bounds must be within the current bounds." %
      (new_var, new_index, new_bounds, old_var, old_index, old_bounds))

def check_variable_slices(name, variable_slices: t.List[VariableSlice]):
  check_type(name, variable_slices, t.List)
  if len(variable_slices) == 0:
    raise ValueError("%s cannot be empty." % name)
  for i, slice in enumerate(variable_slices):
    check_type("%s (slice %s)" % (name, slice), slice, VariableSlice)

def _limits(num_dims, limit):
  is_num_dims_explicit = num_dims is not None
  is_rank_one_limit = np.isscalar(limit) or len(limit) == 1
  if is_num_dims_explicit:
    if is_rank_one_limit:
      return limit * np.ones([num_dims])
    assert num_dims == len(limit)
    return limit
  if np.isscalar(limit):
    return [limit]
  return limit

def _check_range(i, ndims):
  pos = i >= 0
  if pos:
    if i >= ndims:
      raise IndexError
    return  # Is valid
  if abs(i) > ndims:
    raise IndexError

def _check_boundaries(var: Variable, indices, values: np.ndarray):
  indices = np.array(indices)
  lower = var.lower[indices]
  upper = var.upper[indices]
  indices_outside_bounds = indices[
    np.logical_or(values < lower, values > upper)]
  for i, index in enumerate(indices_outside_bounds):
    raise ValueError(
      "The value specified for variable %s index %s is outside its bounds. "
      "It is %s, but must be within [%s, %s]." %
      (var, index, values[i], lower[i], upper[i]))
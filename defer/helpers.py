

import numpy as np
import typing as t
from .bounded_space import subspace_by_dims
from .tree_construction.algorithm import DEFER
from .approximation import DensityFunctionApproximation
from .variables import Variables, diff_variable_slices, as_values_by_variable_slices, VariableSlice
from .tree import MassTree
from .bounded_space import BoundedSpace
from .validation import check_type, as_numpy_scalar, check_numpy_is_finite, check_numpy_is_real

def construct(
  variables: Variables,
  fn: t.Callable,
  is_log_fn:bool,
  num_fn_calls:int,
  callback:t.Callable[[int, DensityFunctionApproximation], None]=
  lambda i, tree:
  print("#Evals: %s. Log Z: %.2f" %
        (tree.num_partitions, np.log(tree.z))),
  callback_freq_fn_calls:int=10,
  is_vectorized_fn=False,
  continue_on: DensityFunctionApproximation=None,
  constant_fn_arguments:t.Dict[VariableSlice, np.ndarray]=None,
):
  check_type("num_fn_calls", num_fn_calls, int)
  if num_fn_calls < 1:
    raise ValueError(
      "num_fn_calls must be greater than zero, but was %s"
      % num_fn_calls)
  check_type("callback", callback, t.Callable)
  check_type("callback_freq_fn_calls", callback_freq_fn_calls, int)
  callback_fn_num_params = callback.__code__.co_argcount
  if callback_fn_num_params != 2:
    raise ValueError(
      "callback must take two arguments, but it takes %s"
      % callback_fn_num_params)
  if callback_freq_fn_calls < 1:
    raise ValueError(
      "callback_freq_fn_calls must be greater than zero, but was %s"
      % callback_freq_fn_calls)
  density, defer = _prepare_defer(
    variables=variables,
    fn=fn,
    is_log_fn=is_log_fn,
    is_vectorized_fn=is_vectorized_fn,
    continue_on=continue_on,
    constant_fn_arguments=constant_fn_arguments
  )
  return _callback_loop(
    defer=defer,
    density=density,
    num_fn_calls=num_fn_calls,
    callback=callback,
    callback_freq_fn_calls=callback_freq_fn_calls
  )

def _callback_loop(defer, density, num_fn_calls, callback, callback_freq_fn_calls):
  first_call = True
  num_partitions_at_last_callback = 0
  for i, _ in enumerate(defer(num_fn_calls)):
    since_last_callback = \
      density.num_partitions - num_partitions_at_last_callback
    if first_call or since_last_callback >= callback_freq_fn_calls:
      num_partitions_at_last_callback = density.num_partitions
      first_call = False
      callback(i, density)
  callback(i + 1, density)
  return density

def _prepare_defer(
  variables: Variables,
  fn: t.Callable,
  is_log_fn:bool,
  is_vectorized_fn=False,
  continue_on: DensityFunctionApproximation=None,
  constant_fn_arguments:t.Dict[VariableSlice, np.ndarray]=None,
):
  check_type("is_vectorized_fn", is_vectorized_fn, bool)
  provided_continue_on = continue_on is not None
  if provided_continue_on:
    check_type("continue_on", continue_on, DensityFunctionApproximation)
  density_fn = density_fn_from_variables(
    variables=variables,
    fn=fn,
    is_log_fn=is_log_fn,
    is_vectorized_fn=is_vectorized_fn,
    constant_fn_arguments=constant_fn_arguments,
  )
  if continue_on is None:
    tree = _tree_root(
      domain=variables.domain,
      density_fn=density_fn,
      is_vectorized_density_fn=is_vectorized_fn
    )
    density = DensityFunctionApproximation(
      variables=variables, tree=tree)
  else:
    tree = continue_on.tree
    density = continue_on
  defer = DEFER(
    root=tree,
    density_func=density_fn,
    is_vectorized_density_func=is_vectorized_fn,
  )
  return density, defer

def construct_marginal(
  variables: Variables,
  fn: t.Callable,
  is_log_fn: bool,
  num_outer_fn_calls: int,
  num_inner_fn_calls: int,
  keep_variable_slices: t.List[VariableSlice]=None,
  marginalize_variable_slices: t.List[VariableSlice]=None,
  is_vectorized_fn=False,
  callback:t.Callable[[int, DensityFunctionApproximation], None]=
  lambda tree: None,
  callback_freq_fn_calls:int=1,
):
  check_type("variables", variables, Variables)
  check_type("is_vectorized_fn", is_vectorized_fn, bool)
  check_type("num_outer_fn_calls", num_outer_fn_calls, int)
  check_type("num_inner_fn_calls", num_inner_fn_calls, int)
  check_type("callback", callback, t.Callable)
  check_type("callback_freq_fn_calls", callback_freq_fn_calls, int)
  def check(name, variable_slices):
    for i, slice in enumerate(variable_slices):
      check_type("%s[%s]" % (name, i), slice, VariableSlice)
  is_keep_specified = keep_variable_slices is not None
  is_remove_specified = marginalize_variable_slices is not None
  if is_remove_specified and is_keep_specified:
    raise ValueError(
      "Specify either which VariableSlice:s to keep "
      "or which to be marginalized, "
      "using keep_variable_slices and marginalize_variable_slices, "
      "respectively.")
  if is_remove_specified:
    check("marginalize_variable_slices", marginalize_variable_slices)
    keep_variable_slices = diff_variable_slices(
      a=variables.variable_slices,
      b=marginalize_variable_slices
    )
  check("keep_variable_slices", keep_variable_slices)
  for i, variable_slice in enumerate(keep_variable_slices):
    check_type("keep_variable_indices[%s]" % i, variable_slice, VariableSlice)
  if num_outer_fn_calls < 1:
    raise ValueError(
      "num_outer_fn_calls must be greater than zero, but was %s"
      % num_outer_fn_calls)
  if num_inner_fn_calls < 1:
    raise ValueError(
      "num_inner_fn_calls must be greater than zero, but was %s"
      % num_inner_fn_calls)
  callback_fn_num_params = callback.__code__.co_argcount
  if callback_fn_num_params != 2:
    raise ValueError(
      "callback must take two arguments, but it takes %s"
      % callback_fn_num_params)
  if callback_freq_fn_calls < 1:
    raise ValueError(
      "callback_freq_fn_calls must be greater than zero, but was %s"
      % callback_freq_fn_calls)
  remaining_variable_slices = diff_variable_slices(
    a=variables.variable_slices,
    b=keep_variable_slices
  )
  if len(remaining_variable_slices) == 0:
    raise ValueError(
      "keep_variable_slices includes all variable slices in variables. "
      "As such, there are no dimensions to marginalize."
    )

  density_fn = density_fn_from_variables(
    variables=variables,
    fn=fn,
    is_log_fn=is_log_fn,
    is_vectorized_fn=is_vectorized_fn,
  )

  keep_dims = []
  for variable_slice in keep_variable_slices:
    dims = variables.dims_by_variable_slice(variable_slice)
    keep_dims.extend(dims)

  marginal_density_fn, marginal_domain = _approx_marginal_density_fn(
    density_fn=density_fn,
    domain=variables.domain,
    keep_dims=keep_dims,
    num_inner_evaluations=num_inner_fn_calls,
    is_vectorized_fn=is_vectorized_fn,
  )
  marg_variables = Variables(
    variable_slices=keep_variable_slices,
    dtype=variables.domain.dtype
  )
  tree = _tree_root(
    domain=marginal_domain,
    density_fn=marginal_density_fn,
    is_vectorized_density_fn=is_vectorized_fn
  )
  density = DensityFunctionApproximation(
    variables=marg_variables, tree=tree)
  defer = DEFER(
    root=tree,
    density_func=marginal_density_fn,
    is_vectorized_density_func=is_vectorized_fn,
  )
  return _callback_loop(
    defer=defer,
    density=density,
    num_fn_calls=num_outer_fn_calls,
    callback=callback,
    callback_freq_fn_calls=callback_freq_fn_calls
  )

def density_fn_from_variables(
  variables: Variables,
  fn: t.Callable,
  is_log_fn: bool,
  is_vectorized_fn: bool,
  constant_fn_arguments:t.Dict[VariableSlice, np.ndarray]=None,
):
  check_type("variables", variables, Variables)
  check_type("fn", fn, t.Callable)
  check_type("is_log_fn", is_log_fn, bool)
  has_constant_args = constant_fn_arguments is not None
  constant_value_by_dim = {}
  if has_constant_args:
    constant_fn_arguments = as_values_by_variable_slices(
      "constant_fn_arguments",
      variables=variables,
      values_by_variable_slice=constant_fn_arguments
    )
    for slice, values in constant_fn_arguments.items():
      dims = variables.dims_by_variable_slice(
        variable_slice=slice
      )
      for dim, value in zip(dims, values):
        constant_value_by_dim[dim] = value

  # TODO Check if wrong arguments in runtime.
  # TODO Check if unaccounted for arguments
  dim_by_variable_and_index = variables.dim_by_variable_and_index

  def density(x):
    """
    If is_vectorized is True, x is expected to be (N, D), else (D, ).
    """
    if is_vectorized_fn:
      assert len(x.shape) == 2
      num_points = x.shape[0]
    else:
      assert len(x.shape) == 1
      num_points = 1

    arguments = []

    current_x_dim = 0
    for slice in variables.variable_slices:
      var = slice.variable
      indices = slice.indices
      dims = [dim_by_variable_and_index[(var, index)] for index in indices]

      values = []
      for dim in dims:
        # Argument values may either come from x,
        # or from constant_fn_arguments.
        if dim in constant_value_by_dim:
          value = constant_value_by_dim[dim]
          if is_vectorized_fn:
            value = np.repeat(value, repeats=num_points)
        else:
          if is_vectorized_fn:
            value = x[:, current_x_dim]
          else:
            value = x[current_x_dim]
          current_x_dim += 1
        values.append(value)

      argument = np.transpose(np.array(values))
      arguments.append(argument)

    def sanitize(f):
      # TODO Add arguments passed (by variable) in error message.
      f = as_numpy_scalar("Output of fn", f)
      check_numpy_is_real("Output of fn", f)
      f = np.float128(f)
      if is_log_fn:
        f = np.exp(f)
      else:
        if f < 0:
          raise ValueError("Output of fn must be non-negative, but was %s" % f)
      check_numpy_is_finite("Output of fn", f)
      return f

    if is_vectorized_fn:
      return np.array([sanitize(f=f) for f in fn(*arguments)])
    return sanitize(f=fn(*arguments))

  return density

def _approx_marginal_density_fn(
  density_fn: t.Callable,
  domain: BoundedSpace,
  keep_dims: t.List[int],
  num_inner_evaluations: int,
  is_vectorized_fn: bool
):
  check_type("density_fn", density_fn, t.Callable)
  check_type("domain", domain, BoundedSpace)
  check_type("keep_dims", keep_dims, t.List)
  check_type("num_inner_evaluations", num_inner_evaluations, int)
  if len(keep_dims) == 0:
    raise ValueError("keep_dims cannot be empty.")
  for dim in keep_dims:
    if not (0 <= dim < domain.ndims):
      raise ValueError(
        "All dimensions in keep_dims must be within the domain, but was %s."
        % keep_dims
      )
  if num_inner_evaluations < 1:
    raise ValueError(
      "num_inner_evaluations cannot be less than 1, but was %s" %
      num_inner_evaluations)
  marginalize_dims = sorted(list(set(range(domain.ndims)) - set(keep_dims)))
  marginal_domain = subspace_by_dims(domain, dims=keep_dims)
  marginalize_domain = subspace_by_dims(domain, dims=marginalize_dims)
  def marginal_density_fn(x_keep):
    def inner_density_fn(x_inner):
      x_full = np.zeros([domain.ndims], dtype=domain.dtype)
      x_full[keep_dims] = x_keep
      x_full[marginalize_dims] = x_inner
      f = density_fn(x_full)
      return f
    inner_tree = _tree_root(
      domain=marginalize_domain,
      density_fn=inner_density_fn,
      is_vectorized_density_fn=is_vectorized_fn
    )
    inner_defer = DEFER(
      root=inner_tree,
      density_func=inner_density_fn,
      is_vectorized_density_func=is_vectorized_fn,
    )
    for _ in inner_defer(num_inner_evaluations):
      pass
    return inner_tree.z
  return marginal_density_fn, marginal_domain

def _tree_root(
  density_fn: t.Callable,
  is_vectorized_density_fn: bool,
  domain: BoundedSpace,
):
  check_type("density_func", density_fn, t.Callable)
  check_type("domain", domain, BoundedSpace)
  if is_vectorized_density_fn:
    f = as_numpy_scalar("f", density_fn(np.array([domain.center_vector])))
  else:
    f = density_fn(domain.center_vector)
  return MassTree(
    domain=domain,
    f=f,
    parent=None
  )


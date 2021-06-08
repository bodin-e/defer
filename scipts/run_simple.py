
import corner
import sys
import matplotlib.pylab as plt
from density_functions.gravitational_wave import *
from density_functions.misc import *
from defer.helpers import *
from defer.variables import Variable
from defer.bounded_space import sample_uniform

# log_density_fn = load_students_t(ndims=3, loc=np.random.sample(3))
# log_density_fn = load_two_bananas()
# log_density_fn = load_gw_injected()
# log_density_fn = load_cigar(ndims=5)
# log_density_fn = load_gaussian(ndims=5, std=0.01)
# log_density_fn = load_alien()
# log_density_fn = load_gaussian_two_component_mixture()
# log_density_fn = load_dart_boards()
log_density_fn = load_eggbox()

x = Variable(
  lower=log_density_fn.domain.lower_limit_vector,
  upper=log_density_fn.domain.upper_limit_vector,
  name="x"
)

variables = Variables([x])

approx: DensityFunctionApproximation = construct(
  fn=log_density_fn,
  is_log_fn=True,
  variables=variables,
  num_fn_calls=3000,
  callback=lambda i, density:
  print("#Evals: %s. Log Z: %.2f" %
        (density.num_partitions, np.log(density.z))),
  callback_freq_fn_calls=30,
  is_vectorized_fn=False
)

# approx.save("/tmp/approx.pickle")
# approx.load("/tmp/approx.pickle")

def plot(density: DensityFunctionApproximation, samples=None):
  if samples is None:
      print("Preparing sampler..")
      sampler = density.sampler()
      print("Sampling..")
      samples_per_variable = sampler(num_samples=10 ** 6)
      samples = np.concatenate(samples_per_variable, axis=-1)
  print("Plotting..")
  figure = corner.corner(
    samples,
    range=density.variables.bounds,
    labels=[
      "%s: dim %s" % (var.name, index)
      for var in density.variables.variable_slices
      for index in var.indices
    ],
    plot_contours=False,
    no_fill_contours=True,
    bins=150,
    plot_datapoints=False,
  )
  plt.show()

sampler = approx.sampler()
x_samples, = sampler(num_samples=10 ** 6)

plot(approx, samples=x_samples)
plt.show()

evidence = approx.z
differential_entropy = approx.expectation(
  lambda f, x, z: -np.log(f / z))
mean = approx.mean()
var = approx.var()
mode = approx.mode()

x_test = sample_uniform(log_density_fn.domain)
f_test = approx(x_test)
p_test = approx.prob(x_test)

## Conditional ##################################

# Only run the rest of script if the domain has more than one dimension.
if approx.variables.domain.ndims < 2:
  sys.exit(0)

# Derive conditional approximation
approx_conditional: DensityFunctionApproximation = approx.conditional({
  x[-1]: mode[-1]
})

plot(approx_conditional)
plt.show()

conditional_evidence = approx_conditional.z
conditional_differential_entropy = approx_conditional.expectation(
  lambda f, x, z: -np.log(f / z))
conditional_mean = approx_conditional.mean()
conditional_var = approx_conditional.var()
conditional_mode = approx_conditional.mode()

## Marginal #####################################

approx_marginal: DensityFunctionApproximation = construct_marginal(
  fn=log_density_fn,
  variables=variables,
  marginalize_variable_slices=[x[-1]],
  is_log_fn=True,
  num_outer_fn_calls=500,
  num_inner_fn_calls=20,
  callback=lambda i, density:
  print("#Evals: %s. Log Z: %.2f" %
        (density.num_partitions, np.log(density.z))),
  callback_freq_fn_calls=100,
)

plot(approx_marginal)
plt.show()

marginal_evidence = approx_marginal.z
marginal_differential_entropy = approx_marginal.expectation(
  lambda f, x, z: -np.log(f / z))
marginal_mean = approx_marginal.mean()
marginal_var = approx_marginal.var()
marginal_mode = approx_marginal.mode()
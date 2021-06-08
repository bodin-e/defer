
from scipy.stats import multivariate_normal
from .core import Simulator
from defer.bounded_space import *
from defer.helpers import *

def unit_domain(ndims: int, dtype) -> BoundedSpace:
  return BoundedSpace(
    lower_limit_vector=np.zeros([ndims], dtype=dtype),
    upper_limit_vector=np.ones([ndims], dtype=dtype)
  )

def load_gaussian_two_component_mixture():
  D = 4
  domain = unit_domain(ndims=D, dtype=np.float64)

  def f(x):

    width_a = 0.010
    width_b = 0.015

    def base_covariance(width):
      covar = np.diag([(1.5 * width) ** 2] * D)
      covar[0, 1] = - (1 * width) ** 2
      covar[1, 0] = - (1 * width) ** 2
      return covar

    covar_a = base_covariance(width_a)
    covar_b = base_covariance(width_b)

    correlation_width = np.minimum(width_a, width_b)

    covar_b[0, 2] = + correlation_width ** 2
    covar_b[2, 0] = + correlation_width ** 2

    covar_b[0, 3] = - correlation_width ** 2
    covar_b[3, 0] = - correlation_width ** 2

    a_log = multivariate_normal.logpdf(
      x,
      mean=[0.63263116, 0.74013062, 0.72316873, 0.24708191],
      cov=covar_a
    )
    b_log = multivariate_normal.logpdf(
      x,
      mean=[0.51387785, 0.46667482, 0.37765008, 0.79952093],
      cov=covar_b
    )
    a = np.exp(np.float128(a_log))
    b = np.exp(np.float128(b_log))
    v = np.log(2.5 * a + b)
    return v.astype(np.float64)

  return Simulator(
    name="gaussian_mixture_4d",
    domain=domain,
    fn=f
  )

def load_canoe(ndims):
  from scipy.stats import multivariate_normal

  D = ndims
  domain = unit_domain(ndims=D, dtype=np.float64)

  mean = 0.5 * np.ones([D])
  inner_c = .95
  inner_cov = inner_c * np.ones([D, D]) + (1 - inner_c) * np.eye(D)
  inner_cov *= 0.01

  outer_c = .6
  outer_cov = outer_c * np.ones([D, D]) + (1 - outer_c) * np.eye(D)
  outer_cov *= 0.02

  def f(x):
    # Base
    base = 1
    # Bump
    outer = np.exp(np.float128(
      multivariate_normal.logpdf(
        x,
        mean=mean,
        cov=outer_cov
      )
    ))
    # Dip
    inner = np.exp(np.float128(
      multivariate_normal.logpdf(
        x,
        mean=mean,
        cov=inner_cov
      )
    ))
    a = 20
    b = 5
    v_unconstrained = 2 * base - a * outer + b * inner
    v = np.maximum(v_unconstrained, 0)
    return np.float64(np.log(v))

  class GaussianSimulator(Simulator):

    def sample(self, num_samples):
      return np.random.multivariate_normal(
        mean=mean,
        cov=inner_cov,
        size=num_samples
      )

    def entropy(self):
      return multivariate_normal.differential_entropy(mean, inner_cov)

  return GaussianSimulator(
    name="canoe_%sd" % ndims,
    domain=domain,
    fn=f
  )

def load_cigar(ndims):
  from scipy.stats import multivariate_normal

  D = ndims
  domain = unit_domain(ndims=D, dtype=np.float64)

  mean = 0.5 * np.ones([D])
  c = 0.99
  cov = c * np.ones([D, D]) + (1 - c) * np.eye(D)
  cov *= 0.01

  def f(x):
    v = multivariate_normal.logpdf(
      x,
      mean=mean,
      cov=cov
    )
    return v

  class GaussianSimulator(Simulator):

    def sample(self, num_samples):
      return np.random.multivariate_normal(
        mean=mean,
        cov=cov,
        size=num_samples
      )

    def entropy(self):
      return multivariate_normal.differential_entropy(mean, cov)

  return GaussianSimulator(
    name="cigar_%sd" % ndims,
    domain=domain,
    fn=f
  )

def load_students_t(ndims, loc):
  assert len(loc) == ndims

  D = ndims

  import tensorflow_probability as tfp
  import tensorflow as tf
  tfd = tfp.distributions

  domain = unit_domain(ndims=D, dtype=np.float64)

  df = 2.5 + (D / 2)
  scale = 0.01 * np.ones([D])

  dist = tfd.StudentT(
    df=df,
    loc=loc,
    scale=scale
  )

  @tf.function(
    input_signature=(
      tf.TensorSpec(shape=D, dtype=np.float64),
    )
  )
  def tf_f(x):
    dist_inside = tfd.StudentT(
      df=df,
      loc=loc,
      scale=scale
    )
    return tf.reduce_sum(dist_inside.log_prob(x))

  def log_density(x):
    return tf_f(x).numpy()

  class DistributionSimulator(Simulator):

    def entropy(self):
      return tf.reduce_sum(dist.entropy()).numpy()

    def loc(self):
      return loc

  return DistributionSimulator(
    name="students_t_%sd" % ndims,
    domain=domain,
    fn=log_density
  )


def load_gaussian(ndims, std=0.1):

  from scipy.stats import multivariate_normal

  D = ndims
  domain = unit_domain(ndims=D, dtype=np.float64)

  mean = 0.6 * np.ones([D])
  cov = np.diag([std ** 2] * np.ones([D]))

  def f(x):
    return multivariate_normal.logpdf(
      x,
      mean=mean,
      cov=cov
    )

  class GaussianSimulator(Simulator):

    def sample(self, num_samples):
      return np.random.multivariate_normal(
        mean=mean,
        cov=cov,
        size=num_samples
      )

  return GaussianSimulator(
    name="gaussian_%sd" % ndims,
    domain=domain,
    fn=f
  )


def load_eggbox():

  def f(x):
    tmax = 5. * np.pi
    t = 2.0 * tmax * x - tmax
    return np.log((2 + np.cos(t[0] / 2.0) * np.cos(t[1] / 2.0)) ** 5.0)

  domain = unit_domain(ndims=2, dtype=np.float64)

  return Simulator(
    name="eggbox",
    domain=domain,
    fn=f
  )


def load_dart_boards():
  ndims = 2
  domain = unit_domain(ndims=ndims, dtype=np.float64)

  def f(x):
    circle_upper_left = np.array([0.25] * ndims)
    circle_lower_right = np.array([0.75] * ndims)

    num_circles_left = 0
    num_circles_right = 0
    radiuses = np.array([0.1, 0.2, 0.3, 0.4]) * \
               (domain.diameter / np.sqrt(2))
    for radius in radiuses:
      if np.linalg.norm(x - circle_upper_left) < radius:
        num_circles_left += 1
      if np.linalg.norm(x - circle_lower_right) < radius:
        num_circles_right += 1

    return np.log(0.5 + 0.1 * (num_circles_left - num_circles_right))

  return Simulator(
    name="dart_boards_%s" % ndims,
    domain=domain,
    fn=f
  )

def load_alien():

  def eval(simulator, unit_x):
    return simulator(absolute_point(simulator.domain, unit_x))

  u_domain = unit_domain(ndims=2, dtype=np.float64)

  def split(
    axis: int,
    upper_bounds,
    sims,
    name: str = "split",
  ):
    assert len(upper_bounds) == len(sims) - 1
    c = upper_bounds[0]
    for v in upper_bounds[1:]:
      assert v > c
      c = v
    def f_split(x):
      index = 0
      current_lower_bound = 0
      for v in upper_bounds:
        if x[axis] < v:
          break
        current_lower_bound = v
        index += 1
      lb = current_lower_bound
      if index < len(upper_bounds):
        ub = upper_bounds[index]
      else:
        ub = 1
      unit_x = x.copy()
      unit_x[axis] = (unit_x[axis] - lb) / (ub - lb)
      sim = sims[index]
      return eval(sim, unit_x=unit_x)
    return Simulator(
      name=name,
      domain=u_domain,
      fn=f_split
    )

  def scale_density(
    sim,
    name="scale_density",
    intensity_scalar=1.
  ):
    def scale(theta):
      log_f = sim(theta)
      v = log_f.astype(np.float128)
      v_exp = np.exp(v)
      v_scaled = intensity_scalar * v_exp
      return np.log(v_scaled)

    return Simulator(
      name=name,
      domain=sim.domain,
      fn=scale
    )

  def add(
    sims,
    name="add",
    intensity_scalar=1.
  ):
    def add(unit_theta):
      thetas_per_sim = np.array([
        sim(absolute_point(sim.domain, np.array(unit_theta)))
        for sim in sims
      ])
      v = thetas_per_sim.astype(np.float128)
      v_exp = np.exp(v)
      v_sum = intensity_scalar * np.mean(v_exp)
      return np.log(v_sum)

    return Simulator(
      name=name,
      domain=u_domain,
      fn=add
    )

  def null():
    return Simulator(
      name="null",
      domain=u_domain,
      fn=lambda theta: -np.inf
    )

  nested_simulator = split(
    axis=0,
    upper_bounds=[0.15],
    sims=[
      split(
        axis=1,
        upper_bounds=[0.5],
        sims=[
          load_banana(
            x_scalar=0.8,
            y_scalar=1,
            rotate_degrees=0
          ),
          load_banana(
            x_scalar=0.8,
            y_scalar=1,
            rotate_degrees=180
          )
        ]
      ),
      split(
        axis=1,
        upper_bounds=[.15, .30],
        sims=[
          split(
            axis=0,
            upper_bounds=np.linspace(1/6, 1, num=6)[:-1],
            sims=6 * [
              scale_density(
                load_gaussian(ndims=2),
                intensity_scalar=1 / 100
              )
            ]
          ),
          split(
            axis=0,
            upper_bounds=[0.5/6, 1 - 0.5/6],
            sims=[
              null(),
              split(
                axis=0,
                upper_bounds=np.linspace(1 / 5, 1, num=5)[:-1],
                sims=5 * [
                  scale_density(
                    load_gaussian(ndims=2),
                    intensity_scalar=1 / 100
                  )
                ]
              ),
              null()
            ]
          ),
          add(
            intensity_scalar=1.5,
            sims=[
              load_banana(
                x_scalar=0.5,
                y_scalar=1.2,
                rotate_degrees=0
              ),
              load_banana(
                x_scalar=1,
                y_scalar=1.2,
                rotate_degrees=0
              ),
              load_banana(
                x_scalar=2,
                y_scalar=0.8,
                rotate_degrees=0
              ),
              split(
                axis=1,
                upper_bounds=[0.4],
                sims=[
                  null(),
                  split(
                    axis=0,
                    upper_bounds=[0.25, 0.75],
                    sims=[
                      null(),
                      null(),
                      scale_density(
                        load_gaussian(ndims=2),
                        intensity_scalar=1 / 100
                      )
                    ]
                  )
                ]
              )
            ]
          ),
        ]
      )
    ]
  )

  return nested_simulator

def load_banana(
  x_scalar=1.0,
  y_scalar=1.2,
  rotate_degrees=0
):

  import tensorflow_probability as tfp
  import tensorflow as tf

  def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)

  def _forward(x):
    y_0 = x[..., 0:1]
    y_1 = x[..., 1:2] - y_0 ** 2 - 1
    y_tail = x[..., 2:-1]
    return tf.concat([y_0, y_1, y_tail], axis=-1)

  def _inverse(y):
    x_0 = y[..., 0:1]
    x_1 = y[..., 1:2] + x_0 ** 2 + 1
    x_tail = y[..., 2:-1]

    return tf.concat([x_0, x_1, x_tail], axis=-1)

  def _inverse_log_det_jacobian(y):
    return tf.zeros(shape=())

  domain = BoundedSpace(
    lower_limit_vector=np.array([-5, -5], dtype=np.float64),
    upper_limit_vector=np.array([5, 5], dtype=np.float64),
  )

  dtype = tf.float64

  @tf.function(
    input_signature=(
      tf.TensorSpec(shape=[2], dtype=dtype),
    )
  )
  def tf_f_split(x_a):
    Sigma_a = np.float32(np.eye(N=2)) + 0.3 * np.eye(N=2)[::-1]
    p_x_a = tfp.distributions.MultivariateNormalTriL(
      scale_tril=tf.linalg.cholesky(Sigma_a)
    )
    banana = tfp.bijectors.Inline(
      forward_fn=_forward,
      inverse_fn=_inverse,
      inverse_log_det_jacobian_fn=_inverse_log_det_jacobian,
      inverse_min_event_ndims=1,
      is_constant_jacobian=True,
    )
    p_y_a = tfp.distributions.TransformedDistribution(distribution=p_x_a, bijector=banana)
    return p_y_a.prob(x_a)

  def tf_f(x):
    x = rotate(x, degrees=rotate_degrees)
    x_centered = \
      (x + tf.constant([0, -3], dtype=dtype)) * tf.constant([x_scalar, y_scalar], dtype=dtype)
    return tf.math.log(tf_f_split(x_centered))

  def f(x):
    assert x.shape == (2, )
    return tf_f(x).numpy()

  class DistributionSimulator(Simulator):

    @staticmethod
    def tf_log_prob(x):
      return tf_f(x)

  return DistributionSimulator(
    name="two_bananas",
    domain=domain,
    fn=f
  )

def load_two_bananas():

  import tensorflow_probability as tfp
  import tensorflow as tf

  def _forward(x):
    y_0 = x[..., 0:1]
    y_1 = x[..., 1:2] - y_0 ** 2 - 1
    y_tail = x[..., 2:-1]
    return tf.concat([y_0, y_1, y_tail], axis=-1)

  def _inverse(y):
    x_0 = y[..., 0:1]
    x_1 = y[..., 1:2] + x_0 ** 2 + 1
    x_tail = y[..., 2:-1]
    return tf.concat([x_0, x_1, x_tail], axis=-1)

  def _inverse_log_det_jacobian(y):
    return tf.zeros(shape=())

  domain = BoundedSpace(
    lower_limit_vector=np.array([-5, -5], dtype=np.float64),
    upper_limit_vector=np.array([5, 5], dtype=np.float64)
  )

  dtype = tf.float64

  @tf.function(
    input_signature=(
      tf.TensorSpec(shape=[2], dtype=dtype),
      tf.TensorSpec(shape=[2], dtype=dtype),
    )
  )
  def tf_f_split(x_a, x_b):
    Sigma_a = np.float32(np.eye(N=2)) + 0.3 * np.eye(N=2)[::-1]
    Sigma_b = np.float32(np.eye(N=2)) + 0.95 * np.eye(N=2)[::-1]
    p_x_a = tfp.distributions.MultivariateNormalTriL(
      scale_tril=tf.linalg.cholesky(Sigma_a)
    )
    p_x_b = tfp.distributions.MultivariateNormalTriL(
      scale_tril=tf.linalg.cholesky(Sigma_b)
    )
    banana = tfp.bijectors.Inline(
      forward_fn=_forward,
      inverse_fn=_inverse,
      inverse_log_det_jacobian_fn=_inverse_log_det_jacobian,
      inverse_min_event_ndims=1,
      is_constant_jacobian=True,
    )
    p_y_a = tfp.distributions.TransformedDistribution(distribution=p_x_a, bijector=banana)
    p_y_b = tfp.distributions.TransformedDistribution(distribution=p_x_b, bijector=banana)
    return p_y_a.prob(x_a) + p_y_b.prob(x_b)

  def tf_f(x):
    x_a = (x + tf.constant([-2,  1], dtype=dtype)) * tf.constant([1, -1], dtype=dtype)
    x_b = (x + tf.constant([2, -2], dtype=dtype)) * tf.constant([1, 1], dtype=dtype)
    return tf.math.log(tf_f_split(x_a, x_b))

  def f(x):
    assert x.shape == (2, )
    return tf_f(x).numpy()

  class DistributionSimulator(Simulator):

    @staticmethod
    def tf_log_prob(x):
      return tf_f(x)

  return DistributionSimulator(
    name="two_bananas",
    domain=domain,
    fn=f
  )
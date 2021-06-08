
from defer.bounded_space import *
from .core import Simulator

name_order = [
  # "mass_1",
  # "mass_2",

  # Used in paper --------------------
  'luminosity_distance',
  'theta_jn',
  'psi',
  'phase',
  'a_1',
  'a_2',
  # -----------------------------

  # 'phi_12',
  # 'phi_jl',

  # 'tilt_1',
  # 'tilt_2',

  # 'dec',
  # 'ra',
]
index_by_name = {
  name: i
  for i, name in enumerate(name_order)
}
name_by_index = {
  index_by_name[name]: name
  for name in index_by_name
}
bounds_by_name = {
  'mass_1': [5, 100],
  'mass_2': [5, 100],
  'luminosity_distance': [100, 5000],
  'theta_jn': [0, np.pi],
  'phase': [0, 2 * np.pi],
  'psi': [0, np.pi],
  'a_1': [0, 0.8],
  'a_2': [0, 0.8],
  'phi_12': [0, 2 * np.pi],
  'phi_jl': [0, 2 * np.pi],
  'tilt_1': [0, np.pi],
  'tilt_2': [0, np.pi],
  'dec': [-0.5 * np.pi, 0.5 * np.pi],
  'ra': [0, 2 * np.pi],
}
bounds = [
  bounds_by_name[name_by_index[i]]
  for i in range(len(name_by_index))
]

def get_log_joint_fn():
  import numpy as np
  import bilby

  # Set the duration and sampling frequency of the data segment that we're
  # going to inject the signal into
  duration = 4.
  sampling_frequency = 2048.

  # Specify the output directory and the name of the simulation.
  # outdir = 'outdir'
  # label = 'fast_tutorial'
  # bilby.core.utils.setup_logger(outdir=outdir, label=label)

  # Set up a random seed for result reproducibility.  This is optional!
  outside_state = np.random.get_state()
  fixed_seed = 88170235
  np.random.seed(fixed_seed) # 88170235

  # We are going to inject a binary black hole waveform.  We first establish a
  # dictionary of parameters that includes all of the different waveform
  # parameters, including masses of the two black holes (mass_1, mass_2),
  # spins of both black holes (a, tilt, phi), etc.
  injection_parameters = dict(
    mass_1=36., mass_2=29.,
    # mass_ratio=29/36,
    a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

  # Fixed arguments passed into the source model
  waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                            reference_frequency=50., minimum_frequency=20.)

  # Create the waveform_generator using a LAL BinaryBlackHole source function
  waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)

  # Set up interferometers.  In this case we'll use two interferometers
  # (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
  # sensitivity
  ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
  ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
  ifos.inject_signal(waveform_generator=waveform_generator,
                     parameters=injection_parameters)

  # Set up a PriorDict, which inherits from dict.
  # By default we will sample all terms in the signal models.  However, this will
  # take a long time for the calculation, so for this example we will set almost
  # all of the priors to be equal to their injected values.  This implies the
  # prior is a delta function at the true, injected value.  In reality, the
  # sampler implementation is smart enough to not sample any parameter that has
  # a delta-function prior.
  # The above list does *not* include mass_1, mass_2, theta_jn and luminosity
  # distance, which means those are the parameters that will be included in the
  # sampler.  If we do nothing, then the default priors get used.
  priors = bilby.gw.prior.BBHPriorDict()
  priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 1,
    maximum=injection_parameters['geocent_time'] + 1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')

  inject_parameters = [
    'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'ra',
    'dec', 'geocent_time', 'phase',  'psi' , 'luminosity_distance',
    'theta_jn', 'mass_1', 'mass_2'
  ]

  for name in name_order:
    print(name, injection_parameters[name])

  for key in priors:
    print(f"{key}: {priors[key]}")

  # Initialise the likelihood by passing in the interferometer data (ifos) and
  # the waveform generator
  likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    # time_marginalization=True,
    # phase_marginalization=True,
    # distance_marginalization=True
  )

  # Save inside and put outside state back
  inside_state = np.random.get_state()
  np.random.set_state(outside_state)

  def function(x):
    nonlocal inside_state, outside_state
    assert x.shape == (len(name_order),)
    x = x.copy()

    # Save outside and put inside state back
    outside_state = np.random.get_state()
    np.random.set_state(inside_state)

    for name in inject_parameters:
      likelihood.parameters[name] = injection_parameters[name]

    for name in index_by_name:
      value = float(x[index_by_name[name]])
      likelihood.parameters[name] = value

    log_prior = np.sum([
      priors[name].ln_prob(x[index_by_name[name]])
      for name in index_by_name
    ])

    if "mass_2" in name_order and "mass_1" in name_order:
      allowed = priors['mass_ratio'].prob(x[index_by_name["mass_2"]] /
                                          x[index_by_name["mass_1"]])
      if not allowed:
        log_prior = -np.inf

    log_likelihood = likelihood.log_likelihood()

    # Save inside and put outside state back
    inside_state = np.random.get_state()
    np.random.set_state(outside_state)

    log_joint = log_likelihood + log_prior
    return log_joint

  return function

def load_gw_injected():

  theta_space = BoundedSpace(
    lower_limit_vector=np.array([l for l, _ in bounds], dtype=np.float64),
    upper_limit_vector=np.array([u for _, u in bounds], dtype=np.float64)
  )

  log_joint_func = get_log_joint_fn()

  def f(x):
    if not is_point_within(theta_space, x):
      return -np.inf
    log_f = log_joint_func(x)
    return log_f

  def simulate(thetas):
    return np.array([f(theta) for theta in thetas])

  return Simulator(
    name="gw_injected_%s" % theta_space.ndims,
    domain=theta_space,
    simulate_thetas=simulate
  )
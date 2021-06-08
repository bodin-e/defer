
import numpy as np
from .bounded_space import sample_uniform

def sampler_in_domains(normalized_masses, domains):
  sample_indices = sampler_of_indices(
    masses=normalized_masses,
  )
  def sample_func(num_samples):
    indices = sample_indices(num_samples=num_samples)
    unique, counts = np.unique(indices, return_counts=True)
    unique_domains = domains[unique]
    samples = []
    for i, domain in enumerate(unique_domains):
      new_samples = sample_uniform(domain, counts[i]).tolist()
      samples.extend(new_samples)
    # Shuffle to avoid samples being ordered
    # implicitly via np.unique of indices
    np.random.shuffle(samples)
    samples = np.array(samples)
    return samples
  return sample_func

def sampler_of_indices(masses):

  # https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

  if np.abs(np.sum(masses) - 1) > 1e-3:
    raise ValueError("masses must be normalized")

  def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
      q[kk] = K * prob
      if q[kk] < 1.0:
        smaller.append(kk)
      else:
        larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
      small = smaller.pop()
      large = larger.pop()

      J[small] = large
      q[large] = q[large] - (1.0 - q[small])

      if q[large] < 1.0:
        smaller.append(large)
      else:
        larger.append(large)

    return J, q

  def alias_draw(J, q, num_samples):
    K = len(J)

    # Draw from the overall uniform mixture.
    kk = np.floor(np.random.rand(num_samples) * K).astype(np.int32)

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    keep_small_one = np.random.rand(num_samples) < q[kk]

    out = J[kk].copy()
    out[keep_small_one] = kk[keep_small_one]
    return out

  J, q = alias_setup(probs=masses)

  def sample_indices(num_samples):
    indices = alias_draw(J, q, num_samples=num_samples)
    return indices
  return sample_indices
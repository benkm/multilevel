#### Constructed when I removed weight_type attribute from the twopt_average
#### function

import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pdb
import os
from tqdm import tqdm

matplotlib.use('TkAgg')
# from density_calculator import *


def region_maker(L, s):
  """
    This function will return a function, which can be used to see which region
    of the lattice a given point is in.

    L : int, size of the lattice
    s : (3, ) tuple of ints, describing how the lattice is to be broken
      down in the directions (x1, ..., x3). For simlicity I have programmed this
      to only accept ints that divide L exactly e.g. powers of 2
  """
  dim = len(s)

  shape = (L, ) * dim
  regions = numpy.zeros(shape)

  for i in s:
    assert type(i) == int
    assert i >= 1
    assert L % i == 0

  # Size of sublattices in each direction
  size = L // numpy.array(s)

  # Use a recursive function to label the regions uniquely
  def label(x, n):
    if n == 0:
      return 0

    # Add 1 to distiguish regions from the boundary
    if n == dim:
      return label(x, n - 1) * s[n - 1] + x[n - 1] // size[n - 1] + 1

    else:
      return label(x, n - 1) * s[n - 1] + x[n - 1] // size[n - 1]

  if numpy.product(numpy.array(s)) == 1:
    def which_region(x, x2=None, x3=None):
      return 1

  else:
    def which_region(x):
      """
        This function returns the region of the lattice a coordinate is in

        x : (d, ) list of ints, representing the coordinate on the lattice. Can
          also be a single int, in the case that the other coordinates are included

        RETURNS :
        ---------
        region : int, label of region. Note that region 0 is reserved for the boundary.
      """
      for i in x:
        assert type(i) == int
        assert i < L and i >= 0
      
      # First check if the coordinate is on the boundary. If so return 0.
      for i in range(dim):
        if s[i] > 1:
          if x[i] % (L // s[i]) == 0:
             return 0

      # If the point isn't on the boundary then use the recursive function
      # label defined previously
      return label(x, dim)
  
  # Now use the function which_region to caclulate the region of each point
  regions = numpy.zeros(shape)

  indices = numpy.indices(shape)

  def cycle(regions, stored, dim):
    if dim == 1:
      for i in range(L):
        new_stored = stored + [i]
        regions[i] = which_region(new_stored)
      
      return regions

    else:
      for i in range(L):
        new_stored = stored + [i]
        regions[i] = cycle(regions[i], new_stored, dim - 1)
      
      return regions

  regions = cycle(regions, [], dim)

  return regions


def plot_spins(spins, ax):
  L = spins.shape[0]

  # create discrete colormap
  cmap = colors.ListedColormap(['red', 'blue'])
  bounds = [-2, 0, 2]
  norm = colors.BoundaryNorm(bounds, cmap.N)

  ax.imshow(spins, cmap=cmap, norm=norm)


def plot_energies(energies, ax):
  # create discrete colormap
  cmap = colors.ListedColormap(['red', 'orange', 'yellow', 'green', 'blue'])
  bounds = [-5, -3, -1, 1, 3, 5]
  norm = colors.BoundaryNorm(bounds, cmap.N)

  ax.imshow(energies, cmap=cmap, norm=norm)


def chessboard(L):
  board = numpy.zeros((L, L), dtype=int)
  for i in range(L):
    for j in range(L):
      board[i, j] = (i + j) % 2
  return board


def neighbour_total(spin_array):
  """
    This function takes an array of spins, and returns at each position in
    the lattice, the total of the neighbouring spins to that lattice position.

    INPUTS

    spin_array: numpy array object with entries of either 1 or (-1) as in
                initialise_problem function

    OUTPUTS

    total_neighbour_spin: an array of the same shape as the spin_array, but
                          at each lattice site is the total of the neighbouring
                          spins.
  """
  total_neighbour_spin = numpy.zeros(spin_array.shape)

  for dim in [0, 1]:
    total_neighbour_spin += numpy.roll(spin_array, 1, axis=dim)
    total_neighbour_spin += numpy.roll(spin_array, -1, axis=dim)

  return total_neighbour_spin


class lattice(object):
  def __init__(self, L, splitting, beta=1, spins=None):
    # L must be a mulitple of 2 for the chessboard method to be valid
    assert L % 2 == 0

    if spins is None:
      spins = numpy.random.randint(0, 2, size=(L, L)) * 2 - 1
    else:
      self.spins = spins

    # For reduced memory usage
    self.spins = numpy.array(spins, dtype=numpy.int8)
    self.chessboard = chessboard(L)
    self.regions = region_maker(L, splitting)
    self.beta = beta

    # To be clear this is the total energy associated with all interactions at
    # a point - to get the correct total energy density this value should be
    # halved.
    self.energies = - self.spins * neighbour_total(self.spins)
    self.L = L

    return None

  def plot_state(self):
    fig, axes = plt.subplots(1, 2)
    plot_spins(self.spins, axes[0])
    plot_energies(self.energies, axes[1])
    plt.show()

  def step(self):
    # Flip the white squares first
    delta_E = - self.energies * 2
    jump_condition = numpy.exp(- self.beta * delta_E) > numpy.random.rand(self.L, self.L)

    # The boundary has self.regions = 0, all other locations are non-boundary
    to_flip = numpy.logical_and(jump_condition,
                    numpy.logical_and(self.regions, self.chessboard))

    self.spins = numpy.where(to_flip, -self.spins, self.spins)
    self.energies = - self.spins * neighbour_total(self.spins)

    # Now flip the black squares
    delta_E = - self.energies * 2
    jump_condition = numpy.exp(- self.beta * delta_E) > numpy.random.rand(self.L, self.L)

    to_flip = numpy.logical_and(jump_condition,
                    numpy.logical_and(self.regions, numpy.logical_not(self.chessboard)))

    self.spins = numpy.where(to_flip, -self.spins, self.spins)
    self.energies = - self.spins * neighbour_total(self.spins)


def generate_states_1_1(L, beta, N, step, initial, rerun=False, initial_spins=False):
  directory = f"../data/ising/L{L}/b{beta:.3f}/"
  filename = f"L{L}_b{beta:.3f}_N{N}_init{initial}_step{step}_spins_1_1.npy"

  def run():
    print(f"Generating states for : L = {L}, N = {N}, beta = {beta:.3f}, splitting = (1, 1)")

    data = numpy.zeros((N, L, L), dtype=numpy.int8)
    
    if initial_spins is not False:
      x = lattice(L, (1, 1), beta=beta, spins=initial_spins)
    
    else:
      x = lattice(L, (1, 1), beta=beta)

    for j in range(initial):
      x.step()

    count = 0
    for i in tqdm(range((N - 1) * step + 1)):
      if(i % step == 0):
        data[count] = x.spins
        count += 1

      x.step()

    if not os.path.isdir(directory):
      os.makedirs(directory)
    
    numpy.save(f"{directory}{filename}", data)

    return data

  if not rerun:
    try:
      data = numpy.load(f"{directory}{filename}")
   
    except:
      data = run()

  else:
    data = run()
  
  return data


def generate_states_splitting(L, beta, N, M, Mstep, splitting, source_file_N, source_file_init, Nstep=1, source_file_step=True, rerun=False):
  """
    source_file : the name of the file containing the starting boundary configurations
    source_file_step : if true set to the same as Mstep
    Nstep : Step in units of the source file step in the N direction, e.g. if
      the source file keeps every 10th lattice state and Nstep is 2, then every
      other saved config is used as a starting point for the sublattice simulation,
      e.g. every 20th iteraction of the source algorithm.
    
    note : Loaded in configs start from first config of input sample. I see no
      reason anyone would want to change this.
  """
  i, j = splitting[0], splitting[1]

  if source_file_step:
    source_file_step = Mstep

  directory = f"../data/ising/L{L}/b{beta:.3f}/"
  filename = f"L{L}_b{beta:.3f}_N{N}_M{M}_Mstep{Mstep}_sourceN{source_file_N}_sourceinit{source_file_init}_sourcestep{source_file_step}_Nstep{Nstep}_spins_{i}_{j}.npy"

  def run():
    # The data has to use an input file of (1, 1) splitting data
    data = numpy.zeros((N, M, L, L), dtype=numpy.int8)
    source_file = f"L{L}_b{beta:.3f}_N{source_file_N}_init{source_file_init}_step{source_file_step}_spins_1_1.npy"

    try:
      numpy.load(f"{directory}{source_file}")

    except FileNotFoundError:
      print("No source configuration files - generating these first")
      generate_states_1_1(L, beta, N, source_file_step, source_file_init, rerun=False)
      source_file = f"L{L}_b{beta:.3f}_N{N}_init{source_file_init}_step{source_file_step}_spins_1_1.npy"

    # Load in N configs from source file in steps of Nstep
    source_data = numpy.load(f"{directory}{source_file}")[0:Nstep * N:Nstep]

    print(f"Generating states for : L = {L}, N = {N}, M = {M}, beta = {beta:.3f}, splitting = ({i}, {j})")

    for k in tqdm(range(N)):
      x = lattice(L, splitting, beta=beta, spins=source_data[k])
      
      for l in tqdm(range((M - 1) * Mstep + 1)):
        if l % Mstep == 0:
          data[k, l // Mstep] = x.spins

        x.step()
  
    numpy.save(f"{directory}{filename}", data)

    return data

  if not rerun:
    try:
      data = numpy.load(f"{directory}{filename}")
   
    except:
      data = run()

  else:
    data = run()

  return data


def normalize_weights(weights, regions):
  """
    This function normalizes weights (type = 1), so that the average weight in
    the boundary region is 1

    weights : (L, L) floats
    regions : (L, L) ints where entries of 0 are at the boundary
  """
  # Find the total size of the boundary
  size = numpy.sum(regions == 0)

  # Find the total of the weights in this region
  wegiht_total = numpy.sum(weights * (regions == 0))

  # Therefore the average weight is given by weight_total/sum
  av = wegiht_total / size

  # Rescale the weights so that the boundary is 1
  weights = weights / av

  return weights


def twopt_site_by_site(spins, regions, delta):
  """
    Utility function
    Does a site by site twopt mean, before the weighting stage. Only recieves
    data for a single N value
  """
  M = spins.shape[0]
  L = regions.shape[0]
  d = len(regions.shape)
  assert regions.shape == spins.shape[1:]

  # Bring the points displaced by delta to the point x
  yspins = spins
  yregions = regions

  for i in range(d):
    yspins = numpy.roll(yspins, -delta[i], axis=i + 1)

    # Do the same with regions and weights
    yregions = numpy.roll(yregions, -delta[i], axis=i)

  # Perform an outer product over the configuration axis
  # This produces an array of shape (M, M, L, ..., L)
  combinatoric = numpy.einsum('j...,k...->jk...', spins, yspins)

  # BE CAREFUL here! We've made combinatoric states. However, this isn't physically
  # valid for twopt functions between points in the same region. This isn't valid
  # within region. It's okay on the boundary because these states are automatically
  # constant along the M-axis.
  within_region = (spins * yspins).reshape((M, 1) + spins.shape[1:]).repeat(M, axis=1)

  # numpy where automatically applies the condition to the highest axes of the
  # array, in this case the (L, L) axes of (M, M, L, L)
  outer_product = numpy.where(regions != yregions, combinatoric, within_region)

  # Average over the 2 outer product directions and over the N axis
  config_mean = numpy.mean(outer_product, axis=(0, 1))

  return config_mean


def twopt_average(spins, regions, weights, delta, full_average=True):
  """
    Here we are finding the real space twopt function of many spin configurations,
    which are assumed to share the same boundary. This function also gives the
    sample standard deviation 

    spins : (N, M, L, ..., L) numpy array of floats, spins[i] is the ith spin configuration
        from a group of spins holding
    regions : (L, ..., L) numpy array of ints, which label the regions of the lattice
        according to the splitting
    delta : (d, ) tuple, list or array of ints in [0, L). This represents the 
        separation vector between two points
    weights : (L, ..., L) array of floats. These should be normalized so that the
      weights on the boundary-boundary twopt functions are 1, or average to 1.

    note :
      > It's expected input weights are twopt type
      > It's expected input weights are normalized
      > It's expected input weights are capped

    N is the number of boundary updates
    where M is the number of sublattice samples
    L is the size of the lattice
    d is the number of dimensions
  """
  N = spins.shape[0]
  M = spins.shape[1]
  L = regions.shape[0]
  d = len(regions.shape)
  assert regions.shape == spins.shape[2:]

  # It is assumed that the boundary weights are non-zero!
  assert numpy.sum((regions == 0) * weights) != 0

  results = numpy.zeros(N)

  for i in tqdm(range(N)):
    config_mean = twopt_site_by_site(spins[i], regions, delta)
    
    results[i] = numpy.average(config_mean, weights=weights)

  mean, std = numpy.mean(results), numpy.std(results, ddof=1) / numpy.sqrt(N)
  
  if full_average:
   return numpy.array([mean, std])
  
  else:
    return results


def get_twopt_weights(spins, regions, delta):
  """
    Uses the standard deviation to estimate twopt weights
  """
  N = spins.shape[0]
  M = spins.shape[1]
  L = spins.shape[2]
  assert regions.shape == spins.shape[2:]
  d = len(regions.shape)

  twopt_weights = numpy.zeros(regions.shape)

  results = numpy.zeros((N, ) + regions.shape)

  for i in range(N):
    results[i] = twopt_site_by_site(spins[i], regions, delta)
  
  twopt_weights = 1 / (numpy.std(results, axis=0, ddof=1) ** 2)

  # Remove any nan's due to zero standard deviation
  # They should be weighted the same as the boundary in that case
  twopt_weights = numpy.where(numpy.isnan(twopt_weights), 1, twopt_weights)
  twopt_weights = numpy.where(numpy.isinf(twopt_weights), 1, twopt_weights)

  # Now normalize the weights
  twopt_weights = normalize_twopt_weights(twopt_weights, regions, delta)
  
  # Now cap the weights
  twopt_weights = cap_twopt_weights(twopt_weights, regions, delta, M)

  return twopt_weights


def get_onept_weights(spins, regions, cap=True):
  M = spins.shape[1]

  # Average over the M axis first
  spins_aved = numpy.mean(spins, axis=1)

  # Then find the standard deviation over the N axis
  std = numpy.std(spins_aved, axis=0, ddof=1)

  # In the case of the boundary having zero values set them to 1 / M, as these
  # sites effectively have the same amount of variation as the boundary
  non_boundary_av = numpy.sum((regions !=0) * std) / numpy.count_nonzero(regions)
  std = numpy.where(abs(std) > 10 ** -10, std, non_boundary_av * numpy.sqrt(M))

  # Use inverse varience method
  weights = 1 / (std ** 2)

  # Normalize and set boundary to 1
  weights = normalize_weights(weights, regions)
  weights = numpy.where(regions, weights, 1)

  # Cap is really here just for debugging, in reality it should always be True
  if cap:
    M = spins.shape[1]
    weights = numpy.minimum(M, weights)
    weights = numpy.maximum(1, weights)

  # Remove any nan's due to zero standard deviation
  # They should be weighted the same as the boundary in that case
  weights = numpy.where(numpy.isnan(weights), 1, weights)
  weights = numpy.where(numpy.isinf(weights), 1, weights)

  return weights


def combine_onept_weights(weights, regions, delta):
  """
    Combines onept weights into twopt weights. The weights are already assumed
    to be defined such that on the boundary they are 1 and in the sublattices
    they are between 1 and M (inclusive)
  """
  yweights = weights
  yregions = regions
  d = len(regions.shape)

  for i in range(d):
    yweights = numpy.roll(yweights, -delta[i], axis=i)
    yregions = numpy.roll(yregions, -delta[i], axis=i)

  new_weights = numpy.where(numpy.logical_and((regions * yregions) != 0, regions != yregions),
                          weights * yweights,
                          numpy.maximum(weights, yweights))

  return new_weights


def normalize_twopt_weights(weights, regions, delta):
  """
    Normalizes the weights so that the boundary-boundary twopt sites have an
    average of 1
  """
  yweights = weights 
  yregions = regions
  d = len(regions.shape)

  for i in range(d):
    yweights = numpy.roll(yweights, -delta[i], axis=i)
    yregions = numpy.roll(yregions, -delta[i], axis=i)

  boundary_boundary = numpy.logical_not(numpy.logical_or(regions, yregions))
  boundary_boundary_size = numpy.count_nonzero(boundary_boundary)

  boundary_av = numpy.sum(boundary_boundary * weights) / boundary_boundary_size

  return weights / boundary_av


def cap_twopt_weights(weights, regions, delta, M):
  """
    Takes twopt weights that are assumed to be normalized, as by the normalize_twopt_weights
    function, and caps them so boundary_boundary = 1, 1 < boundary_other < M,
    1 < different_regions < M^2
  """
  yregions = regions
  d = len(regions.shape)

  for i in range(d):
    yregions = numpy.roll(yregions, -delta[i], axis=i)

  # Cap lower end
  weights = numpy.maximum(weights, 1)

  # Cap the upper end
  weights = numpy.where(regions,
              numpy.where(yregions,
                numpy.where(regions != yregions,
                  numpy.minimum(M ** 2, weights),  # different non-boundary
                  numpy.minimum(M, weights) # same non-boundary
                ),
                  numpy.minimum(M, weights) # y on boundary
              ),
              numpy.where(yregions,
                numpy.minimum(M, weights), # x on boundary
                numpy.minimum(1, weights)  # both on boundary
              )
            )

  return weights


def use_symmetry(weights, splitting):
  """
    Use the symmetry group of sublattices to average onept weights, to smooth
    out fluctuations due to the small size of the training set.
  """
  L = weights.shape[0]
  size = L//2

  weights_new = numpy.ones(weights.shape)

  assert L%2 == 0
  assert size%2 == 0
  subregions = numpy.zeros(splitting + (L//splitting[0] -1, L//splitting[1] - 1))
  for i in range(splitting[0]):
    for j in range(splitting[1]):
      lower_i = int(numpy.rint(L//splitting[0] * i + 1))
      upper_i = int(numpy.rint(L//splitting[0] * (i + 1)))
      lower_j = int(numpy.rint(L//splitting[1] * j + 1))
      upper_j = int(numpy.rint(L//splitting[1] * (j + 1)))
      subregions[i, j] = weights[lower_i: upper_i, lower_j: upper_j]

  # Average over reflections first
  for i in range(splitting[0]):
    for j in range(splitting[1]):
      x = subregions[i, j]
      subregions[i, j] = (x + x[::-1] + x[:, ::-1] + x[::-1, ::-1]) / 4

  # Average over sublattice exchange symmetry
  subregions_sum = numpy.zeros(subregions[i, j].shape) # All subregions are the same shape
  for i in range(splitting[0]):
    for j in range(splitting[1]):
      subregions_sum += subregions[i, j]
  
  result = subregions_sum / 4

  for i in range(splitting[0]):
    for j in range(splitting[1]):
      lower_i = int(numpy.rint(L//splitting[0] * i + 1))
      upper_i = int(numpy.rint(L//splitting[0] * (i + 1)))
      lower_j = int(numpy.rint(L//splitting[1] * j + 1))
      upper_j = int(numpy.rint(L//splitting[1] * (j + 1)))
      weights_new[lower_i: upper_i, lower_j: upper_j] = result

  return weights_new


def use_exchange_sym(weights, splitting):
  L = weights.shape[0]
  weights_new = numpy.zeros(weights.shape)

  for i in range(splitting[0]):
    for j in range(splitting[1]):
      piece = numpy.roll(weights, L//splitting[0] * i, axis=0)
      piece = numpy.roll(piece, L//splitting[1] * j, axis=1)

      weights_new += piece
  
  weights_new = weights_new / (splitting[0] * splitting[1])

  return weights_new

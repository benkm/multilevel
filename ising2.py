import numpy
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pdb
import os
from tqdm import tqdm
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

  # If there is no s then there is only one region with no boudary
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


def generate_states_1_1(L, beta, N, step, initial, rerun=False):
  directory = f"../data/ising/L{L}/b{beta}/"
  filename = f"b{beta}_L{L}_N{N}_init{initial}_step{step}_spins_1_1.npy"

  def run():
    print(f"Generating states for : L = {L}, beta = {beta}, splitting = (1, 1)")

    data = numpy.zeros((N, L, L), dtype=numpy.int8)
    
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

  if not rerun:
    try:
      numpy.load(f"{directory}{filename}")
   
    except:
      run()

  else:
    run()


def generate_states_splitting(L, beta, N, M, step, splitting, source_file_N, source_file_init, rerun=False):
  """
    source_file : the name of the file containing the starting boundary configurations
  """
  i, j = splitting[0], splitting[1]

  directory = f"../data/ising/L{L}/b{beta}/"
  filename = f"b{beta}_L{L}_N{N}_M{M}_step{step}_spins_{i}_{j}.npy"

  def run():
    # The data has to use an input file of (1, 1) splitting data
    data = numpy.zeros((N, M, L, L), dtype=numpy.int8)

    try:
      source_file = f"b{beta}_L{L}_N{source_file_N}_init{source_file_init}_step{step}_spins_1_1.npy"
      numpy.load(f"{directory}{source_file}")

    except FileNotFoundError:
      print("No source configuration files - generating these first")
      generate_states_1_1(L, beta, N, step, initial, rerun=False)
      source_file = f"b{beta}_L{L}_N{N}_init{initial}_step{step}_spins_1_1.npy"

    source_data = numpy.load(f"{directory}{source_file}")[0:N]

    for i in range(N):
      x = lattice(L, splitting, beta=beta, spins=source_data[i])
      
      for j in range((M - 1) * step + 1):
        if j % step == 0:
          data[i, j // step] = x.spins

        x.step()
  
    numpy.save(f"{directory}{filename}", data)

  if not rerun:
    try:
      numpy.load(f"{directory}{filename}")
   
    except:
      run()

  else:
    run()


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


def twopt_average(spins, regions, weights, delta, weight_type=1):
  """
    Here we are finding the real space twopt function of many spin configurations,
    which are assumed to share the same boundary.

    spins : (N, M, L, ..., L) numpy array of floats, spins[i] is the ith spin configuration
        from a group of spins holding
    regions : (L, ..., L) numpy array of ints, which label the regions of the lattice
        according to the splitting
    delta : (d, ) tuple, list or array of ints in [0, L). This represents the 
        separation vector between two points
    weights : (L, ..., L) array of floats. These should be normalized so that the
      weights on the boundary are 1, or average to 1.
    weight_type : int, if 1 then the weights are 'onept' site by site weights.
      If 2 then these are the twopt weights, ready to be used directly.

    N is the number of boundary updates
    where M is the number of sublattice samples
    L is the size of the lattice
    d is the number of dimensions
  """
  N = spins.shape[0]
  M = spins.shape[1]
  L = regions.shape[0]
  d = len(regions.shape)
  assert spins.shape[2] == L
  assert len(spins.shape) == d + 2

  estimators = numpy.zeros((L, ) * d)

  # First normalize the weights
  # It is assumed that the boundary weights are non-zero!
  assert numpy.sum((regions == 0) * weights) != 0

  weights = normalize_weights(weights, regions)

  # Bring the points displaced by delta to the point x
  yspins = spins
  yregions = regions
  yweights = weights
  for i in range(d):
    yspins = numpy.roll(yspins, -delta[i], axis=i + 2)

    # Do the same with regions and weights
    yregions = numpy.roll(yregions, -delta[i], axis=i)
    yweights = numpy.roll(yweights, -delta[i], axis=i)
  
  # Perform an outer product over the configuration axis
  # This produces an array of shape (N, M, M, L, ..., L)
  combinatoric = numpy.einsum('ij...,ik...->ijk...', spins, yspins)

  # BE CAREFUL here! We've made combinatoric states. However, this isn't physically
  # valid for twopt functions between points in the same region. This isn't valid
  # within region. It's okay on the boundary because these states are automatically
  # constant along the M-axis.
  within_region = (spins * yspins).reshape((N, M, 1) + spins.shape[2:]).repeat(M, axis=2)

  outer_product = numpy.where(regions != yregions, combinatoric, within_region)

  # Average over the 2 outer product directions and over the N axis
  config_mean = numpy.mean(outer_product, axis=(0, 1, 2))

  if weight_type == 1:
    ## Turn the site by site weights into 2 point correlator weights

    # Only in the case of two different non-boundary regions do we product the weights together.
    new_weights = numpy.where(numpy.logical_and(regions * yregions != 0, regions != yregions),
                          weights * yweights,
                          numpy.maximum(weights, yweights))
  elif weight_type == 2:
    new_weights = weights

  # Perform a weighted average over the spatial dimensions
  
  estimator = numpy.average(config_mean, weights=new_weights)

  # pdb.set_trace()

  return estimator

def defunct():
  return 0
  # def twopt_combo_bootstrap(spins, regions, delta, boot_samples, return_weights=False):
  #   """
  #     Here we are finding the real space twopt function of many spin configurations,
  #     which are assumed to share the same boundary.

  #     spins : (N, M, L, L) numpy array of floats, spins[i] is the ith spin configuration
  #         from a group of spins holding
  #     regions : (L, L) numpy array of ints, which label the regions of the lattice
  #         according to the splitting
  #     delta : (2, ) tuple, list or array of ints in [0, L). This represents the 
  #         separation vector between two points
  #     boot_samples : number fo bootstrap samples used.
  #     return_weights : Bool, if true then the weights are also returned, e.g. for
  #       plotting etc.

  #     N is the number of boundary updates
  #     where M is the number of sublattice samples
  #     L is the size of the lattice
  #   """
  #   N = spins.shape[0]
  #   M = spins.shape[1]
  #   L = regions.shape[0]
  #   assert spins.shape[2] == L

  #   estimators = numpy.zeros((L, L))

  #   # Bring the points displaced by delta to the point x
  #   yspins = numpy.roll(numpy.roll(spins, -delta[0], axis=2), -delta[1], axis=3)

  #   # Do the same for the regions array
  #   yregions = numpy.roll(numpy.roll(regions, -delta[0], axis=0), -delta[1], axis=1)

  #   # Perform an outer product over the configuration axis
  #   # This produces an array of shape (N, M, M, L, L)
  #   outer_product = numpy.einsum('ij...,ik...->ijk...', spins, yspins)

  #   # Average over the M^2 sublattice configs only.
  #   sub_average = numpy.mean(outer_product, axis=(1, 2))

  #   # Perform a bootstrap over the remaining N degrees of freedom 
  #   random_indices = numpy.random.randint(0, N, size=(boot_samples, N))

  #   results = numpy.zeros((no_configs, L, L))
  #   for sample in range(boot_samples):
  #     data = sub_average[random_indices[sample]]
  #     results[sample] = numpy.mean(data, axis=0)
    
  #   # We then get 2 (L, L) arrays, with bootstrap means and stds for each lattice
  #   # site
  #   means = numpy.mean(results, axis=0)
  #   stds = numpy.std(results, axis=0)

  #   # Use the inverse varience weighting method
  #   weights = 1 / (stds ** 2)

  #   if return_weights:
  #     return numpy.average(means, weights=weights), weights
    
  #   return numpy.average(means, weights=weights)


  # def non_boundary(L, splitting):
  #   grid = numpy.ones((L, L))

  #   if splitting[0] != 1:
  #     step = L // splitting[0]
  #     for i in range(splitting[0]):
  #       grid[i * step, :] = 0

  #   if splitting[1] != 1:
  #     step = L // splitting[1]
  #     for i in range(splitting[1]):
  #       grid[:, i * step] = 0

  #   return grid
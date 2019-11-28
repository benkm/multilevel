from ising2 import *


def test_region_maker():
  L = 8
  # Simplest case
  regions = region_maker(L, (1, ))

  assert numpy.array_equal(regions, numpy.array([1] * L))

  # Split the 1D lattice
  regions = region_maker(L, (2, ))
  assert numpy.array_equal(regions, numpy.array([0, 1, 1, 1, 0, 2, 2, 2]))

  # Let's try a 2D situation
  regions = region_maker(L, (2, 2))
  assert numpy.array_equal(regions, numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 1, 1, 1, 0, 2, 2, 2],
                                                 [0, 1, 1, 1, 0, 2, 2, 2],
                                                 [0, 1, 1, 1, 0, 2, 2, 2],
                                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 3, 3, 3, 0, 4, 4, 4],
                                                 [0, 3, 3, 3, 0, 4, 4, 4],
                                                 [0, 3, 3, 3, 0, 4, 4, 4]]))


def test_chessboard():
  my_chessboard = chessboard(4)

  assert numpy.array_equal(my_chessboard, numpy.array([[0, 1, 0, 1],
                                                       [1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [1, 0, 1, 0]]))


def test_neighbour_total():
    test_array = numpy.array([[14, 13, 2],
                              [16, 12, 11],
                              [17, 8, 15]])

    test_total = neighbour_total(test_array)
    
    # The correct array was found using a calculator
    correct = numpy.array([[48, 36, 53],
                           [54, 48, 45],
                           [53, 57, 38]])

    assert numpy.array_equal(test_total, correct)


def test_lattice_init():
  L = 8
  x = lattice(L, (1, 2))

  assert x.spins.dtype == 'int8'
  assert x.spins.shape == (L, L)
  assert x.energies.shape == (L, L)


def test_energies():
  spins = numpy.array([[1, 1], [1, -1]])
  x = lattice(spins.shape[0], (1, 1), spins=spins)

  # Check the energies have been calculated correctly
  assert numpy.array_equal(x.energies, numpy.array([[-4, 0], [0, 4]]))


def test_generate_states_1_1():
  L = 8
  beta = 0.3
  N = 100
  step = 10
  initial = 100
  generate_states_1_1(L, beta, N, step, initial, rerun=True)

  # Load in the data
  directory = f"../data/ising/L{L}/b{beta}/"
  filename = f"b{beta}_L{L}_N{N}_init{initial}_step{step}_spins_1_1.npy"

  data = numpy.load(f"{directory}{filename}")

  assert data.shape == (N, L, L)
  assert data.dtype == 'int8'

  # Check rerun works
  generate_states_1_1(L, beta, N, step, initial, rerun=True)

  data2 = numpy.load(f"{directory}{filename}")

  assert not numpy.array_equal(data2, data)

  # Check that a rerun==False doesn't cause the data to be reproduced
  generate_states_1_1(L, beta, N, step, initial)

  data3 = numpy.load(f"{directory}{filename}")

  assert numpy.array_equal(data2, data3)


def test_generate_states_1_2():
  L = 8
  beta = 0.3
  source_file_N = 100
  N = 5
  M = 20
  step = 10
  source_file_init = 100
  i, j = 1, 2
  splitting = (i, j)

  directory = f"../data/ising/L{L}/b{beta}/"
  filename = f"b{beta}_L{L}_N{N}_M{M}_step{step}_spins_{i}_{j}.npy"
  source_file = f"b{beta}_L{L}_N{source_file_N}_init{source_file_init}_step{step}_spins_1_1.npy"

  generate_states_splitting(L, beta, N, M, step, splitting, source_file_N, source_file_N, rerun=True)

  sublattice_data = numpy.load(f"{directory}{filename}")
  source_data = numpy.load(f"{directory}{source_file}")

  assert sublattice_data.shape == (N, M, L, L)

  # Check all data slots were filled
  assert numpy.sum(sublattice_data[-1] ** 2) > 0

  # Check the starting points of the sublattice data are the same as the source
  assert numpy.array_equal(sublattice_data[:, 0], source_data[0: N])

  # Check that the boundaries remain unchanged
  assert numpy.sum((sublattice_data[..., 0] - sublattice_data[:, 0, :, 0].reshape(N, 1, L).repeat(M, axis = 1)) ** 2) == 0
  assert numpy.sum((sublattice_data[..., 4] - sublattice_data[:, 0, :, 4].reshape(N, 1, L).repeat(M, axis = 1)) ** 2) == 0

  # Check other parts of the lattice are changed
  assert numpy.sum((sublattice_data[..., 3] - sublattice_data[:, 0, :, 3].reshape(N, 1, L).repeat(M, axis = 1)) ** 2) != 0


def test_normalize_weights():
  # Try if all weights are one
  L = 4

  weights = numpy.ones((L, L))
  regions = region_maker(L, (2, 2))

  norm_weights = normalize_weights(weights, regions)

  assert numpy.array_equal(norm_weights, weights)

  # Try if all the weights are a constant not equal to one
  weights = numpy.ones((L, L)) * 2

  norm_weights = normalize_weights(weights, regions)

  assert numpy.array_equal(norm_weights, numpy.ones((L, L)))

  # Test a more realistic situation where the weights on the boundary are different
  weights = numpy.array([[2, 2, 2, 2],
                         [2, 5, 2, 5],
                         [2, 2, 2, 2],
                         [2, 5, 2, 5]])

  norm_weights = normalize_weights(weights, regions)

  assert numpy.array_equal(norm_weights, weights / 2)

  # Let's try a slightly more difficult case
  weights = numpy.array([[1, 1, 1, 1],
                         [1, 5, 1, 5],
                         [2, 2, 2, 2],
                         [2, 5, 2, 5]])

  norm_weights = normalize_weights(weights, regions)

  assert numpy.array_equal(norm_weights, weights / 1.5)


def test_twopt_average_basics():
  L = 8
  N = 20
  M = 20
  regions = region_maker(L, (2, 2))
  spins = numpy.zeros((N, M, L, L))
  for i in range(N):
    for j in range(M):
      spins[i, j] = regions

  # First check the simple twopt weights
  weights = numpy.ones((L, L))
  delta = (0, 0)
  estimator, std = twopt_average(spins, regions, weights, delta, weight_type=2)

  # Compare to hand caculation
  assert numpy.abs(estimator - numpy.mean(regions ** 2)) < 10 ** -10

  # I think that for any way of rolling the array the result will be lower
  for i in range(1, L):
    for j in range(1, L):
      delta = (i, j)

      estimator_2, std_2 = twopt_average(spins, regions, weights, delta, weight_type=2)

      assert estimator_2 < estimator

  # Check that for a constant spin matrix, the result comes out the same regardless
  x = numpy.random.rand()
  spins = numpy.ones((N, M, L, L)) * x

  for i in range(L):
    for j in range(L):
      for weight_type in [1, 2]:
        delta = (i, j)

        weights = numpy.random.rand(L, L)
        estimator, std = twopt_average(spins, regions, weights, delta, weight_type=weight_type)

        assert numpy.abs(estimator - x ** 2) < 10 ** -10


def test_twopt_average_simplest_example():
  L = 8
  N = 20
  M = 20
  regions = region_maker(L, (2, 2))
  # Let's check that given knowledge of the error by lattice site, we can achieve
  # a reduced error through use of weights
  weights = numpy.where(regions,
                        0,
                        1)

  weights_ext = weights.reshape((1, 1, L, L)).repeat(N, axis=0).repeat(M, axis=1)                        

  N_randomness = numpy.random.normal(0, 1, (N, 1, L, L)).repeat(M, axis=1)
  M_randomness = numpy.random.normal(0, 1, (N, M, L, L))

  # Add the N and M randomness. Think of a point with weight 1 as being on the boundary or
  # completely determined by the boundary, while a point with weight 0 is completely independent
  # of the boundary - e.g. it's a correlation coefficient.
  spins = numpy.zeros((N, M, L, L)) + N_randomness * weights_ext + M_randomness * numpy.sqrt(1 - weights_ext ** 2)

  # Let's check that the boundarys indeed have a larger standard dev
  means = numpy.mean(numpy.mean(spins, axis=1), axis=0)
  stds = numpy.std(numpy.mean(spins, axis=1), axis=0, ddof=1)

  boundary = regions == 0
  non_boundary = regions != 0

  boundary_av_std = numpy.sum(boundary * stds) / numpy.sum(boundary)
  non_boundary_av_std = numpy.sum(non_boundary * stds) / numpy.sum(non_boundary)

  # The boundary std larger than the non_boundary
  assert boundary_av_std > non_boundary_av_std

  delta = (0, 0)
  # naiive method
  mean_1, std_1 = twopt_average(spins, regions, numpy.ones((L, L)), delta, weight_type=2)

  # Higher weights have more varience in the M direction
  mean_2, std_2 = twopt_average(spins, regions, 1 / (weights ** 2 * (1 - 1 / M) + 1 / M), delta, weight_type=1)

  # Check that knowledge of the weights helped us choose a better estimator
  assert std_2 < std_1

  # Check that the answers are correct to 5 sigma
  assert mean_1 - 5 * std_1 < 1
  assert mean_1 + 5 * std_1 > 1
  assert mean_2 - 5 * std_2 < 1
  assert mean_2 + 5 * std_2 > 1

  # Now, if we have a larger delta, then even more benefit can come from the combinatorics,
  # lets check this
  delta = (4, 4)

  # naiive method
  mean_3, std_3 = twopt_average(spins, regions, numpy.ones((L, L)), delta, weight_type=2)

  # Higher weights have more varience in the M direction
  mean_4, std_4 = twopt_average(spins, regions, 1 / (weights ** 2 * (1 - 1 / M) + 1 / M), delta, weight_type=1)

  # Check that knowledge of the weights helped us choose a better estimator
  assert std_2 < std_1
  # Check that the improvement is better than previously
  assert std_4 / std_3 < std_2 / std_1

  # Check the answers are correct to 5 sigma
  assert mean_3 - 5 * std_3 < 0
  assert mean_3 + 5 * std_3 > 0
  assert mean_4 - 5 * std_4 < 0
  assert mean_4 + 5 * std_4 > 0


def test_twopt_average_next_simplest_example():
  L = 8
  N = 20
  M = 20
  regions = region_maker(L, (2, 2))
  # Let's check that given knowledge of the error by lattice site, we can achieve
  # a reduced error through use of weights
  weights = numpy.where(regions,
                        numpy.random.rand(L, L),
                        1)

  weights_ext = weights.reshape((1, 1, L, L)).repeat(N, axis=0).repeat(M, axis=1)                        

  N_randomness = numpy.random.normal(0, 1, (N, 1, L, L)).repeat(M, axis=1)
  M_randomness = numpy.random.normal(0, 1, (N, M, L, L))

  # Add the N and M randomness. Think of a point with weight 0 as being on the boundary or
  # completely determined by the boundary, while a point with weight 1 is completely independent
  # of the boundary
  spins = numpy.zeros((N, M, L, L)) + N_randomness * weights_ext + M_randomness * numpy.sqrt(1 - weights_ext ** 2)

  # Let's check that the boundarys indeed have a larger standard dev
  means = numpy.mean(numpy.mean(spins, axis=1), axis=0)
  stds = numpy.std(numpy.mean(spins, axis=1), axis=0, ddof=1)

  boundary = regions == 0
  non_boundary = regions != 0

  boundary_av_std = numpy.sum(boundary * stds) / numpy.sum(boundary)
  non_boundary_av_std = numpy.sum(non_boundary * stds) / numpy.sum(non_boundary)

  # The boundary std larger than the non_boundary
  assert boundary_av_std > non_boundary_av_std

  delta = (0, 0)
  # naiive method
  mean_1, std_1 = twopt_average(spins, regions, numpy.ones((L, L)), delta, weight_type=2)

  # Higher weights have more varience in the M direction
  mean_2, std_2 = twopt_average(spins, regions, 1 / (weights ** 2 * (1 - 1 / M) + 1 / M), delta, weight_type=1)

  # Check that knowledge of the weights helped us choose a better estimator
  assert std_2 < std_1

  # Check that the answers are correct to 5 sigma
  assert mean_1 - 5 * std_1 < 1
  assert mean_1 + 5 * std_1 > 1
  assert mean_2 - 5 * std_2 < 1
  assert mean_2 + 5 * std_2 > 1

  # Now, if we have a larger delta, then even more benefit can come from the combinatorics,
  # lets check this
  delta = (4, 4)

  # naiive method
  mean_3, std_3 = twopt_average(spins, regions, numpy.ones((L, L)), delta, weight_type=2)

  # Higher weights have more varience in the M direction
  mean_4, std_4 = twopt_average(spins, regions, 1 / (weights ** 2 * (1 - 1 / M) + 1 / M), delta, weight_type=1)

  # Check that knowledge of the weights helped us choose a better estimator
  assert std_2 < std_1
  # Check that the improvement is better than previously
  assert std_4 / std_3 < std_2 / std_1

  # Check the answers are correct to 5 sigma
  assert mean_3 - 5 * std_3 < 0
  assert mean_3 + 5 * std_3 > 0
  assert mean_4 - 5 * std_4 < 0
  assert mean_4 + 5 * std_4 > 0


def test_twopt_weights():
  L = 8
  N = 20
  M = 20
  no_samples = 100
  regions = region_maker(L, (2, 2))
  spin_fraction = 1 / 4 # Fraction of spins used to find weights

  # Let's check that given knowledge of the error by lattice site, we can achieve
  # a reduced error through use of weights
  weights = numpy.where(regions,
                        numpy.random.rand(L, L),
                        1)

  weights_ext = weights.reshape((1, 1, L, L)).repeat(N, axis=0).repeat(M, axis=1)                        

  N_randomness = numpy.random.normal(0, 1, (N, 1, L, L)).repeat(M, axis=1)
  M_randomness = numpy.random.normal(0, 1, (N, M, L, L))

  # Add the N and M randomness. Think of a point with weight 0 as being on the boundary or
  # completely determined by the boundary, while a point with weight 1 is completely independent
  # of the boundary
  spins = numpy.zeros((N, M, L, L)) + N_randomness * weights_ext + M_randomness * numpy.sqrt(1 - weights_ext ** 2)

  # Let's check that the boundarys indeed have a larger standard dev
  means = numpy.mean(numpy.mean(spins, axis=1), axis=0)
  stds = numpy.std(numpy.mean(spins, axis=1), axis=0, ddof=1)

  boundary = regions == 0
  non_boundary = regions != 0

  boundary_av_std = numpy.sum(boundary * stds) / numpy.sum(boundary)
  non_boundary_av_std = numpy.sum(non_boundary * stds) / numpy.sum(non_boundary)

  # The boundary std larger than the non_boundary
  assert boundary_av_std > non_boundary_av_std

  delta = (0, 0)

  # Use the first spin fraction percent of the spins to obtain weights
  twopt_weights = get_twopt_weights(spins[:int(spin_fraction * N)], regions, delta)

  # naiive method - use all the spins
  mean_1, std_1 = twopt_average(spins, regions, numpy.ones((L, L)), delta, weight_type=2)

  # Higher weights have more varience in the M direction
  # Test with rest of the spins
  mean_2, std_2 = twopt_average(spins[int(spin_fraction * N):], regions, twopt_weights, delta, weight_type=2)

  # Check that knowledge of the weights helped us choose a better estimator
  assert std_2 < std_1

  # Check that the answers are correct to 5 sigma
  assert mean_1 - 5 * std_1 < 1
  assert mean_1 + 5 * std_1 > 1
  assert mean_2 - 5 * std_2 < 1
  assert mean_2 + 5 * std_2 > 1

  # Now, if we have a larger delta, then even more benefit can come from the combinatorics,
  # lets check this
  delta = (4, 4)

  # Use the first spin fraction percent of the spins to obtain weights
  twopt_weights = get_twopt_weights(spins[:int(spin_fraction * N)], regions, delta)

  # naiive method - use all the spins
  mean_3, std_3 = twopt_average(spins, regions, numpy.ones((L, L)), delta, weight_type=2)

  # Higher weights have more varience in the M direction
  # Test with rest of the spins
  mean_4, std_4 = twopt_average(spins[int(spin_fraction * N):], regions, twopt_weights, delta, weight_type=2)

  # Check that knowledge of the weights helped us choose a better estimator
  assert std_2 < std_1
  # Check that the improvement is better than previously
  assert std_4 / std_3 < std_2 / std_1

  # Check the answers are correct to 5 sigma
  assert mean_3 - 5 * std_3 < 0
  assert mean_3 + 5 * std_3 > 0
  assert mean_4 - 5 * std_4 < 0
  assert mean_4 + 5 * std_4 > 0

  pdb.set_trace()


# def test_non_boundary():
#   my_non_boundary = non_boundary(4, splitting=(1, 2))

#   assert numpy.array_equal(my_non_boundary, numpy.array([[0, 1, 0, 1],
#                                                          [0, 1, 0, 1],
#                                                          [0, 1, 0, 1],
#                                                          [0, 1, 0, 1]]))

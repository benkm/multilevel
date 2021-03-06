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


def test_twopt_average():
  L = 8
  N = 5
  M = 10
  regions = region_maker(L, (2, 2))
  spins = numpy.zeros((N, M, L, L))
  for i in range(N):
    for j in range(M):
      spins[i, j] = regions

  # First check the simple twopt weights
  weights = numpy.ones((L, L))
  delta = (0, 0)
  estimator = twopt_average(spins, regions, weights, delta, weight_type=2)

  # Compare to hand caculation
  assert estimator == numpy.mean(regions ** 2)

  # I think that for any way of rolling the array the result will be lower
  for i in range(1, L):
    for j in range(1, L):
      delta = (i, j)

      estimator_2 = twopt_average(spins, regions, weights, delta, weight_type=2)

      assert estimator_2 < estimator
  
  # Check that for a constant spin matrix, the result comes out the same regardless
  x = numpy.random.rand()
  spins = numpy.ones((N, M, L, L)) * x

  for i in range(L):
    for j in range(L):
      for weight_type in [1, 2]:
        delta = (i, j)

        weights = numpy.random.rand(L, L)
        estimator = twopt_average(spins, regions, weights, delta, weight_type=weight_type)

        assert numpy.abs(estimator - x ** 2) < 10 ** -10
  
  # Let's check that given knowledge of the error by lattice site, we can achieve
  # a reduced error through use of weights
  weights = numpy.random.rand(L, L)

  weights_extended = weights.reshape((1, 1, L, L)).repeat(N, axis=0).repeat(M, axis=1)

  spins = numpy.ones((N, M, L, L)) + numpy.random.rand(N, M, L, L) * 0.1 * weights

  # Make sure the boundaries are constant in the N direction
  regions_extended = regions.reshape(1, 1, L, L).repeat(N, axis=0).repeat(M, axis=1)
  spins = numpy.where(regions, spins, spins[:, 0, ...].reshape(N, 1, L, L).repeat(M, axis=1))

  # Let's check that the boundarys indeed have a larger standard dev
  # Perform an average in the M axis
  boot_spins = numpy.mean(spins, axis=1)
  random = numpy.random.randint(0, N, size=(100, N))
  results = numpy.zeros((100, L, L))

  for i in range(100):
    some_spins = boot_spins[random[i]]
    results[i] = numpy.mean(some_spins, axis=0)

  means = numpy.mean(results, axis=0)
  stds = numpy.std(results, axis=0)

  boundary = regions == 0
  non_boundary = regions != 0

  boundary_av_std = numpy.sum(boundary * stds) / numpy.sum(boundary)
  non_boundary_av_std = numpy.sum(non_boundary * stds) / numpy.sum(non_boundary)

  # In theory the boundary std should be sqrt(M) larger
  assert boundary_av_std > non_boundary_av_std * numpy.sqrt(M) * 0.7

  # Now let's use the weights, and perform a bootstrap
  # First try with no delta
  estimator_1 = numpy.zeros(100)
  estimator_2 = numpy.zeros(100)

  for i in range(100):
    # naiive method
    estimator_1[i] = twopt_average(spins[random[i]], regions, numpy.ones((L, L)), (0, 0), weight_type=2)

    # using weights, with inverse-varience method
    estimator_2[i] = twopt_average(spins[random[i]], regions, 1 / weights ** 2, delta, weight_type=1)
  
  mean_1 = numpy.mean(estimator_1)
  mean_2 = numpy.mean(estimator_2)
  
  std_1 = numpy.std(estimator_1)
  std_2 = numpy.std(estimator_2)

  pdb.set_trace()

  assert std_2 < std_1



# def test_non_boundary():
#   my_non_boundary = non_boundary(4, splitting=(1, 2))

#   assert numpy.array_equal(my_non_boundary, numpy.array([[0, 1, 0, 1],
#                                                          [0, 1, 0, 1],
#                                                          [0, 1, 0, 1],
#                                                          [0, 1, 0, 1]]))

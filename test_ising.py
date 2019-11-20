from ising2 import *


def test_chessboard():
  my_chessboard = chessboard(4)

  assert numpy.array_equal(my_chessboard, numpy.array([[0, 1, 0, 1],
                                                       [1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [1, 0, 1, 0]]))


def test_non_boundary():
  my_non_boundary = non_boundary(4, splitting=(1, 2))

  assert numpy.array_equal(my_non_boundary, numpy.array([[0, 1, 0, 1],
                                                         [0, 1, 0, 1],
                                                         [0, 1, 0, 1],
                                                         [0, 1, 0, 1]]))


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

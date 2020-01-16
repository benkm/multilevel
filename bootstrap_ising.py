from ising3 import *
from multiprocessing import Pool
from scipy.optimize import *
from copy import deepcopy


numpy.set_printoptions(suppress=True)
numpy.set_printoptions(formatter={'float_kind':'{:10.3f}'.format})
numpy.set_printoptions(linewidth=200)
numpy.set_printoptions(threshold=20)

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['r', 'g', 'b', 'y', 'k', 'gray', 'cyan', 'pink', 'darkgreen', 'orange', 'purple', 'saddlebrown']) 
plt.rcParams.update({'lines.markeredgewidth': 1})

def realspace_horizontal(beta):
  L = 16
  N = 10
  M = 100
  splitting = (2, 2)
  initial = 1000
  step = 100
  Nstep = 100
  Mstep = 100
  rerun_generation = False
  rerun = True
  no_samples = 10

  # Which algorithmic methods are being tested
  # normal : no manipulations
  # connected : the onept function is subtracted before calculating twopt
  # connected_readdition : the onept function (squared) is readded to the result
  #   of the above
  # flipped : Before subtraction of the onept function overall negative states
  #   are flipped
  # Note : flipped readdition just wouldn't make sense
  # single_methods = ["normal", "connected", "connected_readdition", "flipped", "flipped_readdition"]
  # multi_methods = ["normal", "connected", "connected_readdition", "flipped", "flipped_readdition"]
  single_methods = []
  multi_methods = ["normal", "flipped_readdition"]

  single_methods_copy = list.copy(single_methods)
  multi_methods_copy = list.copy(multi_methods)

  # At what spacing are the results calculated? Note that this may well affect
  # the results of the momentum space propagators.
  spacing = 1

  # How many results does this produce in a given direction
  no_results = (L // 2) // spacing

  # Produce the array of deltas that will be investigated
  delta_range = numpy.arange(0, L//2, spacing)
  deltas = [(i, 0) for i in delta_range]
  
  # Create objects that will collect the results that we will generate
  results_single = {}
  results_multi = {}
  for method in single_methods:
    # First check if the result is present already 
    results_single[method] = numpy.zeros((no_samples, no_results))
  
  for method in multi_methods:
    results_multi[method] = numpy.zeros((no_samples, no_results))

  # Generate the N*M states for the single threaded method
  single_spins = generate_states_1_1(L, beta, N * M, step, initial, initial_spins=numpy.ones((L, L)), rerun=rerun_generation)

  # Generate the split states
  multi_spins = generate_states_splitting(L, beta, N, M, Mstep, splitting, N * M, initial, Nstep=Nstep, source_file_step=step, rerun=rerun_generation)

  # Create a place to store the end data
  directory = f"../data/bootstrap/ising/p_correlator/L{L}/b{beta:.3f}/"

  single_filename = {}
  multi_filename = {}
  for method in single_methods:
    single_filename[method] = f"bootstrap_data_single_{method}_L{L}_b{beta:.3f}_N{N}_M{M}_Mstep{Mstep}_sourceN{N * M}_sourceinit{initial}_sourcestep{step}_Nstep{Nstep}_{splitting[0]}_{splitting[1]}_single.npy"
  
  for method in multi_methods:
    multi_filename[method] = f"bootstrap_data_multi_{method}_L{L}_b{beta:.3f}_N{N}_M{M}_Mstep{Mstep}_sourceN{N * M}_sourceinit{initial}_sourcestep{step}_Nstep{Nstep}_{splitting[0]}_{splitting[1]}_single.npy"
  
  regions = region_maker(L, splitting)

  # Create random numbers for performing bootstrap
  resampling_multi = numpy.random.randint(0, N, size=(no_samples, N))
  resampling_single = numpy.random.randint(0, N * M, size=(no_samples, N * M))

  # Before beggining computation, if rerun=False then check if data already
  # exists

  flags = {}
  if not rerun:
    for method in single_methods:
      flags[method] = 0
      try:
        numpy.load(f"{directory}{single_filename[method]}")
      except:
        flags[method] = 1

    for method in flags.keys():
      # If the data is found, then flag = 0, so don't rerun data
      if not flags[method]:
        single_methods.remove(method)
  
  flags = {}
  if not rerun:
    for method in multi_methods:
      flags[method] = 0
      try:
        numpy.load(f"{directory}{multi_filename[method]}")
      except:
        flags[method] = 1

    for method in flags.keys():
      # If the data is found, then flag = 0, so don't rerun data
      if not flags[method]:
        multi_methods.remove(method)

  # Perform a bootstrap ======================================================
  for sample in range(no_samples):
    print(f" ============ Calculating bootstrap sample {sample} =============")
    single_dictionary = {}
    multi_dictionary = {}
  
    # Reorder spins
    single_temp = single_spins[resampling_single[sample]]
    multi_temp = multi_spins[resampling_multi[sample]]

    for method in single_methods:
      single_dictionary[method] = deepcopy(single_temp)

    # Seperate the spins into two groups, where one group finds the weights for the
    # other and vice versa, so that all data is used
    for method in multi_methods:
      multi_dictionary[method] = [deepcopy(multi_temp[:N // 2]), deepcopy(multi_temp[N // 2:])]

    if "flipped" in single_methods:
      negative = (numpy.mean(single_temp, axis=(1, 2)) < 0)

      single_dictionary["flipped"][negative] = -single_dictionary["flipped"][negative]

      # If spins are flipped then connected part is removed
      single_dictionary["flipped"] = single_dictionary["flipped"] - numpy.mean(single_dictionary["flipped"])

    if "flipped_readdition" in single_methods:
      negative = (numpy.mean(single_temp, axis=(1, 2)) < 0)

      single_dictionary["flipped_readdition"][negative] = -single_dictionary["flipped_readdition"][negative]

      single_flipped_onept = numpy.mean(single_dictionary["flipped_readdition"])

      single_dictionary["flipped_readdition"] = single_dictionary["flipped_readdition"] - numpy.mean(single_dictionary["flipped_readdition"])
                           

    # Remove connected part if applicable
    if "connected" in single_methods:
      single_dictionary["connected"] = single_dictionary["connected"] - numpy.mean(single_dictionary["connected"])

    if "connected" in multi_methods:
      for i in [0, 1]:
        multi_dictionary["connected"][i] = multi_dictionary["connected"][i] - numpy.mean([numpy.mean(multi_dictionary["connected"][0]),
                                            numpy.mean(multi_dictionary["connected"][1])])

    if "connected_readdition" in single_methods:
      single_onept_part = numpy.mean(single_dictionary["connected"])
      single_dictionary["connected_readdition"] = single_dictionary["connected_readdition"] - single_onept_part

    # Note : To get the correct results with multi connected readdition, then the
    # part to subtract depends on delta

    # Generate bespoke onept_weights for each multilevel method. 
    onept_weights = {}
    onept_weights_sym = {}
    onept_weights_comb = {}
    for method in multi_methods:
      onept_weights[method] = [0, 0]
      onept_weights_sym[method] = [0, 0]
      onept_weights_comb[method] = [0, 0]

      # Extract onept weights from each spin group
      for i in [0, 1]:
        onept_weights[method][i] = get_onept_weights(multi_dictionary[method][i], regions, cap=False)
        onept_weights_sym[method][i] = use_symmetry(onept_weights[method][i], splitting)
        onept_weights_sym[method][i] = numpy.where(regions, numpy.minimum(M, onept_weights_sym[method][i]), 1)

    # Cycle through each delta ===================================================
    for i, delta in enumerate(deltas):
      for method in single_methods:
        results_single[method][sample, i] = twopt_average(single_dictionary[method].reshape((N * M, 1, L, L)), regions, numpy.ones((L, L)), delta)[0]

      # Re-ad onept part if applicable
      if "connected_readdition" in single_methods:
        results_single["connected_readdition"][sample, i] += single_onept_part ** 2
      
      if "flipped_readdition" in single_methods:
        results_single["flipped_readdition"][sample, i] += single_flipped_onept ** 2

      # Combine the onept weights to get twopt weights
      for method in multi_methods:
        for j in [0, 1]:
          onept_weights_comb[method][j] = combine_onept_weights(onept_weights_sym[method][j], regions, delta)

      C = [0, 0]
      if "connected_readdition" in multi_methods:
        for j in [0, 1]:
          C[j] = 0.5 * numpy.average(numpy.mean(multi_dictionary["connected_readdition"][j], axis=(0,1)), weights=onept_weights_comb["connected_readdition"][not j])
          C[j] += 0.5 * numpy.average(numpy.mean(numpy.roll(numpy.roll(multi_dictionary["connected_readdition"][j], -delta[1], axis=3), -delta[0], axis=2), axis=(0,1)), weights=onept_weights_comb["connected_readdition"][not j])

          multi_dictionary["connected_readdition"][j] = multi_dictionary["connected_readdition"][j] - C[j]
      
      flipped_C = [0, 0]
      if "flipped" in multi_methods:
        # Average over the M axis as well as the spatial axes of the lattice
        for j in [0, 1]:
          negative = (numpy.mean(multi_dictionary["flipped"][j], axis=(1, 2, 3)) < 0)

          multi_dictionary["flipped"][j][negative] = -multi_dictionary["flipped"][j][negative]

          flipped_C[j] = 0.5 * numpy.average(numpy.mean(multi_dictionary["flipped_readdition"][j], axis=(0,1)), weights=onept_weights_comb["flipped_readdition"][not j])
          flipped_C[j] += 0.5 * numpy.average(numpy.mean(numpy.roll(numpy.roll(multi_dictionary["flipped_readdition"][j], -delta[1], axis=3), -delta[0], axis=2), axis=(0,1)), weights=onept_weights_comb["flipped_readdition"][not j])
          
          multi_dictionary["flipped"][j] = multi_dictionary["flipped"][j] - (flipped_C[0] ** 2 + flipped_C[1] ** 2) / 2

      if "flipped_readdition" in multi_methods:
        for j in [0, 1]:
          negative = (numpy.mean(multi_dictionary["flipped_readdition"][j], axis=(1, 2, 3)) < 0)

          multi_dictionary["flipped_readdition"][j][negative] = -multi_dictionary["flipped_readdition"][j][negative]

          flipped_C[j] = 0.5 * numpy.average(numpy.mean(multi_dictionary["flipped_readdition"][j], axis=(0,1)), weights=onept_weights_comb["flipped_readdition"][not j])
          flipped_C[j] += 0.5 * numpy.average(numpy.mean(numpy.roll(numpy.roll(multi_dictionary["flipped_readdition"][j], -delta[1], axis=3), -delta[0], axis=2), axis=(0,1)), weights=onept_weights_comb["flipped_readdition"][not j])

          multi_dictionary["flipped_readdition"][j] = multi_dictionary["flipped_readdition"][j] - flipped_C[j]


      for method in multi_methods:
        # Average over the 2 spin groups, where each group uses the other groups weights
        results_multi[method][sample, i] = (twopt_average(multi_dictionary[method][0], regions, onept_weights_comb[method][1], delta)[0] +
                                            twopt_average(multi_dictionary[method][1], regions, onept_weights_comb[method][0], delta)[0]) / 2

      if "connected_readdition" in multi_methods:
        results_multi["connected_readdition"][sample, i] = results_multi["connected_readdition"][sample, i] + (C[0] ** 2 + C[1] ** 2) / 2

        # Return the spins to their original value
        multi_dictionary["connected_readdition"][j] = multi_dictionary["connected_readdition"][j] + C[j]

      if "flipped_readdition" in multi_methods:
        results_multi["flipped_readdition"][sample, i] = results_multi["flipped_readdition"][sample, i] + (flipped_C[0] ** 2 + flipped_C[1] ** 2) / 2
    
        # Return the spins to their original values
        multi_dictionary["flipped_readdition"][j] = multi_dictionary["flipped_readdition"][j] + flipped_C[j]

  if not os.path.isdir(directory):
    os.makedirs(directory)
  
  for method in single_methods:
    numpy.save(f"{directory}{single_filename[method]}", results_single[method])
  
  for method in multi_methods:
    numpy.save(f"{directory}{multi_filename[method]}", results_multi[method])

  plt.figure()
  # Reload in the data in case it wasn't rerun
  for method in single_methods_copy:
    results_single[method] = numpy.load(f"{directory}{single_filename[method]}")
    plt.errorbar(delta_range, numpy.mean(results_single[method], axis=0),
                                         yerr=numpy.std(results_single[method], axis=0), label = f"single - {method}")

  for method in multi_methods_copy:
    results_multi[method] = numpy.load(f"{directory}{multi_filename[method]}")
    plt.errorbar(delta_range, numpy.mean(results_multi[method], axis=0),
                                         yerr=numpy.std(results_multi[method], axis=0), label = f"multi - {method}")

  plt.legend()
  plt.yscale('log')

  plt.close()
  for method in single_methods_copy:
    plt.plot(delta_range, numpy.std(results_single[method], axis=0), label = f"single - {method}")

  for method in multi_methods_copy:
    plt.plot(delta_range, numpy.std(results_multi[method], axis=0), label = f"multi - {method}")
  
  plt.legend()

  pdb.set_trace()

realspace_horizontal(0.2)
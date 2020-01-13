from ising3 import *
from multiprocessing import Pool
from scipy.optimize import *


numpy.set_printoptions(suppress=True)
numpy.set_printoptions(formatter={'float_kind':'{:10.3f}'.format})
numpy.set_printoptions(linewidth=200)
numpy.set_printoptions(threshold=20)

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['r', 'g', 'b', 'y', 'k', 'gray', 'cyan', 'pink', 'darkgreen', 'orange', 'purple', 'saddlebrown']) 
plt.rcParams.update({'lines.markeredgewidth': 1})

def get_data(L, beta, N, M, delta, splitting, fitting_size, weight_types='all', initial=1000, step=100, Nstep=1, Mstep=100, rerun=False, rerun_generation=False, connected=True):
  """
    weight_types is a list of strings encoding different ways of getting weights
  """
  i, j = splitting[0], splitting[1]
  numpy.random.seed()

  if connected:
    directory = f"../analysis/ising/L{L}/b{beta:.3f}/connected/"
    filename = f"analysis_L{L}_b{beta:.3f}_N{N}_M{M}_delta{delta}_Mstep{Mstep}_sourceN{N * M}_sourceinit{initial}_sourcestep{step}_Nstep{Nstep}_spins_{i}_{j}_connected.npy"
  
  else:
    directory = f"../analysis/ising/L{L}/b{beta:.3f}/"
    filename = f"analysis_L{L}_b{beta:.3f}_N{N}_M{M}_delta{delta}_Mstep{Mstep}_sourceN{N * M}_sourceinit{initial}_sourcestep{step}_Nstep{Nstep}_spins_{i}_{j}.npy"
  
  # If all then compilate list of all the different weight types
  if weight_types == 'all':
    weight_types = ['single_level', 'naiive', 'onept', 'twopt', 'naiive_plus', 'magnetisation', 'trial']

  to_do = []
  exist_old_result = True
  # Check if the results are already saved
  try:
    result_old = numpy.load(f"{directory}{filename}")
    data_type_old = result_old.dtype.descr

  except:
    # All weights must be done from scratch
    to_do = weight_types
    data_type_old = []
    exist_old_result = False

  # If the results sucessfully loaded in
  if to_do != weight_types:
    # Go through the saved results, and add any elements of weight_types that aren't present to the todo list
    for weight in weight_types:
      try:
        x = result_old[weight]

      except:
        to_do.append(weight)
  
  if rerun:
    to_do = weight_types

  for weight in to_do: 
    print(f"Getting results for L = {L}, N = {N}, M = {M}, delta = {delta}, with {weight} type")
  
  regions = region_maker(L, splitting)

  # Initiate the model with all up spins, this is for the purpose of making the
  # low temperature results easier to interpret, and thus aiding in the finding
  # of the critical point.
  start_state = numpy.ones((L, L))

  # # Generate the N*M states for the single threaded method
  single_spins = generate_states_1_1(L, beta, N * M, step, initial, initial_spins=start_state, rerun=rerun_generation)

  # # # Generate the split states
  spins = generate_states_splitting(L, beta, N, M, Mstep, splitting, N * M, initial, Nstep=Nstep, source_file_step=step, rerun=rerun_generation)

  assert spins.shape == (N, M, L, L)
  assert single_spins.shape == (N * M, L, L)
  assert numpy.sum(numpy.abs(single_spins[0: N * Nstep: Nstep] -  spins[:, 0])) < 10 ** -10

  if connected:
    ## Remove the disconnected part
    single_spins = single_spins - numpy.mean(single_spins)
    spins = spins - numpy.mean(spins)

  # # Break the spins down into the fitting set and the data set
  spins_fitting = spins[:fitting_size]
  spins_data = spins[fitting_size:]

  # The data_type of the resulting data is that of the saved results previously plus
  # and new weightings being done
  data_type_new = [(weight, '<f8', (2, )) for weight in to_do]

  if rerun:
    data_type_old = []
  
  data_type = data_type_old + data_type_new

  result_new = numpy.zeros(1, dtype = data_type)


  if exist_old_result:
    for weight in result_old.dtype.names:
      result_new[weight] = result_old[weight]

  if 'onept' in to_do:
    # Extract onept weights from fitting spins
    onept_weights = get_onept_weights(spins_fitting, regions)

    # Combine the onept weights to make twopt weights
    onept_weights = combine_onept_weights(onept_weights, regions, delta)
    
    # result_new['onept'] = numpy.array(twopt_average(spins_data, regions, onept_weights, delta))

  if 'onept_sym' in to_do:
    # Extract onept weights from fitting spins
    onept_weights = get_onept_weights(spins_fitting, regions, cap=False)

    # Symmetrise onept weights with sublattice symmetry group
    onept_weights_sym = use_symmetry(onept_weights, splitting)
    onept_weights_sym = numpy.where(regions, numpy.minimum(M, onept_weights_sym), 1)

    # Combine the onept weights to make twopt weights
    onept_weights_comb = combine_onept_weights(onept_weights_sym, regions, delta)
    
    result_new['onept_sym'] = numpy.array(twopt_average(spins_data, regions, onept_weights_comb, delta))

  if 'twopt' in to_do:
    # Extract twopt weights from the fitting spins
    twopt_weights = get_twopt_weights(spins_fitting, regions, delta)
  
    result_new['twopt'] = numpy.array(twopt_average(spins_data, regions, twopt_weights, delta))

  if 'naiive_plus' in to_do:
    # Construct naiive_plus weights, which implement the boundary being different
    naiive_plus = numpy.where(regions, M, 1)  # These are onept weights

    # Turn naiive plus into twopt weights
    naiive_plus = combine_onept_weights(naiive_plus, regions, delta)

    result_new['naiive_plus'] = numpy.array(twopt_average(spins, regions, naiive_plus, delta))

  if 'magnetisation' in to_do:
    # Average over the M direction
    M_magnetisation = numpy.mean(spins_fitting, axis=1)

    # Take the absolute value
    M_mag_abs = numpy.abs(M_magnetisation)

    # average over the N direction
    av_mag = numpy.mean(M_mag_abs, axis=0)

    # Use the weight formula from before
    mag_weights = 1 / (av_mag ** 2 * (1 - 1 / M) + 1 / M)

    # Combine to get twopt weights
    mag_weights_twopt = combine_onept_weights(mag_weights, regions, delta)

    result_new['magnetisation'] = numpy.array(twopt_average(spins_data, regions, mag_weights_twopt, delta))

  if 'trial' in to_do:
    # Extract onept weights from fitting spins
    onept_weights = get_onept_weights(spins_fitting, regions)

    # Combine the onept weights to make twopt weights
    onept_weights_comb = combine_onept_weights(onept_weights, regions, delta)
    
    # Symmetrise onept weights with sublattice symmetry group
    onept_weights_sym = use_exchange_sym(onept_weights_comb, splitting)
    
    indices = numpy.indices((L, L))
    i_s, j_s = indices[0, ...], indices[1, ...]

    # Fit to the onept weights
    def least_sq(gamma):
      ## Warning this only works for splitting = (1 or 2, 1 or 2)
      trial_alpha = numpy.maximum(numpy.exp(- gamma * numpy.minimum(i_s%(L//splitting[0]), j_s%(L//splitting[1]))), numpy.exp(- gamma * numpy.minimum((L//splitting[0] - i_s)%(L//splitting[0]), (L//splitting[1] - j_s)%(L//splitting[1]))))

      trial_weights = 1 / (trial_alpha ** 2 * (1 - 1 / M) + 1 / M)

      trial_weights = combine_onept_weights(trial_weights, regions, delta)

      return numpy.sum((trial_weights - onept_weights_sym) ** 2)
    
    gamma = minimize_scalar(least_sq, bracket=[0, 1])['x']

    trial_alpha = numpy.maximum(numpy.exp(- gamma * numpy.minimum(i_s%(L//splitting[0]), j_s%(L//splitting[1]))), numpy.exp(- gamma * numpy.minimum((L//splitting[0] - i_s)%(L//splitting[0]), (L//splitting[1] - j_s)%(L//splitting[1]))))

    trial_weights = 1 / (trial_alpha ** 2 * (1 - 1 / M) + 1 / M)

    trial_weights = combine_onept_weights(trial_weights, regions, delta)

    result_new['trial'] = numpy.array(twopt_average(spins_data, regions, trial_weights, delta))

  if 'trial_cp' in to_do:
    # Fit to the expected shape of the critical point twopt correlator function

    # Extract onept weights from fitting spins
    onept_weights = get_onept_weights(spins_fitting, regions)

    # Combine the onept weights to make twopt weights
    onept_weights_comb = combine_onept_weights(onept_weights, regions, delta)
    
    # Symmetrise onept weights with sublattice symmetry group
    onept_weights_sym = use_exchange_sym(onept_weights_comb, splitting)

    indices = numpy.indices((L, L))
    i_s, j_s = indices[0, ...], indices[1, ...]

    # Fit to the onept weights
    def least_sq(x):
      gamma, alpha = x[0], x[1]

      # Distance from the boundary
      dist = numpy.minimum(numpy.minimum(i_s%(L//2), j_s%(L//2)), numpy.minimum((L//2 - i_s)%(L//2), (L//2 - j_s)%(L//2)))

      trial_cor = numpy.maximum(1 - (gamma * dist) ** alpha, 0)

      trial_weights = 1 / (trial_cor ** 2 * (1 - 1 / M) + 1 / M)

      trial_weights = combine_onept_weights(trial_weights, regions, delta)

      return numpy.sum((trial_weights - onept_weights_sym) ** 2)
    
    dist = numpy.minimum(numpy.minimum(i_s%(L//2), j_s%(L//2)), numpy.minimum((L//2 - i_s)%(L//2), (L//2 - j_s)%(L//2)))

    mini_result = minimize(least_sq, numpy.array([1 / L, 0.5]), bounds = [(0, 1), (0, 1)])['x']

    gamma, alpha = mini_result[0], mini_result[1]

    trial_cor = numpy.maximum(1 - (gamma * dist) ** alpha, 0)

    trial_weights = 1 / (trial_cor ** 2 * (1 - 1 / M) + 1 / M)

    trial_weights = combine_onept_weights(trial_weights, regions, delta)

    result_new['trial_cp'] = numpy.array(twopt_average(spins_data, regions, trial_weights, delta))

  if 'single_level' in to_do:
    result_new['single_level'] = numpy.array(twopt_average(single_spins.reshape((N * M, 1, L, L)), regions, numpy.ones((L, L)), delta))

  if 'naiive' in to_do:
    result_new['naiive'] = numpy.array(twopt_average(spins, regions, numpy.ones((L, L)), delta))

  # Save the result!
  if not os.path.isdir(directory):
    os.makedirs(directory)

  numpy.save(f"{directory}{filename}", result_new)

  return result_new


def investigate_delta_beta(beta):
  L = 8
  N = 100
  M = 100
  splitting = (2, 2)
  fitting_size = int(numpy.rint(0.2 * N))
  initial = 1000
  step = 100
  Nstep = 100
  Mstep = 100
  rerun = True
  connected = True
  delta_space = 1 # Gap between subsequently sampled delta values
  delta_min = 3
  delta_max = 4  # Exclusive upper boundary
  assert (delta_max - delta_min - 1)%delta_space == 0
  no_deltas = (delta_max - delta_min - 1)//delta_space + 1
  delta_array = numpy.arange(delta_min, delta_max, delta_space)

  deltas = [(i, 0) for i in delta_array]

  # Types of weighting we want to investigate this run
  weight_types = ['onept_sym', 'single_level']

  # Collect the results by calling get data
  results = {}
  for weight in weight_types:
    results[weight] = numpy.zeros((no_deltas, 2))

  for i, delta in enumerate(deltas):
    # Look at the dependence of delta in the x direction
    
    result = get_data(L, beta, N, M, delta, splitting, fitting_size, weight_types=weight_types, initial=initial, step=step, Nstep=Nstep, Mstep=Mstep, rerun=rerun, connected=connected)

    for weight in weight_types:
      results[weight][i][0] = result[weight][0, 0]  # means
      results[weight][i][1] = result[weight][0, 1]  # std dev

  plt.figure()

  ## Extract the correlation length by fitting an exponential to the twopt correlator

  # First remove noise from the signal - to the 5 sigma level
  signal_condition = (results['single_level'][:, 0] / results['single_level'][:, 1]) > 5
  signal_condition = numpy.logical_and(signal_condition, delta_array > 0)
  twopt_signal = results['single_level'][:, 0][signal_condition]
  deltas_signal = delta_array[signal_condition]
  log_twopt_signal = numpy.log(twopt_signal)

  # Use exponential anzatz to extract
  if len(deltas_signal) > 2:
    params, cov = numpy.polyfit(deltas_signal, log_twopt_signal, 1, cov=True)

    xi = -1 / params[0]

    # By assuming that the error in xi is much less than xi and taking the first order
    # of the taylor series
    xi_err = numpy.sqrt(cov[0, 0]) / params[0] ** 2
  
    plt.plot(delta_array, numpy.exp(-delta_array / xi) * numpy.exp(params[1]), label=f'fit, xi={xi:.2f}')

  elif len(deltas_signal) == 2:
    params = numpy.polyfit(deltas_signal, log_twopt_signal, 1)

    xi = -1 / params[0]
    xi_err = 0

    plt.plot(delta_array, numpy.exp(-delta_array / xi) * numpy.exp(params[1]), label='fit')

  else:
    xi = 0
    xi_err = 0

  # Plot a graph of the means
  for weight in weight_types:
    plt.errorbar(delta_array, results[weight][:, 0], yerr=results[weight][:, 1], marker='_', label=weight)

  plt.xlabel("i, where delta = (i, 0)")
  if connected:
    plt.ylabel("<phi(x)phi(x + delta)> - <phi>^2")
  else:
    plt.ylabel("<phi(x)phi(x + delta)>")

  plt.title(f"""L={L}, N={N}, M={M}, splitting={splitting}, fitting_size={fitting_size},
          initial={initial}, step={step}, Nstep={Nstep}, Mstep={Mstep}""")
  plt.yscale('log')
  plt.legend()

  # Remove the out of range values
  plt.ylim(10 ** -4, 1.1)

  if connected:
    if Nstep == 1:
      if splitting == (2, 2):
        directory = f"../graphs/ising/beta_investigation/L{L}/N{N}/M{M}/connected/"
      else:
        directory = f"../graphs/ising/beta_investigation/L{L}/N{N}/M{M}/connected/splitting_{splitting}/"
    else:
      if splitting == (2, 2):
        directory = f"../graphs/ising/beta_investigation/L{L}/N{N}/M{M}/connected/Nstep{Nstep}/"
      else:
        directory = f"../graphs/ising/beta_investigation/L{L}/N{N}/M{M}/connected/Nstep{Nstep}/splitting_{splitting}/"
 
  else:
    if Nstep == 1:
      if splitting == (2, 2):
        directory = f"../graphs/ising/beta_investigation/L{L}/N{N}/M{M}/"
      else:
        directory = f"../graphs/ising/beta_investigation/L{L}/N{N}/M{M}/splitting_{splitting}/"
    else:
      if splitting == (2, 2):
        directory = f"../graphs/ising/beta_investigation/L{L}/N{N}/M{M}/Nstep{Nstep}/"
      else:
        directory = f"../graphs/ising/beta_investigation/L{L}/N{N}/M{M}/Nstep{Nstep}/splitting_{splitting}/"

  if not os.path.isdir(directory):
    os.makedirs(directory)

  print(f"Saving graphs for beta = {beta:.3f}, L = {L}, N = {N}, M = {M}")
  plt.savefig(f"{directory}delta_L{L}_beta{beta:.3f}_N{N}_M{M}.png")
  plt.close()

  plt.figure()

  # Plot a graph of the standard deviations only
  for weight in weight_types:
    plt.plot(delta_array, results[weight][:, 1], marker='_', label=weight)

  plt.legend()
  plt.xlabel("i, where delta = (i, 0)")
  if connected:
    plt.ylabel("std dev of (<phi(x)phi(x + delta)> - <phi>^2)")
  else:
    plt.ylabel("std dev of <phi(x)phi(x + delta)>")
  plt.title(f"""std : L={L}, N={N}, M={M}, splitting={splitting}, fitting_size={fitting_size},
          initial={initial}, step={step}, Nstep={Nstep}, Mstep={Mstep}""")
  plt.yscale('log')


  plt.savefig(f"{directory}std_delta_L{L}_beta{beta:.3f}_N{N}_M{M}.png")
  plt.close()

  plt.figure()

  # Plot a graph of the signal to noise ratio on a log scale
  for weight in weight_types:
    # Make sure no to include the first point (delta = (0, 0)) because std = 0 here
    plt.plot(delta_array[1:], numpy.abs(results[weight][1:, 0]) / results[weight][1:, 1], marker='_', label=weight)
  
  plt.xlabel("i, where delta = (i, 0)")
  if connected:
    plt.ylabel("signal to noise ratio for connected twopt correlator>")
  else:
    plt.ylabel("signal to noise ratio for twopt correlator>")

  plt.title(f"""signal/noise : L={L}, N={N}, M={M}, splitting={splitting}, fitting_size={fitting_size},
          initial={initial}, step={step}, Nstep={Nstep}, Mstep={Mstep}""")
  plt.yscale('log')
  plt.legend()

  # Remove everything with a signal to noise smaller than 3 as this is the
  # expectation value 0 area of the graph
  ax = plt.gca()
  ylim = list(ax.get_ylim())
  if ylim[1] > 2:
    ylim[0] = 2
    plt.ylim(ylim)

  plt.savefig(f"{directory}singal2noise_delta_L{L}_beta{beta:.3f}_N{N}_M{M}.png")
  plt.close()

  # # Return the improvement ratio and the correlation length
  # improvement_ratio = numpy.average(results['single_level'][:, 1]) / numpy.average(results['onept'][:, 1])

  # to_return = numpy.array([xi, improvement_ratio])

  # return to_return


def investigate_p_correlator(beta):
  L = 32
  N = 100
  M = 100
  splitting = (2, 2)
  fitting_size = int(numpy.rint(0.2 * N))
  initial = 1000
  step = 100
  Nstep = 100
  Mstep = 100
  rerun = True
  connected = True
  flip = True # Controls whether the mean spin of a config is forced to be non-negative

  full_results_single = numpy.zeros((L, L, N * M))
  full_results_onept = numpy.zeros((L, L, N - fitting_size))

  # # Generate the N*M states for the single threaded method
  single_spins = generate_states_1_1(L, beta, N * M, step, initial, initial_spins=numpy.ones((L, L)), rerun=False)

  # # # Generate the split states
  spins = generate_states_splitting(L, beta, N, M, Mstep, splitting, N * M, initial, Nstep=Nstep, source_file_step=step, rerun=False)

  directory = f"../data/ising/p_correlator/L{L}/b{beta:.3f}/"
  filename_single = f"data_L{L}_b{beta:.3f}_N{N}_M{M}_Mstep{Mstep}_sourceN{N * M}_sourceinit{initial}_sourcestep{step}_Nstep{Nstep}_{splitting[0]}_{splitting[1]}_single.npy"
  filename_onept = f"data_L{L}_b{beta:.3f}_N{N}_M{M}_Mstep{Mstep}_sourceN{N * M}_sourceinit{initial}_sourcestep{step}_Nstep{Nstep}_{splitting[0]}_{splitting[1]}_onept.npy"
  
  if flip:
    directory += "flipped/"
    filename_single = filename_single[:-4] + "flipped.npy"
    filename_onept = filename_onept[:-4] + "flipped.npy"
    single_negative = (numpy.mean(single_spins, axis=(1, 2)) < 0)

    # Average over the M axis as well as the spatial axes of the lattice
    negative = (numpy.mean(spins, axis=(1, 2, 3)) < 0)

    single_spins[single_negative] = -single_spins[single_negative]
    spins[negative] = -spins[negative]

  regions = region_maker(L, splitting)

  spins_fitting = spins[:fitting_size]
  spins_data = spins[fitting_size:]

  # Extract onept weights from fitting spins
  onept_weights = get_onept_weights(spins_fitting, regions, cap=False)

  # Symmetrise onept weights with sublattice symmetry group
  onept_weights_sym = use_symmetry(onept_weights, splitting)
  onept_weights_sym = numpy.where(regions, numpy.minimum(M, onept_weights_sym), 1)

  if connected:
    directory += "connected/"
    filename_single = filename_single[:-4] + "connected.npy"
    filename_onept = filename_onept[:-4] + "connected.npy"

    ## Remove the disconnected part, keep track of this for later
    single_spin_mag = numpy.mean(single_spins)
    single_spins = single_spins - single_spin_mag

  try:
    full_results_single = numpy.load(f"{directory}{filename_single}")
    full_results_onept = numpy.load(f"{directory}{filename_onept}")

  except:
    rerun = True
  
  if rerun:
    # Use symmetry and save computer time
    temp_results_single = {}
    temp_results_onept = {}

    for i in range(0, L//2 + 1):
      for k in range(L):
        print(f"Calculating for delta = {i}, {k}")

        # Combine the onept weights to make twopt weights
        onept_comb = combine_onept_weights(onept_weights_sym, regions, (i, k))
        
        if connected:
          # Perform some manipulation to get a better variance
          C = 0.5 * numpy.average(numpy.mean(spins_data, axis=(0,1)), weights=onept_comb)
          C += 0.5 * numpy.average(numpy.mean(numpy.roll(numpy.roll(spins_data, -k, axis=3), -i, axis=2), axis=(0,1)), weights=onept_comb)
          spins_data_temp = spins_data - C

        else:
          spins_data_temp = spins_data
        
        temp_results_single[i, k] = twopt_average(single_spins.reshape((N * M, 1, L, L)), regions, numpy.ones((L, L)), (i, k), full_average=False)
        temp_results_onept[i, k] = twopt_average(spins_data_temp, regions, onept_comb, (i, k), full_average=False)
        
        if connected:
          temp_results_onept[i, k] += C ** 2
          temp_results_single[i, k] += single_spin_mag ** 2

    for i in range(L):
      for j in range(L):
        if i > L//2:
          i_label = L - i
          j_label = (L - j)%L
        else:
          i_label = i
          j_label = j
        
        full_results_single[i, j] = temp_results_single[i_label, j_label]
        full_results_onept[i, j] = temp_results_onept[i_label, j_label]

  if not os.path.isdir(directory):
    os.makedirs(directory)
  
  numpy.save(f"{directory}{filename_single}", full_results_single)
  numpy.save(f"{directory}{filename_onept}", full_results_onept)

  results_single = numpy.zeros((L, L, N * M), dtype=complex)
  results_onept = numpy.zeros((L, L, N - fitting_size), dtype=complex)

  print(f"Combining real space data into momentum space data, beta = {beta:.3f}, L = {L}, N = {N}, M = {M}")
  for p_i in tqdm(range(L)):
    for p_j in range(L):
        for i in range(L):
          for j in range(L):
            results_single[p_i, p_j] += (1 / L ** 2) * numpy.exp(- 1j * (2 * numpy.pi / L ) * (p_i * i + p_j * j)) * full_results_single[i, j]
            results_onept[p_i, p_j] += (1 / L ** 2) * numpy.exp(- 1j * (2 * numpy.pi / L ) * (p_i * i + p_j * j)) * full_results_onept[i, j]

  assert numpy.sum(numpy.imag(results_single)) < 10 ** -5
  assert numpy.sum(numpy.imag(results_onept)) < 10 ** -5

  # use 1 degree of freedom as the mean is unknown
  stds_single = (numpy.std(results_single, axis=2, ddof=1) / numpy.sqrt(N * M)).real
  stds_onept = (numpy.std(results_onept, axis=2, ddof=1) / numpy.sqrt(N - fitting_size)).real
  
  improvement = stds_single / stds_onept

  directory = f"../graphs/ising/beta_investigation/L{L}/N{N}/M{M}/"
  # Plot log of the improvement using imshow
  if flip:
    directory += "flipped/"
  if connected:
    directory += "connected/"
  if Nstep:
    directory += f"Nstep{Nstep}/"
  if splitting != (2, 2):
    directory += f"splitting_{splitting}/"
 
  plt.imshow(numpy.log10(improvement), cmap='RdBu')
  plt.colorbar(extend='both')

  # Make the limits the theoretically best and worst differences between the methods
  plt.clim(-0.5 *numpy.log10(N * M / (N - fitting_size)), 0.5 * numpy.log10(M * N / (N - fitting_size)))

  plt.xlabel("j")
  plt.ylabel("i")
  plt.title(f"""Improvement due to multilevel, for momentum space twopt correlator, for delta = (i, j)
                 L={L}, N={N}, M={M}, splitting={splitting}, fitting_size={fitting_size},
                initial={initial}, step={step}, Nstep={Nstep}, Mstep={Mstep}""")

  print(f"Saving graphs for beta = {beta:.3f}, L = {L}, N = {N}, M = {M}")
  
  if not os.path.isdir(directory):
    os.makedirs(directory)
  
  plt.savefig(f"{directory}improv_ratio_delta_L{L}_beta{beta:.3f}_N{N}_M{M}.png")
  plt.close()

  # Also make a real_space version of this graph just for comparison.
  stds_single_real = numpy.std(full_results_single, axis=2, ddof=1) / numpy.sqrt(N * M)
  stds_onept_real = numpy.std(full_results_onept, axis=2, ddof=1) / numpy.sqrt(N - fitting_size)

  improvement_real = stds_single_real / stds_onept_real

  plt.imshow(numpy.log10(improvement_real), cmap='RdBu')
  plt.colorbar(extend='both')

  plt.clim(-0.5 *numpy.log10(N * M / (N - fitting_size)), 0.5 * numpy.log10(M * N / (N - fitting_size)))
  plt.xlabel("j")
  plt.ylabel("i")
  plt.title(f"""Improvement due to multilevel, for real space twopt correlator, for delta = (i, j)
                 L={L}, N={N}, M={M}, splitting={splitting}, fitting_size={fitting_size},
                initial={initial}, step={step}, Nstep={Nstep}, Mstep={Mstep}""")

  plt.savefig(f"{directory}realspace_improv_ratio_delta_L{L}_beta{beta:.3f}_N{N}_M{M}.png")
  plt.close()

  # Let's do a correctness check to 10 sigma in each
  single = numpy.mean(results_single, axis=2)
  onept = numpy.mean(results_onept, axis=2)
  for i in range(L):
    for j in range(L):
      if single[i, j] > onept[i, j]:
        if (single[i, j] - 10 * stds_single[i, j] > onept[i, j] + 10 * stds_onept[i, j]):
          print(f"ERROR : RESULTS DONT SEEM TO MATCH, L = {L}, beta = {beta}, i={i}, j={j}")
      elif onept[i, j] > single[i, j]:
        if single[i, j] + 10 * stds_single[i, j] < onept[i, j] - 10 * stds_onept[i, j]:
          print(f"ERROR : RESULTS DONT SEEM TO MATCH, L = {L}, beta = {beta}, i={i}, j={j}")

  # Let's save a copy of the multilevel weights used to generate the data
  plt.imshow(onept_weights_sym)
  plt.colorbar()
  plt.savefig(f"{directory}onept_weights_sym_delta_L{L}_beta{beta:.3f}_N{N}_M{M}.png")
  plt.close()

  # Let's save a copy of the results in momentum space
  plt.imshow(single.real)
  plt.colorbar()
  plt.savefig(f"{directory}single_results_delta_L{L}_beta{beta:.3f}_N{N}_M{M}.png")
  plt.close()

  plt.imshow(onept.real)
  plt.colorbar()
  plt.savefig(f"{directory}onept_results_delta_L{L}_beta{beta:.3f}_N{N}_M{M}.png")
  plt.close()

  # Let's plot the real space correlators against mod x
  x_s = numpy.zeros(L ** 2)
  single_twopt = numpy.zeros(L ** 2)
  single_err = numpy.zeros(L ** 2)
  onept_twopt = numpy.zeros(L ** 2)
  onept_err = numpy.zeros(L ** 2)
  
  for i in range(L):
    for j in range(L):
      x_s[i * L + j] = numpy.sqrt(min(i, L - i) ** 2 + min(j, L - j) ** 2)
      single_twopt[i * L + j] = numpy.mean(full_results_single[i, j])
      single_err[i * L + j] = numpy.std(full_results_single[i, j], ddof=1) / numpy.sqrt(N * M)
      onept_twopt[i * L + j] = numpy.mean(full_results_onept[i, j])
      onept_err[i * L + j] = numpy.std(full_results_onept[i, j], ddof=1) / numpy.sqrt(N - fitting_size)

  plt.errorbar(x_s, single_twopt, yerr=single_err, label="single", ls='', capsize=2)
  plt.errorbar(x_s, onept_twopt, yerr=onept_err, label="onept", ls='', capsize=2)
  plt.legend()

  plt.savefig(f"{directory}realspace_twopt_L{L}_beta{beta:.3f}_N{N}_M{M}.png")
  plt.close()


min_beta = 0.05
max_beta = 0.8
no_beta = 16
x_range = numpy.linspace(min_beta, max_beta, no_beta)

p = Pool(8)
# p.map(investigate_delta_beta, x_range)
p.map(investigate_p_correlator, x_range)
# investigate_p_correlator(0.65)
# investigate_delta_beta(0.2)

# data = numpy.zeros((no_beta, 2))

# improvements = numpy.zeros((no_beta, 32, 32))

# improvements = numpy.array(p.map(investigate_p_correlator, x_range))

# p.close()

# pdb.set_trace()

## Plot a graph of the magnetisation vs. Temperature

no_points = 40
T_s = numpy.linspace(0.1, 4, no_points)
betas = 1 / T_s

L = 8
N = 100
M = 100
splitting = (2, 2)
fitting_size = int(numpy.rint(0.2 * N))
initial = 1000
step = 100
Nstep = 100
Mstep = 100
rerun = False

single_mag = numpy.zeros(no_points)
multi_mag = numpy.zeros(no_points)

def mag(beta):
  # Load in data
  single_spins = generate_states_1_1(L, beta, N * M, step, initial)
  spins = generate_states_splitting(L, beta, N, M, Mstep, splitting, N * M, initial, Nstep=Nstep, source_file_step=step)

  # Average spins
  single_mag = abs(numpy.mean(single_spins))
  multi_mag = abs(numpy.mean(spins))

  return numpy.array([single_mag, multi_mag])


p = Pool(8)

results = numpy.array(p.map(mag, betas))

p.close()
single_mag = results[:, 0]
multi_mag = results[:, 1]

plt.plot(T_s, single_mag, label='single')
plt.plot(T_s, multi_mag, label='multi')
plt.legend()

plt.show()
pdb.set_trace()
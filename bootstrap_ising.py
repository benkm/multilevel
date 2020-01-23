from ising3 import *
from multiprocessing import Pool
from scipy import optimize
from copy import deepcopy
from scipy import stats
from datetime import date


numpy.set_printoptions(suppress=True)
numpy.set_printoptions(formatter={'float_kind':'{:10.3f}'.format})
numpy.set_printoptions(linewidth=200)
numpy.set_printoptions(threshold=20)

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['r', 'g', 'b', 'y', 'k', 'gray', 'cyan', 'pink', 'darkgreen', 'orange', 'purple', 'saddlebrown']) 
plt.rcParams.update({'lines.markeredgewidth': 1})


# Let's attempt to find the correlation length for each bootstrap sample, and
# the error in said correlation lengths
def xi_fit(x, p0, p1, p2, p3, cor_type="Diagonal"):
  if cor_type == "Diagonal":
    return p0 * ((numpy.exp(-x * numpy.sqrt(2) / p1) + numpy.exp( - (L - x) * numpy.sqrt(2) / p1)) / (x ** p2)) + p3


def xi_fit2(x, p):
  return xi_fit(x, *p)


def fit_bootstrap(p0, deltas, datay, function, bounds):

    # Use a least squares method
    def errfunc(p2, x, y):
      return numpy.sum((function(x,p2) - y) ** 2)

    no_samples = datay.shape[0]

    ps = []
    for i in range(no_samples):
      randomfit = optimize.minimize(errfunc, p0, args=(deltas, datay[i]), bounds=bounds)['x']

      ps.append(randomfit) 

    ps = numpy.array(ps)
    mean_pfit = numpy.mean(ps, axis=0)

    err_pfit = numpy.std(ps, axis=0) 

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit
    return pfit_bootstrap, perr_bootstrap


def realspace(beta):
  L = 16
  N = 100
  single_low_mode_s = {}
  single_low_mode_errs = {}
  multi_low_mode_s = {}
  multi_low_mode_errs = {}
  results_single_M = {}
  results_multi_M = {}
  for M in 2 ** numpy.arange(1, 8):  # Between 2 and 128
    single_low_mode_s[M] = {}
    single_low_mode_errs[M] = {}
    multi_low_mode_s[M] = {}
    multi_low_mode_errs[M] = {}
    splitting = (2, 2)
    initial = 1000
    step = 100
    Nstep = M  # This spreads the configs through the region
    Mstep = 100
    rerun_generation = False
    rerun = False
    no_samples = 100
    cor_type = "Diagonal" # Which direction are the correlators calculated in?
    # Options are "Horizontal" and "Verical"
    different_weights = False # If true then weights are calculated after some
      # manipulation for connected and flipped
    return_mom_mode = False

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
    single_methods = ["flipped"]
    multi_methods = ["flipped"]

    single_methods_copy = list.copy(single_methods)
    multi_methods_copy = list.copy(multi_methods)

    # At what spacing are the results calculated? Note that this may well affect
    # the results of the momentum space propagators.
    spacing = 1

    # How many results does this produce in a given direction
    no_results = (L // 2) // spacing + 1

    # Produce the array of deltas that will be investigated
    delta_range = numpy.arange(0, L//2 + 1, spacing)

    if cor_type == "Horizontal":
      deltas = [(0, i) for i in delta_range]
    
    if cor_type == "Diagonal":
      deltas = [(i, i) for i in delta_range]

    # Generate the N*M states for the single threaded method
    single_spins = generate_states_1_1(L, beta, N * M, step, initial, initial_spins=numpy.ones((L, L)), rerun=rerun_generation)

    # Generate the split states
    multi_spins = generate_states_splitting(L, beta, N, M, Mstep, splitting, N * M, initial, Nstep=Nstep, source_file_step=step, rerun=rerun_generation)

    # Create a place to store the end data
    directory = f"../data/bootstrap/ising/p_correlator/L{L}/b{beta:.3f}/"

    single_filename = {}
    multi_filename = {}
    for method in single_methods:
      single_filename[method] = f"bootstrap_data_{no_samples}_{cor_type}_single_{method}_L{L}_b{beta:.3f}_N{N}_M{M}_Mstep{Mstep}_sourceN{N * M}_sourceinit{initial}_sourcestep{step}_Nstep{Nstep}_{splitting[0]}_{splitting[1]}_single.npy"
    
    for method in multi_methods:
      multi_filename[method] = f"bootstrap_data_{no_samples}_{cor_type}_multi_{method}_L{L}_b{beta:.3f}_N{N}_M{M}_Mstep{Mstep}_sourceN{N * M}_sourceinit{initial}_sourcestep{step}_Nstep{Nstep}_{splitting[0]}_{splitting[1]}_single.npy"
    
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

    # Create objects that will collect the results that we will generate
    results_single = {}
    results_multi = {}
    for method in single_methods:
      # First check if the result is present already 
      results_single[method] = numpy.zeros((no_samples, no_results))
    
    for method in multi_methods:
      results_multi[method] = numpy.zeros((no_samples, no_results))

    # Perform a bootstrap ======================================================
    for sample in range(no_samples):
      print(f" ============ Calculating bootstrap sample {sample} =============")
      single_dictionary = {}
      multi_dictionary = {}
    
      # # Reorder spins
      single_temp = single_spins[resampling_single[sample]]
      multi_temp = multi_spins[resampling_multi[sample]]

      for method in single_methods:
        single_dictionary[method] = deepcopy(single_temp)
      
      # For getting the onept weights
      single_dictionary["normal"] = deepcopy(single_temp)
      multi_dictionary["normal"] = [deepcopy(multi_temp[:N // 2]), deepcopy(multi_temp[N // 2:])]

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

        # Note that using different weights introduces a non-constant difference
        # between connected_readdition and connected and similarly with flipped
        # This doens't mean the results are biased though.
        if different_weights:
          # Extract onept weights from each spin group
          for i in [0, 1]:
            onept_weights[method][i] = get_onept_weights(multi_dictionary["normal"][i], regions, cap=False)
            onept_weights_sym[method][i] = use_symmetry(onept_weights[method][i], splitting)
            onept_weights_sym[method][i] = numpy.where(regions, numpy.minimum(M, onept_weights_sym[method][i]), 1)
        
        else:
          # Extract onept weights from each normal group for all
          for i in [0, 1]:
            onept_weights[method][i] = get_onept_weights(multi_dictionary["normal"][i], regions, cap=False)
            onept_weights_sym[method][i] = use_symmetry(onept_weights[method][i], splitting)
            onept_weights_sym[method][i] = numpy.where(regions, numpy.minimum(M, onept_weights_sym[method][i]), 1)

      if "flipped" in multi_methods:
        for j in [0, 1]:
          negative = (numpy.mean(multi_dictionary["flipped"][j], axis=(1, 2, 3)) < 0)

          multi_dictionary["flipped"][j][negative] = -multi_dictionary["flipped"][j][negative]

      if "flipped_readdition" in multi_methods:
        for j in [0, 1]:
          negative = (numpy.mean(multi_dictionary["flipped_readdition"][j], axis=(1, 2, 3)) < 0)

          multi_dictionary["flipped_readdition"][j][negative] = -multi_dictionary["flipped_readdition"][j][negative]

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
        if "connected" in multi_methods:
          for j in [0, 1]:
            C[j] = 0.5 * numpy.average(numpy.mean(multi_dictionary["connected"][j], axis=(0,1)), weights=onept_weights_comb["connected"][not j])
            C[j] += 0.5 * numpy.average(numpy.mean(numpy.roll(numpy.roll(multi_dictionary["connected"][j], -delta[1], axis=3), -delta[0], axis=2), axis=(0,1)), weights=onept_weights_comb["connected"][not j])

            multi_dictionary["connected"][j] = multi_dictionary["connected"][j] - C[j]
        
        C2 = [0, 0]
        if "connected_readdition" in multi_methods:
          for j in [0, 1]:
            C2[j] = 0.5 * numpy.average(numpy.mean(multi_dictionary["connected_readdition"][j], axis=(0,1)), weights=onept_weights_comb["connected_readdition"][not j])
            C2[j] += 0.5 * numpy.average(numpy.mean(numpy.roll(numpy.roll(multi_dictionary["connected_readdition"][j], -delta[1], axis=3), -delta[0], axis=2), axis=(0,1)), weights=onept_weights_comb["connected_readdition"][not j])

            multi_dictionary["connected_readdition"][j] = multi_dictionary["connected_readdition"][j] - C2[j]
        
        flipped_C = [0, 0]
        if "flipped" in multi_methods:
          # Average over the M axis as well as the spatial axes of the lattice
          for j in [0, 1]:
            flipped_C[j] = 0.5 * numpy.average(numpy.mean(multi_dictionary["flipped"][j], axis=(0,1)), weights=onept_weights_comb["flipped"][not j])
            flipped_C[j] += 0.5 * numpy.average(numpy.mean(numpy.roll(numpy.roll(multi_dictionary["flipped"][j], -delta[1], axis=3), -delta[0], axis=2), axis=(0,1)), weights=onept_weights_comb["flipped"][not j])
            
            multi_dictionary["flipped"][j] = multi_dictionary["flipped"][j] - flipped_C[j]

        flipped_C2 = [0, 0]
        if "flipped_readdition" in multi_methods:
          for j in [0, 1]:
            flipped_C2[j] = 0.5 * numpy.average(numpy.mean(multi_dictionary["flipped_readdition"][j], axis=(0,1)), weights=onept_weights_comb["flipped_readdition"][not j])
            flipped_C2[j] += 0.5 * numpy.average(numpy.mean(numpy.roll(numpy.roll(multi_dictionary["flipped_readdition"][j], -delta[1], axis=3), -delta[0], axis=2), axis=(0,1)), weights=onept_weights_comb["flipped_readdition"][not j])

            multi_dictionary["flipped_readdition"][j] = multi_dictionary["flipped_readdition"][j] - flipped_C2[j]


        for method in multi_methods:
          # Average over the 2 spin groups, where each group uses the other groups weights
          part0 = twopt_average(multi_dictionary[method][0], regions, onept_weights_comb[method][1], delta)[0]
          part1 = twopt_average(multi_dictionary[method][1], regions, onept_weights_comb[method][0], delta)[0]
          
          results_multi[method][sample, i] = (part1 + part0) / 2

        if "connected_readdition" in multi_methods:
          results_multi["connected_readdition"][sample, i] = results_multi["connected_readdition"][sample, i] + (C2[0] ** 2 + C2[1] ** 2) / 2

          # Return the spins to their original value
          for j in [0, 1]:
            multi_dictionary["connected_readdition"][j] = multi_dictionary["connected_readdition"][j] + C2[j]

        if "flipped_readdition" in multi_methods:
          results_multi["flipped_readdition"][sample, i] = results_multi["flipped_readdition"][sample, i] + (flipped_C2[0] ** 2 + flipped_C2[1] ** 2) / 2
      
          # Return the spins to their original values
          for j in [0, 1]:
            multi_dictionary["flipped_readdition"][j] = multi_dictionary["flipped_readdition"][j] + flipped_C2[j]

        # Similarly, for the non-readdition methods, such a return must be also made
        if "connected" in multi_methods:
          for j in [0, 1]:
            multi_dictionary["connected"][j] = multi_dictionary["connected"][j] + C[j]

        if "flipped" in multi_methods:
          for j in [0, 1]:
            multi_dictionary["flipped"][j] = multi_dictionary["flipped"][j] + flipped_C[j]

    if not os.path.isdir(directory):
      os.makedirs(directory)

    for method in single_methods:
      numpy.save(f"{directory}{single_filename[method]}", results_single[method])

    for method in multi_methods:
      numpy.save(f"{directory}{multi_filename[method]}", results_multi[method])

    # Now load back in the data
    for method in single_methods_copy:
      results_single[method] = numpy.load(f"{directory}{single_filename[method]}")

    for method in multi_methods_copy:
      results_multi[method] = numpy.load(f"{directory}{multi_filename[method]}")

    # Calculate the low momentum mode as this is a proxy for the correlation
    # length
    for method in results_single.keys():
      # Perform a bootstrap
      low_modes = [0, ] * no_samples
      for i in range(no_samples):
        low_modes[i] = numpy.mean(numpy.exp(-1j * (2 * numpy.pi / L) * delta_range) * results_single[method][i]).real * 2
      single_low_mode_s[M][method] = numpy.mean(low_modes)
      single_low_mode_errs[M][method] = numpy.std(low_modes)

    for method in results_multi.keys():
      # Perform a bootstrap
      low_modes = [0, ] * no_samples
      for i in range(no_samples):
        low_modes[i] = numpy.mean(numpy.exp(-1j * (2 * numpy.pi / L) * delta_range) * results_multi[method][i]).real * 2
      multi_low_mode_s[M][method] = numpy.mean(low_modes)
      multi_low_mode_errs[M][method] = numpy.std(low_modes)

    results_single_M[M] = results_single
    results_multi_M[M] = results_multi

  if return_mom_mode:
    return single_low_mode_s, single_low_mode_errs, multi_low_mode_s, multi_low_mode_errs
  
  else:
    return results_single_M, results_multi_M


betas = numpy.linspace(0.1, 0.8, 8)
p = Pool(8)
p.map(realspace, betas)
M_range = numpy.arange(1, 8)

for L in [16]:
  delta_range = numpy.arange(L//2 + 1)
  gradients = []
  xi_s = []
  xi2_s = []
  for beta in betas:
    results_single_M, results_multi_M = realspace(beta)

    ## Find the correlation length - use the data with the least error
    largest = max(results_multi_M.keys())
    results_single = results_single_M[largest]['flipped']

    pstart = numpy.array([1, 1, 0, 0], dtype=float)
    bounds = [(10 ** -6, 10), (10 ** -6, 10 ** 6), (0, 10), (-1, 1)]

    # I want to check that the residuals follow a normal distribution
    # Use the D'Agostino-Pearson test
    p_value = 0
    lower_bound = 1
    while(p_value < 0.05):
      pfit, perr = fit_bootstrap(pstart, delta_range[lower_bound:], results_single[:, lower_bound:], xi_fit2, bounds)

      if len(delta_range) - lower_bound >= 8:
        residuals = numpy.mean(results_single[:, lower_bound:], axis=0) - xi_fit2(delta_range[lower_bound:], pfit)

        statistic, p_value = stats.normaltest(residuals)

        # plt.scatter(delta_range[lower_bound:], numpy.mean(results_single[:, lower_bound:], axis=0))
        # plt.plot(delta_range[lower_bound:], xi_fit2(delta_range[lower_bound:], pfit))
        lower_bound += 1

      else:
        break
    
    # Alternative method of finding xi.. use the Fourier transform
    low_modes = []
    for i in range(results_single.shape[0]):
      low_modes.append(numpy.mean(numpy.exp(-1j * (2 * numpy.pi / L) * delta_range) * results_single[i]).real)
    
    zero_mode = numpy.mean(results_single[i])
    # This is increadibly janky, but for the sake of it, I'm going to add back the
    # constant factor found in the fitting. This is clearly dodgy anyone reading this
    zero_mode -= pfit[3]

    xi2 = numpy.sqrt((1 / (2 * numpy.pi / L) ** 2) * (1 - numpy.mean(low_modes) / zero_mode))
    xi2_s.append(xi2)

    single_errs = numpy.array([numpy.std(results_single_M[M]['flipped'], axis=0)[L//2] for M in 2 ** M_range])
    multi_errs = numpy.array([numpy.std(results_multi_M[M]['flipped'], axis=0)[L//2] for M in 2 ** M_range])

    difference = single_errs / multi_errs

    # Let's fit a gradient to the difference
    coef = numpy.polyfit(M_range, numpy.log2(difference), 1)
    gradients.append(coef[0])
    xi_s.append(pfit[1])

    plot = plt.scatter(M_range, difference, marker='+', label=f"beta = {beta}, xi/L = {pfit[1]/L:.3f} +- {perr[1]/L:.3f}")
    # plot = plt.scatter(M_range, difference, marker='+', label=f"beta = {beta}, xi/L = {xi2/L:.3f}")
    plt.plot(M_range, 2 ** (coef[0] * M_range + coef[1]), color=plot._facecolors[0], ls='--', label=f'grad = {coef[0]:.3f}')
    plt.plot(M_range, numpy.sqrt(2) ** M_range, ls='--', color='k')
    plt.plot(M_range, 1 / (numpy.sqrt(2) ** M_range), ls='--', color='k')
    plt.fill_between(M_range, 1 / (numpy.sqrt(2) ** M_range), numpy.sqrt(2) ** M_range, color='k', alpha=0.02)
    plt.yscale('log')

  plt.legend()

  pdb.set_trace()

  directory = f"../graphs/{str(date.today())}/"
  if not os.path.isdir(directory):
        os.makedirs(directory)

  plt.savefig(f"{directory}gain_L{L}.png")

  plt.close()

  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('beta')
  ax1.set_ylabel('gain_gradient', color=color)
  ax1.plot(betas, gradients, label='gain gradient', color=color)

  # Plot the best and worst case scenarios
  ax1.plot([min(betas), max(betas)], [0.5, 0.5], color='k', ls='--')
  ax1.plot([min(betas), max(betas)], [-0.5, -0.5], color='k', ls='--')
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('xi / L', color=color)  # we already handled the x-label with ax1
  ax2.plot(betas, numpy.array(xi_s) / L, label='correlation length', color=color)

  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  plt.savefig(f"{directory}gain_vs_xi_L{L}.png")

pdb.set_trace()
single_p_s = {}
single_p_errs = {}
multi_p_s = {}
multi_p_errs = {}

pstart = numpy.array([1, 1, 0, 0], dtype=float)
bounds = [(10 ** -6, 10), (10 ** -6, 10 ** 6), (0, 10), (0, 1)]

# Remove delta = 0 from the deltas
deltas_fit = delta_range[delta_range > 0]
for method in results_single.keys():
  pfit, perr = fit_bootstrap(pstart, deltas_fit, results_single[method][:, 1:], xi_fit2, bounds)
  single_p_s[method] = pfit
  single_p_errs[method] = perr

deltas_fit = delta_range[delta_range > 0]
for method in results_multi.keys():
  pfit, perr = fit_bootstrap(pstart, deltas_fit, results_multi[method][:, 1:], xi_fit2, bounds)
  multi_p_s[method] = pfit
  multi_p_errs[method] = perr


plot_range = numpy.linspace(0.1, max(delta_range), 1000)

plt.figure()
# Reload in the data in case it wasn't rerun
pdb.set_trace()
for method in results_single:
  p0, p1, p2, p3 = single_p_s[method]
  plt.errorbar(delta_range, numpy.mean(results_single[method], axis=0), yerr=numpy.std(results_single[method], axis=0), label = f"single - {method}")
  plt.plot(plot_range, xi_fit(plot_range, p0, p1, p2, p3), color='r', ls='--')

for method in results_multi:
  plt.errorbar(delta_range, numpy.mean(results_multi[method], axis=0), yerr=numpy.std(results_multi[method], axis=0), label = f"multi - {method}")

plt.legend()
pdb.set_trace()

plt.close()
for method in results_single:
  plt.plot(delta_range, numpy.std(results_single[method], axis=0), label = f"single - {method}")

for method in results_multi:
  plt.plot(delta_range, numpy.std(results_multi[method], axis=0), label = f"multi - {method}")

plt.legend()

pdb.set_trace()
  


# p = Pool(8)
# betas = [0.2, 0.25, 0.3, 0.325, 0.35, 0.4, 0.45, 0.5]

# p.map(realspace, betas)

# pdb.set_trace()
L = 8
cor_type="Diagonal"
results_single, results_multi, delta_range = realspace(0.2)



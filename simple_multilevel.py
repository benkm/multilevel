from ising2 import *
from multiprocessing import Pool

N = 100
M = 100
splitting = (2, 2)
fitting_fraction = 1 / 4

for L in range(4, 34, 2):
    regions = region_maker(L, splitting)

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

    # Break the spins down into the fitting set and the data set
    spins_fitting = spins[:int(fitting_fraction * N)]
    spins_data = spins[int(fitting_fraction):]

    # Also produce some single level data for comparison. Use N + M samples. 
    single_spins = numpy.random.normal(0, 1, (N + M, 1, L, L))

    # Let's check that the boundarys indeed have a larger standard dev
    means = numpy.mean(numpy.mean(spins, axis=1), axis=0)
    stds = numpy.std(numpy.mean(spins, axis=1), axis=0, ddof=1)

    boundary = regions == 0
    non_boundary = regions != 0

    boundary_av_std = numpy.sum(boundary * stds) / numpy.sum(boundary)
    non_boundary_av_std = numpy.sum(non_boundary * stds) / numpy.sum(non_boundary)

    # The boundary std larger than the non_boundary
    assert boundary_av_std > non_boundary_av_std

    # Extract onept weights from fitting spins
    onept_weights = get_onept_weights(spins_fitting, regions)


    def measuring_function(i):
        delta = (i, i)

        print(f"Calculating for delta = ({i}, {i})")
        
        # Extract twopt weights from the fitting spins
        twopt_weights = get_twopt_weights(spins_fitting, regions, delta)

        result = numpy.array([
        numpy.array(twopt_average(single_spins, regions, numpy.ones((L, L)), delta)),
        numpy.array(twopt_average(spins, regions, numpy.ones((L, L)), delta)),
        numpy.array(twopt_average(spins_data, regions, onept_weights, delta, weight_type=1)),
        numpy.array(twopt_average(spins_data, regions, twopt_weights, delta, weight_type=2))])

        return result

    p = Pool()

    results = numpy.array(p.map(measuring_function, numpy.arange(1, L//2)))

    means_1, stds_1 = results[:, 0, 0], results[:, 0, 1]
    means_2, stds_2 = results[:, 1, 0], results[:, 1, 1]
    means_3, stds_3 = results[:, 2, 0], results[:, 2, 1]
    means_4, stds_4 = results[:, 3, 0], results[:, 3, 1]

    plt.figure()

    plt.errorbar(numpy.arange(1, L//2), means_1, yerr=stds_1, color='k', marker='+', label='single level')
    plt.errorbar(numpy.arange(1, L//2)+0.05, means_2, yerr=stds_2, color='b', marker='+', label='naiive weights')
    plt.errorbar(numpy.arange(1, L//2)+0.1, means_3, yerr=stds_3, color='g', marker='+', label='onept weights')
    plt.errorbar(numpy.arange(1, L//2)+0.15, means_4, yerr=stds_4, color='r', marker='+', label='twopt weights')
    plt.plot([1, L//2 - 1], [0, 0], ls='--', color='gray', label='correct answer')

    plt.legend()
    plt.xlabel("i, where delta = (i, i)")
    plt.ylabel("<phi(x)phi(x + delta)>")
    plt.title(f"L = {L}, N = {N}, M = {M}, splitting = {splitting}, fitting_fraction = {fitting_fraction}")

    plt.savefig(f"../graphs/simple_test_L{L}_N{N}_M{M}_splitting{splitting}.png")

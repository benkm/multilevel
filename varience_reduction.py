import numpy
import pdb
import matplotlib.pyplot as plt

no_samples = 1000
data_size = 100
constant = 100

results = []
new_results = []
results_readdition = []

# Generate random data
data1_start = numpy.random.normal(size=data_size) + constant
data2_start = numpy.random.normal(size=data_size) + constant

# Perform a bootstrap
reshuffle = numpy.random.randint(0, data_size, size=(no_samples, data_size))
for i in range(no_samples):
  data1 = data1_start[reshuffle[i]]
  data2 = data2_start[reshuffle[i]]

  # Simplest way
  results.append(numpy.mean(data1 * data2))

  # Subtract expectation value from each
  new_data1 = data1 - numpy.mean(data1)
  new_data2 = data2 - numpy.mean(data2)

  new_results.append(numpy.mean(new_data1 * new_data2))

  # Add back on the constant factor subtracted
  constant_factor = numpy.mean(data1) * numpy.mean(data2)
  result_readdition = numpy.mean(new_data1 * new_data2 + constant_factor)

  results_readdition.append(result_readdition)

print("--------Standard Method-------")
print(f"mean : {numpy.mean(results)} varience : {numpy.std(results)}")

print("--------New Method-------")
print(f"mean : {numpy.mean(new_results)} varience : {numpy.std(new_results)}")

print("--------Readdition Method-------")
print(f"mean : {numpy.mean(results_readdition)} varience : {numpy.std(results_readdition)}")

pdb.set_trace
## Now let's imagine that there is a signal we want to extract in terms of some
# parameter, e.g. distance

## PARAMTERS
L = 20
xi = 2
constant = 100
data_size = 100 # Number of data points
no_samples = 1000 # Number of bootstrap samples
random_noise_magnitude = 1
onept_noise_magnitude = 10
signal_magnitude = 10

# Try a typical example where there is a onept contribution to a twopt
# function as well as a random noise element, but the onept is dominant
results = numpy.zeros((no_samples, L))
new_results = numpy.zeros((no_samples, L))
results_readdition = numpy.zeros((no_samples, L))
distances = numpy.arange(0, L)

# Each sample will have a random constant
random_onept_start = constant + onept_noise_magnitude * numpy.random.normal(size=data_size)
 
# On top of that assume that there is a source of noise from both sides of
# the correlator
noise_start = random_noise_magnitude * numpy.random.normal(1, size=(data_size, L))

# Perform a bootstrap - use same random numbers as earlier
for i in range(no_samples):
  random_onept = random_onept_start[reshuffle[i]].reshape((data_size, 1)).repeat(L, axis=1)
  noise = noise_start[reshuffle[i]]

  # The signal is composed of these things added together
  signal = random_onept + noise

  twopt = numpy.zeros(L)
  # Calculate the twopt correlator
  for j in range(L):
    # Average over data_size samples and over space
    twopt[j] = numpy.mean(signal * numpy.roll(signal, j, axis=1))
  
  # Now add in the actual signal that real data would contain
  twopt += numpy.exp(-distances / xi) * signal_magnitude

  results[i] = twopt

  pdb.set_trace()
  # Now try subtracting away the onept_function
  signal = signal - numpy.mean(signal)

  twopt = numpy.zeros(L)
  # Calculate the twopt correlator
  for j in range(L):
    # Average over data_size samples and over space
    twopt[j] = numpy.mean(signal * numpy.roll(signal, j, axis=1))
  
  pdb.set_trace()
  
  twopt += numpy.exp(-distances / xi) * signal_magnitude

  new_results[i] = twopt

pdb.set_trace()
plt.plot(distances, numpy.mean(results, axis=0), color='k')
plt.fill_between(distances, numpy.mean(results, axis=0) - numpy.std(results, axis=0),
                            numpy.mean(results, axis=0) + numpy.std(results, axis=0),
                            color='b', alpha=0.05)
plt.show()
plt.plot(distances, numpy.mean(new_results, axis=0))
plt.fill_between(distances, numpy.mean(new_results, axis=0) - numpy.std(new_results, axis=0),
                            numpy.mean(new_results, axis=0) + numpy.std(new_results, axis=0),
                            color='b', alpha=0.05)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# load data
exp_2times_c0_995 = np.load("exp_10times_c0_995_i100.npy")
lin_2times_c0_995 = np.load("lin_10times_c0_995_i100.npy")
log_2times_c0_995 = np.load("log_10times_c0_995_i100.npy")

# plot the data
plt.figure()
plt.plot(np.mean(exp_2times_c0_995, axis=0), label="exp cooling")
plt.plot(np.mean(lin_2times_c0_995, axis=0), label="lin cooling")
plt.plot(np.mean(log_2times_c0_995, axis=0), label="log cooling")
plt.legend()
# make log
# plt.yscale("log")
plt.xlabel("Iterations")
plt.show()



# plot difference of x to x+1
plt.figure()
plt.plot(abs(np.mean(exp_2times_c0_995, axis=0)[1:] - np.mean(exp_2times_c0_995, axis=0)[:-1]), label="exp cooling")

# scatter

plt.plot(abs(np.mean(lin_2times_c0_995, axis=0)[1:] - np.mean(lin_2times_c0_995, axis=0)[:-1]), label="lin cooling")
plt.plot(abs(np.mean(log_2times_c0_995, axis=0)[1:] - np.mean(log_2times_c0_995, axis=0)[:-1]), label="log cooling")
# make log
plt.yscale("log")
plt.ylim(10**-4, 10**4)
plt.legend()
plt.xlabel("Iterations")
plt.show()

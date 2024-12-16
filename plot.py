# plot result.csv
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("results.csv")

# plot all with blue alpha = 0.5
for i in range(len(df)):
    plt.plot(df.iloc[i], 'b', alpha=0.5)
# plot mean blue line width = 2
plt.plot(df.mean(), 'b', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.legend()
plt.show()
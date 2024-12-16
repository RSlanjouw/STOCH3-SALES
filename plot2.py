# ,Cooling Schedule,Iterations,Mean Distance,Standard Error,Distances
# 0,lin,1000,30822.97844543407,30.25908503601314,"[30918.66607406059, 30727.290816807552]"
# 1,lin,10000,27155.430055870027,137.33699255068996,"[27589.727759327783, 26721.132352412267]"
# 2,exp,1000,29677.900305774543,308.72686983611896,"[30654.180389351015, 28701.620222198075]"
# 3,exp,10000,7271.250639539789,66.2762536650879,"[7480.834555904549, 7061.666723175029]"
# 4,log,1000,18052.978406666574,51.09273125621755,"[17891.40900401805, 18214.547809315096]"
# 5,log,10000,6523.311055280889,15.18329503062326,"[6475.297260597804, 6571.324849963975]"

# read exp_10times_c0_995_i100.csv
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("exp_10times_c0_995_i100.csv")
df = df.drop(columns=["Unnamed: 0"])
# plot the std error and mean distance for the lin exp and log cooling schedules
plt.figure()
for i, coolingsc in enumerate(df["Cooling Schedule"].unique()):
    data = df[df["Cooling Schedule"] == coolingsc]
    plt.errorbar(data["Iterations"], data["Mean Distance"], yerr=data["Standard Error"], label=coolingsc)
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Mean Distance")
plt.show()

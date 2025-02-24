import pandas as pd

iter = [1, 2, 3, 4, 5]

df_cpu_list = []
df_gpu_list = []
df_gpu_fremont_list = []
df_cpu_fremont_list = []
for i in iter:
    df = pd.read_csv("vmap_time_macbook_jax_cpu_" + str(i) + ".csv")
    df_cpu_list.append(df)
    df = pd.read_csv("vmap_time_macbook_jax_gpu_" + str(i) + ".csv")
    df_gpu_list.append(df)
    df = pd.read_csv("vmap_time_fremont_jax_gpu_" + str(i) + ".csv")
    df_gpu_fremont_list.append(df)
    df = pd.read_csv("vmap_time_fremont_jax_cpu_" + str(i) + ".csv")
    df_cpu_fremont_list.append(df)
    
df_cpu = pd.concat(df_cpu_list)
df_gpu = pd.concat(df_gpu_list)
df_gpu_fremont = pd.concat(df_gpu_fremont_list)
df_cpu_fremont = pd.concat(df_cpu_fremont_list)

mean_cpu = df_cpu.groupby("Power").mean()
std_cpu = df_cpu.groupby("Power").std()
mean_gpu = df_gpu.groupby("Power").mean()
std_gpu = df_gpu.groupby("Power").std()
mean_gpu_fremont = df_gpu_fremont.groupby("Power").mean()
std_gpu_fremont = df_gpu_fremont.groupby("Power").std()
mean_cpu_fremont = df_cpu_fremont.groupby("Power").mean()
std_cpu_fremont = df_cpu_fremont.groupby("Power").std()

# Plot.
import matplotlib.pyplot as plt
plt.plot(mean_cpu.index, mean_cpu["Time"], 'o-', label="CPU Apple M1")
plt.fill_between(mean_cpu.index, mean_cpu["Time"] - std_cpu["Time"], mean_cpu["Time"] + std_cpu["Time"], alpha=0.3)
plt.plot(mean_cpu_fremont.index, mean_cpu_fremont["Time"], 'o-', label="CPU Intel i9")
plt.fill_between(mean_cpu_fremont.index, mean_cpu_fremont["Time"] - std_cpu_fremont["Time"], mean_cpu_fremont["Time"] + std_cpu_fremont["Time"], alpha=0.3)
plt.plot(mean_gpu.index, mean_gpu["Time"], 'o-', label="GPU Apple M1")
plt.fill_between(mean_gpu.index, mean_gpu["Time"] - std_gpu["Time"], mean_gpu["Time"] + std_gpu["Time"], alpha=0.3)
plt.plot(mean_gpu_fremont.index, mean_gpu_fremont["Time"], 'o-', label="GPU NVIDIA TITAN RTX")
plt.fill_between(mean_gpu_fremont.index, mean_gpu_fremont["Time"] - std_gpu_fremont["Time"], mean_gpu_fremont["Time"] + std_gpu_fremont["Time"], alpha=0.3)
plt.xlabel("Power of 10")
plt.ylabel("Time taken in seconds")
plt.title("Time taken for vmap")
plt.xticks(mean_cpu.index)
plt.grid()
plt.legend()
plt.show()
plt.savefig("vmap_time_macbook_rtx.png")
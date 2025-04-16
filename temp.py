import os
import re
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import tkinter as tk
from tkinter import ttk, messagebox

# --- Setup ---
folder_path = r"C:\Users\jamyd\Desktop\temps\neodc\esacci\land_surface_temperature\data\SSMI_SSMIS\L3C\v2.33\yearly"
date_file_map = {}

# --- Extract year → filename mapping ---
for f in os.listdir(folder_path):
    if f.endswith(".nc"):
        match = re.search(r"(\d{4})000000", f)
        if match:
            year = match.group(1)
            date_file_map[year] = f

# --- GUI ---
root = tk.Tk()
root.title("ESA LST Viewer & Forecast")

# Dropdown
year_var = tk.StringVar()
year_dropdown = ttk.Combobox(root, textvariable=year_var, values=sorted(date_file_map.keys()))
year_dropdown.pack(pady=10)
year_dropdown.set("Select a year")

# --- Load and plot map ---
def load_and_plot():
    year = year_var.get()
    if year not in date_file_map:
        messagebox.showerror("Error", "Please select a valid year.")
        return

    path = os.path.join(folder_path, date_file_map[year])
    ds = xr.open_dataset(path)

    if "lst" not in ds:
        messagebox.showerror("Error", "'lst' variable not found.")
        return

    lst = ds["lst"].isel(time=0)
    lst.plot()
    plt.title(f"Land Surface Temperature for {year}")
    plt.show()

# --- Forecast future using ARIMA ---
def forecast_future():
    yearly_means = []
    for y, fname in sorted(date_file_map.items()):
        ds = xr.open_dataset(os.path.join(folder_path, fname))
        if "lst" in ds:
            mean_lst = ds["lst"].isel(time=0).mean().item()
            yearly_means.append((int(y), mean_lst))

    if len(yearly_means) < 5:
        messagebox.showerror("Error", "Not enough data for ARIMA forecasting.")
        return

    df = pd.DataFrame(yearly_means, columns=["Year", "MeanLST"]).set_index("Year")

    model = ARIMA(df["MeanLST"], order=(1, 1, 1))
    model_fit = model.fit()

    n_years = 10
    forecast = model_fit.forecast(steps=n_years)
    forecast_years = np.arange(df.index.max() + 1, df.index.max() + 1 + n_years)

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["MeanLST"], label="Historical", marker="o")
    plt.plot(forecast_years, forecast, label="ARIMA Forecast", linestyle="--", marker="x")
    plt.title("Mean LST Forecast (ARIMA)")
    plt.xlabel("Year")
    plt.ylabel("Mean Land Surface Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Buttons
plot_button = tk.Button(root, text="Load and Plot Map", command=load_and_plot)
plot_button.pack(pady=5)

forecast_button = tk.Button(root, text="Forecast Future (ARIMA)", command=forecast_future)
forecast_button.pack(pady=5)

# Run the GUI
root.mainloop()

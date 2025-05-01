import yfinance as yf
import pandas as pd

# 1. Define ticker and date range
ticker = "HBL.PS"  # Mari Energies Limited on Yahoo Finance :contentReference[oaicite:3]{index=3}
start_date = "2024-09-16"  # PSX first trading day after Sep 15, 2024 :contentReference[oaicite:4]{index=4}
end_date   = "2025-04-21"  # Today’s date :contentReference[oaicite:5]{index=5}

# 2. Download historical data
df = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    progress=False
)  # Uses yfinance.download under the hood :contentReference[oaicite:6]{index=6}

# 3. Select and rename columns
df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]

# 4. Export to Excel
output_file = "mari_data.xlsx"
df.to_excel(output_file, index=False)
print(f"Excel file saved as: {output_file}")

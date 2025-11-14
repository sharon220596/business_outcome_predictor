# Load dataset
import pandas as pd
df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/TelecomCustomerChurn.csv')

# Preview
df.head()

# Save to CSV locally
df.to_csv("TelecomCustomerChurn.csv", index=False)

print("CSV saved successfully!")

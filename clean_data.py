#Import libraries
import pandas as pd
import numpy as np

# Control Pandas display so large datasets
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

# Load your raw dataset
# low_memory=False stops pandas from guessing column types
df = pd.read_csv("co2_data.csv",low_memory=False)

print(f"Rows : {len(df):,}")
print(f"Cols : {len(df.columns)}")

# Display the first 10 rows to inspect client data
df.head(10)

df.tail(10)

# Display data types and non-null counts for each variable
df.info()

# Display all variable names in the dataset
list(df.columns)

df.shape

# Descriptive statistics for NUMERIC columns
key_numeric = ['Ewltp (g/km)', 'm (kg)', 'ep (KW)', 'Electric range (km)', 'ec (cm3)','year','Mt']

#Display basic descriptive statistics for numeric variables
print(df[key_numeric].describe().round(2))

# descriptive statistics for categorical columns only
df.describe(include='object')

# Check key categorical columns
for col in ['Country', 'year', 'Status','Ft']:
    print(f'{col}' )
    print(df[col].value_counts())
    print()

# Inconsistency Check
print('Fuel Type — Inconsistency Check ')
print(df['Ft'].value_counts())
print() 

# Top 20 manufacturers — check for variants of same brand
# ISSUE: KIA + KIA SLOVAKIA are the same brand (different factories)
print('TOP MANUFACTURERS ')
print(df['Mh'].value_counts().head(25))

# Count the number of missing values per variable
df.isnull().sum()

# Calculate the percentage of missing values per variable
df.isna().sum() / len(df) * 100

print("\n-isnull() — missing value count&pct -")
miss_count = df.isnull().sum()
miss_pct = (miss_count/len(df)*100).round(2)
null = pd.DataFrame({'Missing':miss_count,'Pct %':miss_pct})
print(null[null['Missing']>0].sort_values('Pct %',ascending=False))

#Outlier Detection(IQR method)
# Get CO2 values for petrol/diesel cars only (ignore electric cars)
ice_co2 = df[df['Ewltp (g/km)'] > 0]['Ewltp (g/km)']

# Calculate IQR
Q1 = ice_co2.quantile(0.25)
Q3 = ice_co2.quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR

# Count outliers
outliers = (ice_co2 > upper_fence).sum()

print(f"Q1: {Q1:.0f}, Q3: {Q3:.0f}, IQR: {IQR:.0f}")
print(f"Upper fence: {upper_fence:.0f}")
print(f"Outliers found: {outliers:,}")
print("Decision: Keep all — high CO2 cars are real, not errors")

# Check car mass for outliers
print("\nCar mass summary:")
print(df['Mt'].describe().round(1))
print("Mass looks normal — no outliers")
#Decision: Remove CO2 > 200 g/km for non-electric cars
#Reason: Normal passenger car range is 0–200 g/km; values above are data errors

# Check for duplicate rows

df[df.duplicated()]
 
# count duplicate columns 
print('Duplicate columns:', df.columns.duplicated().sum())


# Always work on a copy of the original data
# Never modify the original dataframe directly
df_clean = df.copy()
print(f'Working copy created: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns')

# Fix Inconsistency — Fuel Type Case 
print('Before fix:')
print(df_clean['Ft'].value_counts())

#Fix fuel type - The same fuel has mixed uppercase/lowercase.
df_clean['Ft'] = df_clean['Ft'].str.upper().str.strip()

# Map to clean display labels
Fuel_map = {
    'PETROL':           'Petrol',
    'DIESEL':           'Diesel',
    'ELECTRIC':         'Electric',
    'PETROL/ELECTRIC':  'Plug-in Hybrid',
    'DIESEL/ELECTRIC':  'Plug-in Hybrid',
    'NG':               'Other',
    'LPG':              'Other',
    'HYDROGEN':         'Other',
    'E85':              'Other',
    'NG-BIOMETHANE':    'Other',
}

df_clean['Ft'] = df_clean['Ft'].map(Fuel_map).fillna(df_clean['Ft'])


#After Fix
print(df_clean['Ft'].value_counts())

#Remove the Outliers 
# Decision: CO2 > 200 g/km for non-electric = data error → DELETE the row
# Weight outside 900–3500 kg = invalid passenger car → DELETE
# Engine power > 700 kW = hypercar, not representative fleet → DELETE
before = len(df_clean)
# CO2 outliers — only for combustion engine cars
# Electric cars legitimately have CO2 = 0, so we exclude them from this rule

df_clean = df_clean[
    (df_clean['Ewltp (g/km)'] <= 200) |
    (df_clean['Ft'] == 'Electric')
]

# Weight outliers — normal passenger cars are 900–3500 kg
df_clean = df_clean[
    (df_clean['m (kg)'] >= 900) &
    (df_clean['m (kg)'] <= 3500)
]
# Engine power outliers
df_clean = df_clean[
    (df_clean['ep (KW)'].isna()) |
    (df_clean['ep (KW)'] <= 700) ]

after = len(df_clean)
print(f'Outliers removed: {before - after:,} rows')

df_clean.shape


# Handle missing values
cols = ['ep (KW)', 'ec (cm3)', 'Electric range (km)']

for col in cols:
    print(col)
    print("Missing:", df_clean[col].isnull().sum())
    print("Percent:", (df_clean[col].isnull().sum() / len(df_clean)) * 100)
    print()

# Impute Missing value - Electric Range
# WHY IT IS MISSING: Petrol/diesel cars have NO electric range — by design, not an error
mask = df_clean['Ft'].isin(['Petrol', 'Diesel', 'Other']) & df_clean['Electric range (km)'].isna()
df_clean.loc[mask, 'Electric range (km)'] = 0

# Missing value— Engine Capacity
# EV cars → engine capacity = 0
df_clean.loc[df_clean['Ft'] == 'Electric', 'ec (cm3)'] = 0

# Find median for non-electric cars
median_value = df_clean[df_clean['Ft'] != 'Electric']['ec (cm3)'].median()

# Fill missing values for non-EV cars
df_clean.loc[df_clean['Ft'] != 'Electric', 'ec (cm3)'] = \
    df_clean.loc[df_clean['Ft'] != 'Electric', 'ec (cm3)'].fillna(median_value)

# Electric range missing for Electric cars
# Find the middle value for Electric cars

median_electric = df_clean[df_clean['Ft'] == 'Electric']['Electric range (km)'].median()
print(f"Middle value for Electric cars: {median_electric} km")

# Fill the blank cells for Electric cars
df_clean['Electric range (km)'] = df_clean['Electric range (km)'].where(
    df_clean['Ft'] != 'Electric',
    df_clean['Electric range (km)'].fillna(median_electric)
)

# Find the middle value for Plug-in Hybrid cars
median_hybrid = df_clean[df_clean['Ft'] == 'Plug-in Hybrid']['Electric range (km)'].median()
print(f"Middle value for Plug-in Hybrid: {median_hybrid} km")

# Fill the blank cells for Plug-in Hybrid cars
df_clean['Electric range (km)'] = df_clean['Electric range (km)'].where(
    df_clean['Ft'] != 'Plug-in Hybrid',
    df_clean['Electric range (km)'].fillna(median_hybrid)
)

# Check it worked
print(f"\nMissing now: {df_clean['Electric range (km)'].isna().sum()}")
# Should show: 0

# Missing value — Engine Power
# Fill missing engine power in df_clean
# df_clean is the dataframe we actually use for the dashboard

median_power = df_clean['ep (KW)'].median()
print(f"Median engine power: {median_power} kW")

df_clean['ep (KW)'] = df_clean['ep (KW)'].fillna(median_power)

# Now check df_clean — this will show 0
print(f"Missing now: {df_clean['ep (KW)'].isnull().sum()}")

#After handle missing values 
# Handle missing values
cols = ['ep (KW)', 'ec (cm3)', 'Electric range (km)']

for col in cols:
    print(col)
    print("Missing:", df_clean[col].isnull().sum())
    print("Percent:", (df_clean[col].isnull().sum() / len(df_clean)) * 100)
    print()

# Group manifacturer variants
# If we keep them separate, the KIA bar in our chart is split in two
# making KIA look smaller than VW when actually they have similar volumes
# Same logic for HYUNDAI (3 variants) and FORD (3 variants)

print('KIA variants before grouping:')
print(df_clean[df_clean['Mh'].str.contains('KIA', na=False)]['Mh'].value_counts())

print('HYUNDAI variants before grouping:')
print(df_clean[df_clean['Mh'].str.contains('HYUNDAI', na=False)]['Mh'].value_counts())

print('FORD variants before grouping:')
print(df_clean[df_clean['Mh'].str.contains('FORD', na=False)]['Mh'].value_counts())

print('BMW variants before grouping:')
print(df_clean[df_clean['Mh'].str.contains('BMW', na=False)]['Mh'].value_counts())

mfr_map = {
    # KIA — Korean manufacturer, Slovakia plant is same brand
    'KIA'                : 'KIA',
    'KIA SLOVAKIA'       : 'KIA',

    # HYUNDAI — Korean manufacturer, Czech and Turkey plants same brand
    'HYUNDAI'            : 'HYUNDAI',
    'HYUNDAI CZECH'      : 'HYUNDAI',
    'HYUNDAI ASSAN'      : 'HYUNDAI',

    # FORD — American manufacturer, German plant same brand
    'FORD WERKE GMBH'    : 'FORD',
    'FORD INDIA'         : 'FORD',
    'FORD MOTOR COMPANY' : 'FORD',

    # BMW — German manufacturer, both legal entities same brand
    'BMW AG'             : 'BMW',
    'BMW GMBH'           : 'BMW',

    # Others — keep as is
    'MERCEDES-BENZ AG'   : 'MERCEDES-BENZ',
    'VOLKSWAGEN'         : 'VOLKSWAGEN',
}

# Map known variants — unmapped names keep their original value
df_clean['Mh'] = df_clean['Mh'].map(mfr_map).fillna(df_clean['Mh'])

print('\nKIA after grouping:')
print(df_clean[df_clean['Mh'] == 'KIA']['Mh'].value_counts())

print('\nHYUNDAI after grouping:')
print(df_clean[df_clean['Mh'] == 'HYUNDAI']['Mh'].value_counts())

print('\nFORD after grouping:')
print(df_clean[df_clean['Mh'] == 'FORD']['Mh'].value_counts())

print('\nBMW after grouping:')
print(df_clean[df_clean['Mh'] == 'BMW']['Mh'].value_counts())

#Check the country column
print(df_clean['Country'].unique())

#Add country full names
country_map = {
    'NO': 'Norway',
    'IE': 'Ireland',
    'FI': 'Finland',
    'GR': 'Greece',
    'HR': 'Croatia'
}

df_clean['Country'] = df_clean['Country'].map(country_map)

#After fix country full name
print(df_clean['Country'].unique())

# Drop unless columns
# and makes the code confusing. We only keep what we actually use.

drop_cols = [
    'ID',            # Internal EEA ID — no analytical value for dashboard
    'VFN',           # Vehicle family number — technical registration code, not needed
    'Man',           # OEM manufacturer name — duplicate of Mh (EU standard name)
    'MMS',           # Registry manufacturer name — 79.7% null AND duplicate
    'Tan',           # Type approval number — technical regulatory code
    'T',             # Vehicle type code — not relevant to sustainability analysis
    'Va',            # Variant code — too granular, we use manufacturer level
    'Ve',            # Version code — too granular, same reason as above
    'Mk',            # Make — duplicate of Mh and Man, redundant
    'Ct',            # Category type approved — regulatory code, not analytical
    'Cr',            # Category registered — same as Ct, not needed
    'Mt',            # WLTP test mass — 10.7% null, using m(kg) instead which has 0 nulls
    'Enedc (g/km)',  # NEDC CO2 measurement — 58.8% null AND deprecated since 2021
    'W (mm)',        # Wheelbase — 31.6% null, irrelevant to CO2 sustainability story
    'At1 (mm)',      # Steering axle width — 31.5% null, irrelevant
    'At2 (mm)',      # Other axle width — 31.8% null, irrelevant
    'Fm',            # Fuel mode — redundant information already in Ft (fuel type)
    'IT',            # Innovative technology code — 62.8% null, too sparse
    'Ernedc (g/km)', # NEDC eco-innovation — 96.3% null, almost entirely empty
    'Erwltp (g/km)', # WLTP eco-innovation — 68.8% null, too sparse for analysis
    'De',            # Deviation factor — 97.7% null, regulatory only
    'Vf',            # Verification factor — 95% null, regulatory only
    'Status',        # All values = F (Final) — zero variation, useless column
    'Date of registration', # 28.8% null — Year column gives us enough time info
    'Fuel consumption ',    # 58.9% null — CO2 is the better sustainability metric
    'ech',           # 86.6% null — electric charging data, too sparse
    'RLFI',          # 95.5% null — regulatory factor, not useful for analysis
    'z (Wh/km)',     # 64.7% null — Electric range (km) column is better (cleaner)
]

# Only drop columns that exist (prevents errors if column name changed)
drop_cols = [c for c in drop_cols if c in df_clean.columns]
df_clean.drop(columns=drop_cols, inplace=True)

print(f'Columns remaining: {df_clean.shape[1]}')
print(f'Columns kept: {df_clean.columns.tolist()}')


#  Rename the columns
# WHY: Column names like 'Ewltp (g/km)' and 'ep (KW)' are EEA internal codes
# Readable names like 'CO2_gkm' and 'Engine_Power_kW' make the code self-documenting
# This also prevents errors when typing column names in app.py

df_clean = df_clean.rename(columns={
    'Country'            : 'Country',
    'Mp'                 : 'Pool',               # Manufacturer pool (group)
    'Mh'                 : 'Manufacturer',        # EU standard manufacturer name
    'Cn'                 : 'Car_Model',           # Commercial name of the car
    'r'                  : 'Registrations',       # Number of registrations
    'm (kg)'             : 'Weight_kg',           # Vehicle mass in kg
    'Ewltp (g/km)'       : 'CO2_gkm',            # CO2 emissions (WLTP standard)
    'Ft'                 : 'Fuel_Type',           # Fuel type
    'ec (cm3)'           : 'Engine_Capacity_cm3', # Engine size
    'ep (KW)'            : 'Engine_Power_kW',     # Engine power
    'Electric range (km)': 'Electric_Range_km',   # EV driving range
    'year'               : 'Year',               # Registration year
})

print(f"Columns now: {list(df_clean.columns)}")

for col in df_clean.columns:
    print(f"\nColumn: {col}")
    print(df_clean[col].unique())

# Feature Enginering— Add EU Target 
# Our equivalent: adding EU Target from regulation — NOT in raw data
# Without this, we cannot build the target comparison chart
# Source: EU Regulation 2019/631 — fleet targets for passenger cars

target_map = {
    2019: 130,  # Old NEDC target transitioning to WLTP
    2020: 115.1,   # New 115.1 g/km WLTP target introduced
    2021: 115.1,
    2022: 115.1,
    2023: 115.1,   # 93.6 g/km from 2025 onwards
}
df_clean['EU_Target_gkm'] = df_clean['Year'].map(target_map)
print(df_clean.groupby('Year')['EU_Target_gkm'].first())

#Feature Enginering— Add EU Target 
# WHY "Discretisation — converting continuous to categorical"
# CO2 g/km (continuous) → category label (discrete) for colour coding charts

def co2_category(co2):
    if co2 == 0:       return 'Zero Emission'
    elif co2 <= 100:   return 'Low (0-100)'
    elif co2 <= 130:   return 'Medium (101-130)'
    elif co2 <= 170:   return 'High (131-170)'
    else:              return 'Very High (170+)'

df_clean['CO2_Category'] = df_clean['CO2_gkm'].apply(co2_category)
print(df_clean['CO2_Category'].value_counts())

print(df.columns)
#Feature Enginering — Is Electric + Gap to Target
#Electric cars list
ev_types = ['Electric', 'Plug-in Hybrid']

# Create Is_Electric column (True/False)
df_clean['Is_Electric'] = df_clean['Fuel_Type'].isin(ev_types)

# Create Gap_to_Target column
df_clean['Gap_to_Target'] = df_clean['CO2_gkm'] - 115.1
# Print results
print("Electric cars:", df_clean['Is_Electric'].sum())
print("EV percentage:", df_clean['Is_Electric'].mean() * 100)


# Create Weight_Category
def weight_category(kg):
    if kg < 1400:    return 'Light (<1400kg)'
    elif kg < 1900:  return 'Medium (1400-1900kg)'
    else:            return 'Heavy (>1900kg)'

df_clean['Weight_Category'] = df_clean['Weight_kg'].apply(weight_category)
print(df_clean['Weight_Category'].value_counts())
df_clean.shape

df_clean.info()

# AGGREGATION/SUMMARISATION
# Group by year → count transactions, average price
# Our equivalent: Group by Country+Manufacturer+Fuel+Year → average CO2, count cars
#
# Result: 592,011 individual car rows → ~900 group summary rows
# This is also Numerosity Reduction
# Reduced data produces SAME results as original

df_agg = df_clean.groupby([
    'Country',
    'Manufacturer',
    'Fuel_Type',
    'Year',
    'Pool',
    'CO2_Category',
    'EU_Target_gkm',
]).agg(
    Avg_CO2_gkm           = ('CO2_gkm',            'mean'),
    Avg_Weight_kg         = ('Weight_kg',           'mean'),
    Avg_Power_kW          = ('Engine_Power_kW',     'mean'),
    Avg_Engine_Cap_cm3    = ('Engine_Capacity_cm3', 'mean'),
    Avg_Electric_Range_km = ('Electric_Range_km',   'mean'),
    Total_Registrations   = ('Registrations',       'sum'),
    Car_Count             = ('CO2_gkm',             'count'),
    EV_Count              = ('Is_Electric',         'sum'),
).reset_index().round(2)

# Calculate EV share % and Gap to Target on aggregated data
df_agg['EV_Share_pct'] = (df_agg['EV_Count'] / df_agg['Car_Count'] * 100).round(1)
df_agg['Gap_to_Target'] = (df_agg['Avg_CO2_gkm'] - df_agg['EU_Target_gkm']).round(2)

# Fill remaining nulls
df_agg['Avg_Engine_Cap_cm3']    = df_agg['Avg_Engine_Cap_cm3'].fillna(0)
df_agg['Avg_Electric_Range_km'] = df_agg['Avg_Electric_Range_km'].fillna(0)

print(f'Aggregation complete (Lecture Slide 44 — Aggregation/Summarisation)')
print(f'Raw rows:        {len(df_clean):,}')
print(f'Aggregated rows: {len(df_agg):,}')
print(f'Reduction:       {(1 - len(df_agg)/len(df_clean))*100:.1f}%')



# Save clean Dataset 
# WHY: app.py reads from a CSV file — it cannot access a notebook directly
# index=False: row numbers are not data — exclude them from the file

df_clean.to_csv('co2_data_clean.csv', index=False)
 
print('Clean dataset saved: co2_data_clean.csv')
print('Rows   :', len(df_clean))
print('Columns:', df_clean.shape[1])
print()

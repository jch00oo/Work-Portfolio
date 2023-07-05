# Food Safety

In this report, I will investigate restaurant food safety scores for restaurants in San Francisco. The scores and violation information have been [made available by the San Francisco Department of Public Health](https://data.sfgov.org/Health-and-Social-Services/Restaurant-Scores-LIVES-Standard/pyih-qa8i).

```python
import numpy as np
import pandas as pd
import math

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('fivethirtyeight')

import zipfile
from pathlib import Path
import os # Used to interact with the file system
```
## Obtaining the Data

### File Systems and I/O
```python
from pathlib import Path
data_dir = Path('.')
data_dir.mkdir(exist_ok = True)
file_path = data_dir / Path('data.zip')
dest_path = file_path
```
I first begin by looking into the organization of the data to answer questions such as the following:

* Is the data in a standard format or encoding?
* Is the data organized in records?
* What are the fields in each record?

### EDA

```python
# looking inside and extracting zip files

my_zip = zipfile.ZipFile(dest_path, 'r')
list_names = my_zip.namelist()

# unzip csv files into subdirectory 'data'

data_dir = Path('.')
my_zip.extractall(data_dir)
!ls {data_dir / Path("data")}
```

## Reading in and Verifying Data

```python
# path to directory containing data
dsDir = Path('data')

bus = pd.read_csv(dsDir/'bus.csv', encoding='ISO-8859-1')
ins2vio = pd.read_csv(dsDir/'ins2vio.csv')
ins = pd.read_csv(dsDir/'ins.csv')
vio = pd.read_csv(dsDir/'vio.csv')

ins_test = ins
```

### Sanity checks
```python
# check basic structure of data frames created

assert all(bus.columns == ['business id column', 'name', 'address', 'city', 'state', 'postal_code',
                           'latitude', 'longitude', 'phone_number'])
assert 6250 <= len(bus) <= 6260

assert all(ins.columns == ['iid', 'date', 'score', 'type'])
assert 26660 <= len(ins) <= 26670

assert all(vio.columns == ['description', 'risk_category', 'vid'])
assert 60 <= len(vio) <= 65

assert all(ins2vio.columns == ['iid', 'vid'])
assert 40210 <= len(ins2vio) <= 40220

# check that statistics match what is expected based on given info

bus_summary = pd.DataFrame(**{'columns': ['business id column', 'latitude', 'longitude'],
 'data': {'business id column': {'50%': 75685.0, 'max': 102705.0, 'min': 19.0},
  'latitude': {'50%': -9999.0, 'max': 37.824494, 'min': -9999.0},
  'longitude': {'50%': -9999.0,
   'max': 0.0,
   'min': -9999.0}},
 'index': ['min', '50%', 'max']})

ins_summary = pd.DataFrame(**{'columns': ['score'],
 'data': {'score': {'50%': 76.0, 'max': 100.0, 'min': -1.0}},
 'index': ['min', '50%', 'max']})

vio_summary = pd.DataFrame(**{'columns': ['vid'],
 'data': {'vid': {'50%': 103135.0, 'max': 103177.0, 'min': 103102.0}},
 'index': ['min', '50%', 'max']})

from IPython.display import display

print('What we expect from Businesses dataframe:')
display(bus_summary)
print('What we expect from Inspections dataframe:')
display(ins_summary)
print('What we expect from Violations dataframe:')
display(vio_summary)
```
The code below defines a testing function used to verify that the data has the same statistics as expected. Run these cells to define the function. The `df_allclose` function has this name because we are verifying that all of the statistics for the dataframe are close to the expected values.

```python
"""Run this cell to load this utility comparison function that we will use in various
tests below (both tests you can see and those we run internally for grading).

Do not modify the function in any way.
"""


def df_allclose(actual, desired, columns=None, rtol=5e-2):
    """Compare selected columns of two dataframes on a few summary statistics.

    Compute the min, median and max of the two dataframes on the given columns, and compare
    that they match numerically to the given relative tolerance.

    If they don't match, an AssertionError is raised (by `numpy.testing`).
    """    
    # summary statistics to compare on
    stats = ['min', '50%', 'max']

    # For the desired values, we can provide a full DF with the same structure as
    # the actual data, or pre-computed summary statistics.
    # We assume a pre-computed summary was provided if columns is None. In that case,
    # `desired` *must* have the same structure as the actual's summary
    if columns is None:
        des = desired
        columns = desired.columns
    else:
        des = desired[columns].describe().loc[stats]

    # Extract summary stats from actual DF
    act = actual[columns].describe().loc[stats]

    return np.allclose(act, des, rtol)
```

## Data Cleaning

Business dataframe
```python
# rename bid column
bus = bus.rename(columns={"business id column": "bid"})

# filtering out invalid ZIP codes
valid_zips = pd.read_json('data/sf_zipcodes.json', dtype = str)['zip_codes']
invalid_zip_bus = bus[~bus['postal_code'].isin(valid_zips)]

bus['postal5'] = bus['postal_code'].str.slice(0,5)
bus.loc[~bus['postal5'].isin(valid_zips), 'postal5'] = None
```

Inspection dataframe
```python
# split iid column to extract bid
ins['bid'] = ins['iid'].str.split("_")[:].apply(lambda x: x[0]).astype('int64')
```

Merging dataframes
```python
# filter out missing scores so negative scores don't influence results
ins = ins[ins["score"] > 0]

# create new dataframe ins_named; same as ins except it should have the name and address of every business
# ins with name and address; if dne, NaN
ins_named = pd.merge(ins, bus, on = 'bid').drop(columns = ['city', 'state','postal_code','latitude','longitude','phone_number', 'postal5'])
```

# Exploring Inspection scores

I begin by looking at the distribution of inspection scores.
```python
fig, ax = plt.subplots()
score_counts = ins['score'].value_counts().plot(ax=ax, kind='bar', title = 'Distribution of Inspection Scores', xlabel = 'Score', ylabel = 'Count')
```
![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/52dc5881-770f-4151-a76d-2c2007d34291)
The model has a left tail with no symmetry. Even though the general trend of having an upward curve holds, there are weird values, namely at 91 and 93, where it goes down randomly. But generally, this implies there are more restaurants with higher inspection scores

## Restaurant Ratings Over Time

### Relationship between first and second scores for the businesses with 2 inspections in a year

I focus on the year 2018 for this question.

```python
ins2018 = ins[ins['year'] == 2018]
# Create the dataframe here

# find ones with 2 inspections
scores_pairs_by_business = ins2018.sort_values('iid').groupby('bid').filter(lambda df: df['score'].size == 2).groupby('bid')['score'].agg(lambda df: [df.iloc[0], df.iloc[1]]).to_frame().rename(columns={'score':'score_pair'})

# scatter plot
plt.scatter(*zip(*scores_pairs_by_business['score_pair']), facecolors = 'none', edgecolors = 'b')
plt.plot([55,100],[55,100], color = 'r')
plt.xlim(55, 100)
plt.ylim(55,100)
plt.xlabel('First Score')
plt.ylabel('Second Score')
plt.title('First Inspection Score vs Second Inspection Score')
```
![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/5419f1dd-9d85-43b4-ab17-bc2fd40ac862)

If restaurants' scores tend to improve from the first to the second inspection, then most of the points in the previous scatter plot should be above the y = x red line. However, such is not the case as the points seem to be evenly distributed between above and below the red line.

### Distribution of restaurant scores over time
```python
# create boxplots to show distribution of restaurant scores over time

sns.set()

risk_ins = pd.merge(ins2vio, ins, on = 'iid').merge(vio, on = 'vid')

plt.figure(figsize=(12,8))
lyn = risk_ins[risk_ins["year"]>2016]
sns.boxplot(x = lyn["year"], y = lyn["score"], hue = lyn["risk_category"], hue_order = ["Low Risk","Moderate Risk","High Risk"]);
plt.show()
```
![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/d3894922-77d1-4ca8-9c6e-a53793c7dd24)

## Relationship between location and average scores
```python
busIns = pd.merge(bus, ins_named, on = 'name')

valid_coord = busIns[~np.isclose(busIns['latitude'], -9999)]

mean_valid_coord = valid_coord.groupby('bid_x').mean()

mean_valid_coord = mean_valid_coord[~np.isclose(mean_valid_coord['latitude'], 0)]

plt.figure(figsize=(12,8))
plt.scatter(mean_valid_coord.latitude, mean_valid_coord.longitude, c = ((mean_valid_coord['score'])/100), cmap = plt.cm.bwr)
plt.xlim(min(mean_valid_coord['latitude']),max(mean_valid_coord['latitude']))
plt.ylim(min(mean_valid_coord['longitude']),max(mean_valid_coord['longitude']))
"""
Closer to red: higher average score
Closer to blue: lower average score
"""
```
![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/7865517c-f4b1-4dd3-a897-63ffd7f0cdf5)
The visualization seems to show there is a higher population of higher scored restaurants in the top right area.
However, it is hard to say if such really is the case as there is also higher population of restaurants in the top right area in general. Regardless, there is no doubt that there are much more red colored restaurants (highly scored) than blue color restaurants (low scored) in the top right area, compared to the left side where the colors are slightly fainter which indicates middle range scores.

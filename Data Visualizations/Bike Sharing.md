# Bike Sharing

## Introduction

Bike sharing systems are a new generation of traditional bike rentals where the process of signing up, renting and returning is automated. Through these systems, users are able to easily rent a bike from one location and return them to another. I will be analyzing bike sharing data from Washington D.C.

```python
# Run this cell to set up notebook
import seaborn as sns
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path
import ds100_utils

# Default plot configurations
%matplotlib inline
plt.rcParams['figure.figsize'] = (16,8)
plt.rcParams['figure.dpi'] = 150
sns.set()

import warnings
warnings.filterwarnings("ignore")

from IPython.display import display, Latex, Markdown
```

## Loading Bike Sharing Data
The data we are exploring is collected from a bike sharing system in Washington D.C.

The variables in this data frame are defined as:

Variable       | Description
-------------- | ------------------------------------------------------------------
instant | record index
dteday | date
season | 1. spring <br> 2. summer <br> 3. fall <br> 4. winter
yr | year (0: 2011, 1:2012)
mnth | month ( 1 to 12)
hr | hour (0 to 23)
holiday | whether day is holiday or not
weekday | day of the week
workingday | if day is neither weekend nor holiday
weathersit | 1. clear or partly cloudy <br> 2. mist and clouds <br> 3. light snow or rain <br> 4. heavy rain or snow
temp | normalized temperature in Celsius (divided by 41)
atemp | normalized "feels-like" temperature in Celsius (divided by 50)
hum | normalized percent humidity (divided by 100)
windspeed| normalized wind speed (divided by 67)
casual | count of casual users
registered | count of registered users
cnt | count of total rental bikes including casual and registered

<sup style="display: inline-block;">**tip:** click on the pencil icon on the left to clear the editor)</sup>

## Data Preparation

Decode the `holiday`, `weekday`, `workingday`, and `weathersit` fields:

1. `holiday`: Convert to `yes` and `no`
1. `weekday`: It turns out that Monday is the day with the most holidays.  Mutate the `'weekday'` column to use the 3-letter label (`'Sun'`, `'Mon'`, `'Tue'`, `'Wed'`, `'Thu'`, `'Fri'`, and `'Sat'` ...) instead of its current numerical values. Assume `0` corresponds to `Sun`, `1` to `Mon` and so on, in order of the previous sentence.
1. `workingday`: Convert to `yes` and `no`.
1. `weathersit`: Replace each value with one of `Clear`, `Mist`, `Light`, or `Heavy`. Assume `1` corresponds to `Clear`, `2` corresponds to `Mist`, and so on in order of the previous sentence.

```python
# Modify holiday weekday, workingday, and weathersit here
factor_dict = {'holiday': {0: 'no', 1: 'yes'}, 'workingday': {0: 'no', 1: 'yes'}, 'weekday': {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}, 'weathersit': {1: 'Clear', 2: 'Mist', 3: 'Light', 4: 'Heavy'}}

bike.replace(factor_dict, inplace=True)
```

Next I construct a data frame named `daily_counts` indexed by `dteday` with the following columns:
* `casual`: total number of casual riders for each day
* `registered`: total number of registered riders for each day
* `workingday`: whether that day is a working day or not (`yes` or `no`)

```python
daily_counts = bike.groupby('dteday').agg({'casual':'sum', 'registered':'sum', 'workingday': 'first'})
```

# EDA
I begin the exploratory data analysis by comparing the distribution of the daily counts of casual and registered riders through various visualizations.

```python
sns.distplot(daily_counts['casual'], label = 'casual')
sns.distplot(daily_counts['registered'], color = 'green', label = 'registered')
plt.xlim(-2000, 8000)
plt.title('Distribution Comparison of Casual vs Registered Riders')
plt.xlabel('Rider Count')
plt.ylabel('Density')
plt.legend()
```

![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/750fde59-c70b-4aca-8e1f-08ddbd6de1b4)

- Density curve for 'Casual' is more right skewed with a right tail, less symmetrical, has a mode closer to 0 and near the 1000 mark, has a dip around the 500 mark, with outliers on the right tail compared to the 'Registered' density curve
- Density curve for 'Registered' has mode near the 4000 mark, is more symmetrical with no obvious skewness or tail, no gaps, and more outliers near 0
<br>
<br>
The 'Registered' density curve has greater variability as it's more spread out (the data ranges from 0 to around 7000 whereas 'Casual' ranges from 0 to around 3500).

```python
# Make the font size a bit bigger
sns.set(font_scale=1)
sns.lmplot(x = 'casual', y = 'registered', hue = 'workingday', data = bike, scatter_kws={"s": 1}, fit_reg=True, height = 4, truncate = False).set(xlim=(-50,400), ylim = (-200, 1600))
plt.title('Comparison of Casual vs Registered Riders on Working and Non-working Days')
```

![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/0b343ab6-5e6e-4873-9acb-b12523f78d86)

The scatterplot seems to reveal that for non working days (weekends), there is a stronger linear relationship between the casual and registered riders, whereas for working days, there isn't as strong of a relationship between the two. Overplotting makes it harder to make this judgement as there are a lot of overlapping data points plotted in the left corner where x = [0, 100] and y = [0, 400].

# Data Visualizations

```python
# Set the figure size for the plot
plt.figure(figsize=(12,8))

# Set 'is_workingday' to a boolean array that is true for all working_days
is_workingday = np.array( daily_counts['workingday'] == 'yes')

# Bivariate KDEs require two data inputs.
# In this case, we will need the daily counts for casual and registered riders on workdays
# Hint: consider using the .loc method here.
casual_workday = daily_counts.loc[is_workingday, 'casual']
registered_workday = daily_counts.loc[is_workingday, 'registered']

# Use sns.kdeplot on the two variables above to plot the bivariate KDE for weekday rides
sns.kdeplot(x=casual_workday, y=registered_workday, cmap = "Reds", data = daily_counts, label = "Workday")

not_workingday = np.array( daily_counts['workingday'] == 'no')
# Repeat the same steps above but for rows corresponding to non-workingdays
# Hint: Again, consider using the .loc method here.
casual_non_workday = daily_counts.loc[not_workingday, 'casual']
registered_non_workday = daily_counts.loc[not_workingday, 'registered']

# Use sns.kdeplot on the two variables above to plot the bivariate KDE for non-workingday rides
sns.kdeplot(x=casual_non_workday, y=registered_non_workday, cmap = "Blues", label = "Non-Workday")

plt.legend()
plt.title('KDE Plot Comparing Registered vs Casual Riders')
```

![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/c85d1ec4-3611-45ac-b31c-a796d4944256)

I like to see this KDE plot like a topographic map; the smaller circles/lines represent higher density of data while the outer, white line represents the range of the data. The opacity of the color also signifies density, so the stronger the color, the more data points there are within it.
Unlike the previous scatter plot, it is easier to determine where points are densely plotted in this plot. In addition, it makes it easier to see the general trend of the data outside of density, as in the previous scatter plot it was harder to see the relationship between casual and registered riders on working days.

```python
sns.jointplot(data = daily_counts, x = 'casual', y = 'registered', kind = 'kde')

plt.suptitle("KDE Contours of Casual vs Registered Rider Count")
plt.subplots_adjust(top=0.9);
```

![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/f08cd36e-fc23-4774-899e-83ac7dd741ea)

## Understanding Daily Patterns
```python
time_bike = bike.groupby('hr').mean()

sns.lineplot(data = time_bike, x = 'hr', y = 'casual', label = 'casual')
sns.lineplot(data = time_bike, x = 'hr', y = 'registered', label = 'registered')

plt.xlabel('Average Count')
plt.ylabel('Hour of the Day')

plt.title('Average Count of Casual vs Registered by Hour')
```
![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/0ae4e54b-1970-4e86-85b1-adee99497658)

Registered users' bike usage peak in the morning (around 8 am) and in the evening (around 6 pm) while casual riders have a a smooth and low peak around 3 pm. I hypothesize that registered users are more regular users who bike in the morning before work and then in the evening after work, while casual riders rent bikes mainly in their free time/when trying to relax, hence the early evening/afternoon peak.

## Exploring Ride Sharing and Weather
```python
bike['prop_casual'] = bike['casual']/bike['cnt']

plt.figure(figsize=(10, 7))
sns.scatterplot(data=bike, x="temp", y="prop_casual", hue="weekday");

# attempt linear regression
sns.lmplot(data=bike, x="temp", y="prop_casual", hue="weekday", scatter_kws={"s": 20}, height=10)
plt.title("Proportion of Casual Riders by Weekday");
```
![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/7e9f170c-c1a4-48ec-b527-90a7b0c584b2)

This plot is not ideal due to the sheer amount of data that is aggregated; even when overlaying linear regression on top, the line hints at some relationships between temperature and proportional casual but the plot is still fairly unconvincing.

I instead use local smoothing. The basic idea is that for each x value, I compute some sort of representative y value that captures the data close to that x value. One technique for local smoothing is "Locally Weighted Scatterplot Smoothing" or LOWESS.

```python
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.figure(figsize=(10,8))

#Mon
xnoise_mon = np.array(bike.loc[bike['weekday'] == 'Mon', 'temp'])
ynoise_mon = np.array(bike.loc[bike['weekday'] == 'Mon', 'prop_casual'])

ysmooth_mon = lowess(ynoise_mon, xnoise_mon, return_sorted = False)
sns.lineplot(xnoise_mon, ysmooth_mon, label = 'Mon')

#Tue
xnoise_Tue = np.array(bike.loc[bike['weekday'] == 'Tue', 'temp'])
ynoise_Tue = np.array(bike.loc[bike['weekday'] == 'Tue', 'prop_casual'])

ysmooth_Tue = lowess(ynoise_Tue, xnoise_Tue, return_sorted = False)
sns.lineplot(xnoise_Tue, ysmooth_Tue, label = 'Tue')

#Wed
xnoise_Wed = np.array(bike.loc[bike['weekday'] == 'Wed', 'temp'])
ynoise_Wed = np.array(bike.loc[bike['weekday'] == 'Wed', 'prop_casual'])

ysmooth_Wed = lowess(ynoise_Wed, xnoise_Wed, return_sorted = False)
sns.lineplot(xnoise_Wed, ysmooth_Wed, label = 'Wed')

#Thu
xnoise_Thu = np.array(bike.loc[bike['weekday'] == 'Thu', 'temp'])
ynoise_Thu = np.array(bike.loc[bike['weekday'] == 'Thu', 'prop_casual'])

ysmooth_Thu = lowess(ynoise_Thu, xnoise_Thu, return_sorted = False)
sns.lineplot(xnoise_Thu, ysmooth_Thu, label = 'Thu')

#Fri
xnoise_Fri = np.array(bike.loc[bike['weekday'] == 'Fri', 'temp'])
ynoise_Fri = np.array(bike.loc[bike['weekday'] == 'Fri', 'prop_casual'])

ysmooth_Fri = lowess(ynoise_Fri, xnoise_Fri, return_sorted = False)
sns.lineplot(xnoise_Fri, ysmooth_Fri, label = 'Fri')

#Sat
xnoise_Sat = np.array(bike.loc[bike['weekday'] == 'Sat', 'temp'])
ynoise_Sat = np.array(bike.loc[bike['weekday'] == 'Sat', 'prop_casual'])

ysmooth_Sat = lowess(ynoise_Sat, xnoise_Sat, return_sorted = False)
sns.lineplot(xnoise_Sat, ysmooth_Sat, label = 'Sat')

#Sun
xnoise_Sun = np.array(bike.loc[bike['weekday'] == 'Sun', 'temp'])
ynoise_Sun = np.array(bike.loc[bike['weekday'] == 'Sun', 'prop_casual'])

ysmooth_Sun = lowess(ynoise_Sun, xnoise_Sun, return_sorted = False)
sns.lineplot(xnoise_Sun, ysmooth_Sun, label = 'Sun')

plt.legend()
plt.xlabel('Temperature (Celsius)')
plt.ylabel('Causal Rider Proportion')
```

![image](https://github.com/jch00oo/Work-Portfolio/assets/67119923/1dc5b462-11b1-419c-bc7c-ecb49dfe24cc)

In general, it seems that as temperature increases, the proportion of casual riders increases. Such increase is especially noticeable on the weekend, with Saturday and Sunday overall having higher proportions of casual riders as well as a higher rate of increase of proportion of casual riders as temperature increases.

# Future Steps

One interesting expansion of the analysis would be to assess equity in transportation. Equity in transportation includes: improving the ability of people of different socio-economic classes, genders, races, and neighborhoods to access and afford the transportation services, and assessing how inclusive transportation systems are over time.

However, the current data as it is cannot help assess equity as the `bike` dataset doesn't show any indications about the demographics of the renters. To make the dataset more fit to conduct such analysis, I would change the granularity so that each row represents individual renters, including information about their demographics (race, gender, neighborhood/zip code, etc) and the date/time they rented bikes. I feel things such as working day, weather, windspeed, casual and registered will not be useful for this purpose, so I would get rid of those variables.

Another interesting observation is that [bike sharing is growing in popularity](https://www.bts.gov/newsroom/bike-share-stations-us) and new cities and regions are making efforts to implement bike sharing systems that complement their other transportation offerings. The [goals of these efforts](https://www.wired.com/story/americans-falling-in-love-bike-share/) are to have bike sharing serve as an alternate form of transportation in order to alleviate congestion, provide geographic connectivity, reduce carbon emissions, and promote inclusion among communities.

Based on these plots, in order to exapnd bike sharing, I'd first recommend expanding to warmer urban cities. From the plot in 6b, we see that as temperature increases, the proportion of casual riders also increase. Since casual riders are likely to convert to registered, regular users, targeting warmer cities to increase overall casual rider proportion would be benefitial in the long run. In addition, expanding to urban cities would help with increasing registered users as according to the plot from 5a, my understanding was that registered users used the bike to get to and from work. Hence targeting urban cities will increase those target audience of workers that will register and use the bikes regularly during those peak hours.

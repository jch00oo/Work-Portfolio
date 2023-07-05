# Helpful links for data

* [2020 Georgia General Election Results](https://results.enr.clarityelections.com/GA/105369/web.264614/#/summary)
* [2020 Georgia Recount Results](https://results.enr.clarityelections.com/GA/107231/web.264614/#/summary)
* [2016 Georgia General Election Results](https://results.enr.clarityelections.com/GA/63991/184321/en/summary.html)
* [2020 Presidential Election Lives Results; tabulations for early balloting vs final tabulations](https://www.270towin.com/2020-election-results-live/)


```python
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
```

**Problem context** - claims of fraud in the 2020 US Elections: Charles J. Cicchetti, Ph.D., filed a declaration in the recent Supreme Court challenge to the election results in several states. Among other things, Cicchetti discusses hypothesis tests about the vote in Georgia. Paragraph 11 reports a Z-score of 396.3; paragraph 12 reports a Z-score of 108.7; and paragraph 15 reports a Z-score 1,891.

# Hypothesis Tests

<br>The first null hypothesis is that Clinton and Biden in the 2016 and 2020 elections respectively would have an equal number of votes being that other things were the same. <br> The second null hypothesis is that the two candidates received the same percentage of votes in their election. <br> The last null hypothesis is that the distribution of the votes tabulated in the two time periods (early and late) were similar.

<br> The Z-score was calculated, according to Cicchetti, by taking the "difference between the two candidates' mean values divided by the square root of the sum of their respective variances," and the variances are estimated by "multiplying the mean times the probability of the candidate not getting a vote" (Cicchetti, 3). For the other Z-score, I calculated it using the formula in the [textbook](https://ucb-stat-159-s21.github.io/site/Notes/tests.html#example-of-an-approximate-z-test-with-a-built-in-type-iii-error) but replacing $\bar{x}$ and $\bar{y}$ with proportions or "probabilities" accordingly.

<br> The null hypotheses Cicchetti proposes would indicate whether the number of votes and proportion of votes Biden received in the 2020 election -- based on the assumption that its sample is similar to that of Clinton's voting sample in 2016 as well as in comparing the early and subsequent voters sample -- were unlikely or not.

<br> If the null hypothese are false, it does not imply there was error or fraud in 2020; it simply implies that the results we saw of Biden getting majority votes were unlikely and can cast doubt on the results.

# Data

<br> I found Georgia's election results -- from 2016 and 2020, including the recount -- from the [Secretary of State in Georgia website](https://sos.ga.gov/index.php/Elections/current_and_past_elections_results). For the data to make calculations comparing election results in the different time periods, I used the [270towin.com Live Results link](https://www.270towin.com/2020-election-results-live/) and the [Wayback Machine](https://archive.org/web/), though I couldn't find the exact data and times as Cicchetti used in his declaration and hence used the closest date/time available. 

<br> I was able to find the same inputs as Cicchetti for the first two hypotheses/tests from the Secretary of State in Georgia website, using the 2016 results and the General Election (not recount) 2020 results. For the tabulations comparing early and subsequent votes in Georgia, I could not find the same inputs Cicchetti did; Cicchetti used data from 3:10 AM EST on November 4 and 2 PM EST on November 18 but I used the closest I could find which were 3:24 AM EST on November 4 and 9:30 PM EST on November 14. 

<br> I was able to replicate all the Z-scores except for the Z-score reported in paragraph 15. I suspect this is because I wasn't able to retrieve the same data/inputs Cicchetti did for this specific test, unlike the other two for which I could find the same inputs Cicchetti did.


```python
# Results of the election in Georgia in 2016

# proportion
trump_2016_p = 0.5105
clinton_2016_p = 0.4589
third_2016_p = 0.0306

# number of votes
trump_2016 = 2089104
clinton_2016 = 1877963
third_2016 = 125306
total_2016 = 4092373
```


```python
# Results of the election in Georgia in 2020 (not recount)

# proportion
trump_2020_p = 0.4925
biden_2020_p = 0.4951
third_2020_p = 0.0124

# number of votes
trump_2020 = 2461837
biden_2020 = 2474507
third_2020 = 62138
total_2020 = 4998482
```


```python
# trying to replicate Ciccheti's first z statistic of 396.3

# estimate variance by multiplying mean times the probability of candidate not getting a vote
biden_var = (1 - biden_2020_p)*biden_2020
clinton_var = (1 - clinton_2016_p)*clinton_2016

# calculate z score by the difference between the two candidates' mean values divided by
# the square root of the sum of their respective variances

cicchetti_z_1 = (biden_2020 - clinton_2016) / np.sqrt(biden_var + clinton_var)
cicchetti_z_1
```




    396.32931485932113




```python
# trying to replicate Cicchetti's second z statistic of 108.7

p_hat = (clinton_2016 + biden_2020) / (total_2016 + total_2020)

# calcuate z score using formula on this page (https://ucb-stat-159-s21.github.io/site/Notes/tests.html#example-of-an-approximate-z-test-with-a-built-in-type-iii-error)
# but using proportions of each candidate instead of x bar and y bar

var_diff = (p_hat*(1 - p_hat))*((1/total_2020)+(1/total_2016))

cicchetti_z_2 = (biden_2020_p - clinton_2016_p) / np.sqrt(var_diff)
cicchetti_z_2
```




    108.70125092292297




```python
# Results of election in Georgia at two different time points

# 3:24 AM EST on November 4
biden_early_GA_p = 0.483
trump_early_GA_p = 0.505

biden_early_GA = 2280258
trump_early_GA = 2382070
early_GA_total = 4719768

# 9:30 PM EST on November 14
biden_sub_GA_p = 0.494
trump_sub_GA_p = 0.493

biden_sub_GA = 2456845
trump_sub_GA = 2452825
sub_GA_total = 4971342
```


```python
# trying to replicate Cicchetti's third z statistic of 1891

# Compare distribution of Biden's votes in the two time poitns

p_hat_time = (trump_early_GA + trump_sub_GA) / (early_GA_total + sub_GA_total)

# calculate z score using same method of second z score

var_diff_time = (p_hat_time*(1 - p_hat_time))*((1/early_GA_total)*(1/sub_GA_total))

cicchetti_z_3 = (trump_early_GA_p - trump_sub_GA_p) / np.sqrt(var_diff_time)
cicchetti_z_3
```




    116254.41067838951



# Z tests

<br> The assumptions of a Z-test are that the samples must be independent and that the population from which the samples are taken must have a normal distribution with a known standard deviation, or have large sample sizes.
<br> The complete null hypothesis for a Z-test is that for some given test statistic $f(x)$, $f(X)$ ~ $N(0,1)$, meaning that some function of the data has a standard Normal distribution.
<br> Though Cicchetti seems to think that all the assumptions are met, I feel that the independent samples condition is not met. Voters' political preferences and affiliations are heavily influenced by their location and their general environment, so it is difficult to assume that these samples are independent.

# Z tests in Cicchetti’s analysis

Cicchetti’s analysis assumes that votes are random and there is a given probability that each voter will vote for the Democratic candidate, independent of all other voters. Suppose that’s true.

<br> According to Cicchetti, it seems the probability distribution of the number of votes for the Democratic candidate is a Bernoulli distribution with parameter $p$ being the proportion of voters who voted Democratic. Hence each voter is represented by indicator $I_{j}$ which equals to 1 with probability $p$ when they vote Democratice.

<br> The probability distribution of the difference in vote shares across elections would be the difference of two Binomial distributions, each with parameters $n$ total number of voters and $p$ proportion of voters that voted Democratic/Republican who are independent of each other as he makes the assumption that each voter groups/samples are independent and hence would produce the same voting results everytime. 

<br> I do believe the normal approximation to that distribution is accurate to a part in a quintillion. The Binomial distributions meet the condition to use normal approximation, since $n$ is sufficiently large (millions) and $p$ is near $0.5$ (population distribution is approximately "bell shaped").

The strength of Cicchetti's analysis is that he attempts to account for different variables in the data, such as comparing proportions to account for the increased number of ballots casted in the 2020 election. He also uses readily available and reliable data, citing his sources clearly in the footnotes to make them  as accessible as possible.

# Shortcomings

There are many shortcoming to Cicchetti's analysis, mostly in flawed assumptions and models used throughout the process.
* His statement that "other things being the same they would have an equal number of votes" crosses many assumptions that I cast a lot of doubt about. According to him, the two groups of voters (2016 vs 2020 and early vs subsequently counted ballots) are similar and thus assumes any two large groups should yield similar results. However, so many different factors can go into influencing these results. For example, voting/party preference can greatly vary by location; Fulton County near Atlanta votes overwhelmingly more Democratic compared to Charlton County. And during the 2020 election we saw that Democratic voters tended to cast mail-in ballots (which were counted later) instead of in-person. In addition, the political scene and changes during the 2016-2020 presidential term would have also greatly influenced individuals' voting preference. Hence comparing such voter samples with the assumption that they would vote similarly is very flawed.
* I also feel that his treatment of each voter's decision to vote as being independent and binomial (like flipping a biased coin before casting their vote that would decide whether each individual would vote one candidate or another) is unreasonable. It seems very unlikely that each voter's decisions are independent of one another; the household or location each individual lives in as well as their education or race may all bring a lot of influene in their decisions. So treating each person as being independent and having the same "probabilities" of voting each candidate would not make sense.

# How strong is Cicchetti’s argument that the reported results in Georgia are wrong?

Cicchetti's argument that the reported results in Georgia are wrong is not very strong. The claims he makes about the voter population/samples jump through a lot of assumptions about the distribution of the samples regarding their voting preference/affiliations. The tests and analyses heavily rely on the assumption that the votes were uniformly distributed and independent amongst every sample and county, hence the ability to compare them to understand the differences in number of votes/proportions we saw. However, these assumptions are flawed; absentee/late ballots tended to be from larger cities which leaned towards Democratic, and the voter population in 2016 and 2020 are not comparable. Because these assumptions are so flawed, Cicchetti's argument cannot have strong, reliable conclusions.

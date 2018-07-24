---
layout: post
title: A Hypothetical Presentation to The College Board to Increase SAT Participation Rates
date: 2017-05-19
published: true
categories: projects
tags:
---
Here is the first project given to us at the General Assembly data science immersion program. We were given two databases of 2017 SAT and ACT average scores by state. We were given detailed tasks involving cleaning and exploring the data, and then told to make presentations to the College Board to suggest ways of increasing participation. Here is the presentation I made with my suggestions, followed by the notebook I used.

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vSVDp-ccEE5h_Tc21rCYAETyhq23CQ_zeRwmJ4zasy4HGsBWL7xhxfUDsmJbKE46aMIrYcjmVqb0QE3/embed?start=true&loop=true&delayms=5000" frameborder="0" width="720" height="434" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

This blog will be mostly a detailed desciption of my code along with my resulting presentation at the end. However, I have taken the liberty of deleting many cells that didn't really end up contributing to the final presentation as well as for the sake of readibility. 

Please keep in my that I was able to do this work after just two weeks of instruction.  

Loading in the file:


```python
import warnings
warnings.simplefilter("ignore")
import pandas as pd
csv_file_act = "../data/act.csv"
csv_file_sat = "../data/sat.csv"
act = pd.read_csv(csv_file_act)
sat = pd.read_csv(csv_file_sat)
```


```python
act.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>State</th>
      <th>Participation</th>
      <th>English</th>
      <th>Math</th>
      <th>Reading</th>
      <th>Science</th>
      <th>Composite</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>National</td>
      <td>60%</td>
      <td>20.3</td>
      <td>20.7</td>
      <td>21.4</td>
      <td>21.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Alabama</td>
      <td>100%</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Alaska</td>
      <td>65%</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Arizona</td>
      <td>62%</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Arkansas</td>
      <td>100%</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
sat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>State</th>
      <th>Participation</th>
      <th>Evidence-Based Reading and Writing</th>
      <th>Math</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Alabama</td>
      <td>5%</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Alaska</td>
      <td>38%</td>
      <td>547</td>
      <td>533</td>
      <td>1080</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Arizona</td>
      <td>30%</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Arkansas</td>
      <td>3%</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>California</td>
      <td>53%</td>
      <td>531</td>
      <td>524</td>
      <td>1055</td>
    </tr>
  </tbody>
</table>
</div>




```python
sat.dtypes
```




    Unnamed: 0                             int64
    State                                 object
    Participation                         object
    Evidence-Based Reading and Writing     int64
    Math                                   int64
    Total                                  int64
    dtype: object



The SAT dataset is all integers with the exception of Participation and State, which are strings. 


```python
act.dtypes
```




    Unnamed: 0         int64
    State             object
    Participation     object
    English          float64
    Math             float64
    Reading          float64
    Science          float64
    Composite        float64
    dtype: object



Data dictionary:

- State: state the data was taken from.

- Participation rate: number of students in that state participated in the SAT during 2017. The exact parameters for this are unclear.

- Composite/Total: average total score for either test, by state.

- Evidence-Based Reading and Writing/Math/English/Reading/Science: average scores for each section of the test, by state.

The data looks quite complete, as it should, considering this is taken from a site that took the data directly from the college board's annual report. Any possible issues might be that the participation column is not formatted as float64s because it contains percentage signs. The other possible issue is that some states require mandatory participation, which would bias participation results. Obviously there may be many other sources of bias. Additionally, the ACT data has a 'National' row and the SAT data does not, which will create a problem for merging, so that row must be dropped. Also, I will drop the extra index rows. Finally, two of Maryland's scores are incorrect. Maryland SAT math reads as 52, which is lower than what is mathematically possible (I did well on the SAT math). If we subtract Maryland's SAT EBRW score from the total score, we get the true math score, which I will insert. Maryland ACT sci is also impossibly low, so I set it to the missing value by computing what it should have been for the ACT composite score to be correct. 

Here I take off the percentage signs from the participation variables and replace them with nothing. Then I divide by 100 to get the ratio. Finally I assign the new columns to my new datasets, newsat and newact.


```python
newact=act
newact.Participation=newact.Participation.map(lambda x: float(x.replace('%',""))/100)

newsat=sat
newsat.Participation=newsat.Participation.map(lambda x: float(x.replace('%',""))/100)
```

I created two dictionaries to use as values to rename the columns. 


```python
column_map1={
    "State": "ST",
    "Participation": "SATPTN",
    "Evidence-Based Reading and Writing":"SATEBRW", 
    "Math": "SATMATH",
    "Total": "SATTOT"
}
column_map2={
    "State": "ST",
    "Participation": "ACTPTN",
    "Math": "ACTMATH",
    "English": "ACTENG",
    "Reading": "ACTREAD",
    "Science": "ACTSCI",
    "Composite": "ACTCOMP"
}
newsat=newsat.rename(columns=column_map1)
newact=newact.rename(columns=column_map2)
```

Then I dropped the unnamed rows and removed the national column. Then I reset the index so it starts at 0.


```python
newsat.drop('Unnamed: 0', axis=1, inplace=True)
newact.drop('Unnamed: 0', axis=1, inplace=True)
newact.drop(0, axis=0, inplace=True)
newact.reset_index(drop=True, inplace=True)
```

I fixed the errant Maryland scores by working backwards. I found SAT math by subtracting EBRW from the total. I calculated ACT sci by looking up how the composite score is calculated, which is by average. Thus, I set up a little algebraic expression to get my missing value.


```python
newsat.loc[20,'SATMATH']=524
newsat.loc[20,'SATMATH']
newact.loc[20, 'ACTSCI']=(newact.loc[20,'ACTCOMP']*4)-(newact.loc[20,'ACTENG']+newact.loc[20,'ACTREAD']+newact.loc[20,'ACTMATH'])
```

Merged the two dataframes on the state column. 


```python
satact = pd.merge(newsat, newact, on='ST')
satact.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>SATPTN</th>
      <th>SATEBRW</th>
      <th>SATMATH</th>
      <th>SATTOT</th>
      <th>ACTPTN</th>
      <th>ACTENG</th>
      <th>ACTMATH</th>
      <th>ACTREAD</th>
      <th>ACTSCI</th>
      <th>ACTCOMP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>0.05</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>0.38</td>
      <td>547</td>
      <td>533</td>
      <td>1080</td>
      <td>0.65</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>0.30</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
      <td>0.62</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>0.03</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>0.53</td>
      <td>531</td>
      <td>524</td>
      <td>1055</td>
      <td>0.31</td>
      <td>22.5</td>
      <td>22.7</td>
      <td>23.1</td>
      <td>22.2</td>
      <td>22.8</td>
    </tr>
  </tbody>
</table>
</div>



Here I was told to use create a column for standard deviation without using the one included in numpy. 


```python
import numpy as np
sd = [(np.sqrt(sum((x - np.mean(x)) ** 2)/50)) for x in np.array(satact.iloc[:,1:].T)]
sd.insert(0, 'SD')
sd
```




    ['SD',
     0.35276632270013036,
     45.66690138768932,
     47.12139516560329,
     92.49481172519046,
     0.32140842015886834,
     2.35367713980303,
     1.9819894936505533,
     2.0672706264873146,
     1.7533922304280611,
     2.020694891154341]



Adding the standard deviation column to the dataframe, though I had to drop it before I did the correlations.


```python
satact.loc[51]=sd
satact.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>SATPTN</th>
      <th>SATEBRW</th>
      <th>SATMATH</th>
      <th>SATTOT</th>
      <th>ACTPTN</th>
      <th>ACTENG</th>
      <th>ACTMATH</th>
      <th>ACTREAD</th>
      <th>ACTSCI</th>
      <th>ACTCOMP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>0.05</td>
      <td>593.0</td>
      <td>572.0</td>
      <td>1165.0</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>0.38</td>
      <td>547.0</td>
      <td>533.0</td>
      <td>1080.0</td>
      <td>0.65</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>0.30</td>
      <td>563.0</td>
      <td>553.0</td>
      <td>1116.0</td>
      <td>0.62</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>0.03</td>
      <td>614.0</td>
      <td>594.0</td>
      <td>1208.0</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>0.53</td>
      <td>531.0</td>
      <td>524.0</td>
      <td>1055.0</td>
      <td>0.31</td>
      <td>22.5</td>
      <td>22.7</td>
      <td>23.1</td>
      <td>22.2</td>
      <td>22.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
satact=satact.drop([51])
satact.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>SATPTN</th>
      <th>SATEBRW</th>
      <th>SATMATH</th>
      <th>SATTOT</th>
      <th>ACTPTN</th>
      <th>ACTENG</th>
      <th>ACTMATH</th>
      <th>ACTREAD</th>
      <th>ACTSCI</th>
      <th>ACTCOMP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>0.05</td>
      <td>593.0</td>
      <td>572.0</td>
      <td>1165.0</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>0.38</td>
      <td>547.0</td>
      <td>533.0</td>
      <td>1080.0</td>
      <td>0.65</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>0.30</td>
      <td>563.0</td>
      <td>553.0</td>
      <td>1116.0</td>
      <td>0.62</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>0.03</td>
      <td>614.0</td>
      <td>594.0</td>
      <td>1208.0</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>0.53</td>
      <td>531.0</td>
      <td>524.0</td>
      <td>1055.0</td>
      <td>0.31</td>
      <td>22.5</td>
      <td>22.7</td>
      <td>23.1</td>
      <td>22.2</td>
      <td>22.8</td>
    </tr>
  </tbody>
</table>
</div>



Some more examples of things I did for the assignment. 


```python
satact.sort_values(["SATMATH"]).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>SATPTN</th>
      <th>SATEBRW</th>
      <th>SATMATH</th>
      <th>SATTOT</th>
      <th>ACTPTN</th>
      <th>ACTENG</th>
      <th>ACTMATH</th>
      <th>ACTREAD</th>
      <th>ACTSCI</th>
      <th>ACTCOMP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>District of Columbia</td>
      <td>1.00</td>
      <td>482.0</td>
      <td>468.0</td>
      <td>950.0</td>
      <td>0.32</td>
      <td>24.4</td>
      <td>23.5</td>
      <td>24.9</td>
      <td>23.5</td>
      <td>24.2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Delaware</td>
      <td>1.00</td>
      <td>503.0</td>
      <td>492.0</td>
      <td>996.0</td>
      <td>0.18</td>
      <td>24.1</td>
      <td>23.4</td>
      <td>24.8</td>
      <td>23.6</td>
      <td>24.1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Idaho</td>
      <td>0.93</td>
      <td>513.0</td>
      <td>493.0</td>
      <td>1005.0</td>
      <td>0.38</td>
      <td>21.9</td>
      <td>21.8</td>
      <td>23.0</td>
      <td>22.1</td>
      <td>22.3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Michigan</td>
      <td>1.00</td>
      <td>509.0</td>
      <td>495.0</td>
      <td>1005.0</td>
      <td>0.29</td>
      <td>24.1</td>
      <td>23.7</td>
      <td>24.5</td>
      <td>23.8</td>
      <td>24.1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Florida</td>
      <td>0.83</td>
      <td>520.0</td>
      <td>497.0</td>
      <td>1017.0</td>
      <td>0.73</td>
      <td>19.0</td>
      <td>19.4</td>
      <td>21.0</td>
      <td>19.4</td>
      <td>19.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
satact[satact['SATMATH'] > 500].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>SATPTN</th>
      <th>SATEBRW</th>
      <th>SATMATH</th>
      <th>SATTOT</th>
      <th>ACTPTN</th>
      <th>ACTENG</th>
      <th>ACTMATH</th>
      <th>ACTREAD</th>
      <th>ACTSCI</th>
      <th>ACTCOMP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>0.05</td>
      <td>593.0</td>
      <td>572.0</td>
      <td>1165.0</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>0.38</td>
      <td>547.0</td>
      <td>533.0</td>
      <td>1080.0</td>
      <td>0.65</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>0.30</td>
      <td>563.0</td>
      <td>553.0</td>
      <td>1116.0</td>
      <td>0.62</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>0.03</td>
      <td>614.0</td>
      <td>594.0</td>
      <td>1208.0</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>0.53</td>
      <td>531.0</td>
      <td>524.0</td>
      <td>1055.0</td>
      <td>0.31</td>
      <td>22.5</td>
      <td>22.7</td>
      <td>23.1</td>
      <td>22.2</td>
      <td>22.8</td>
    </tr>
  </tbody>
</table>
</div>



Here I made histograms of participation. 


```python
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.hist(satact.SATPTN, bins=51, alpha=0.5, label='SAT ptn')
plt.title('SAT Participation Rate')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.hist(satact.ACTPTN, bins=51, alpha=0.5, label='ACT ptn', color = 'r')
plt.title('ACT Participation Rate')
plt.legend(loc='upper right')

plt.show()
```


![png](/images/final_for_blog_files/final_for_blog_30_0.png)


Histograms of the math sections. 


```python
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.hist(satact.SATMATH, bins=51, alpha=0.5, label='SAT math')
plt.title('SAT Math Avg Score')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.hist(satact.ACTMATH, bins=51, alpha=0.5, label='ACT math', color = 'r')
plt.title('ACT Math Avg Score')
plt.legend(loc='upper right')

plt.show()
```


![png](/images/final_for_blog_files/final_for_blog_32_0.png)


Histograms of the non math or science sections. 


```python
x = [satact.ACTENG]
y = [satact.ACTREAD]

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.hist(x, bins=51, alpha=0.5, label='ACT English')
plt.hist(y, bins=51, alpha=0.5, label='ACT reading')
plt.title('ACT English/Reading')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.hist(satact.SATEBRW, bins=51, alpha=0.5, label='SAT ebrw')
plt.title('SAT Evidence-Based Reading and Writing')
plt.legend(loc='upper right')

plt.show()
```


![png](/images/final_for_blog_files/final_for_blog_34_0.png)



```python
satact.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST</th>
      <th>SATPTN</th>
      <th>SATEBRW</th>
      <th>SATMATH</th>
      <th>SATTOT</th>
      <th>ACTPTN</th>
      <th>ACTENG</th>
      <th>ACTMATH</th>
      <th>ACTREAD</th>
      <th>ACTSCI</th>
      <th>ACTCOMP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>0.05</td>
      <td>593.0</td>
      <td>572.0</td>
      <td>1165.0</td>
      <td>1.0</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
  </tbody>
</table>
</div>



Looking at the correlations between variables. 


```python
import seaborn as sns
sns.set_palette("husl")
sns.set(style="ticks", color_codes=True)
sns.pairplot(satact)
```




    <seaborn.axisgrid.PairGrid at 0x113172da0>




![png](/images/final_for_blog_files/final_for_blog_37_1.png)



```python
plt.figure(figsize=(10,10))
sns.set_palette("husl")
sns.heatmap(satact.corr(), annot=True);
```


![png](/images/final_for_blog_files/final_for_blog_38_0.png)


This heatmap of our data is annoted with the strength of the correlations for all relationships. As described before, we see strong correlation between aptitude across individual tests, e.g., a strong ACT science score is associated with a strong ACT reading score.  

Obviously, there is a strong correlation between doing well in one part of one test and doing well in another part of the same test. What is also interesting to note is that there doesn't seem to be a correlation between sat scores and act scores, as one would expect one. However, the tests are quite different. Finally, there is a correlation, albeit a weaker one, between participation and score. This is a negative correlation, indicating that the higher level of participation a state has, the lower we would expect the average scores to be. Unfortunately, we would expect this to be a causal relationship from participation to score, not the other way around, and thus does not help us in our quest to increase participation. There also seems to be a negative correlation between ACT participation and SAT participation, again, which we would expect. However, this might help us in our plan to increase SAT participation. 

Now, some further visualizations. 


```python
ax = sns.boxplot(data=satact[['SATEBRW', 'SATMATH']], palette="Set1")
```


![png](/images/final_for_blog_files/final_for_blog_42_0.png)



```python
ax = sns.boxplot(data=satact[['ACTENG',
       'ACTMATH', 'ACTREAD', 'ACTSCI']], palette="Set2")
```


![png](/images/final_for_blog_files/final_for_blog_43_0.png)



```python
ax = sns.boxplot(data=satact[['SATPTN', 'ACTPTN']], palette="Set3")
```


![png](/images/final_for_blog_files/final_for_blog_44_0.png)



```python
plt.figure(figsize=(10,10));

plt.subplot(4, 3, 1);
plt.hist(satact.SATMATH, bins=51, alpha=0.5, label='SAT math');
plt.title('SAT Math Avg Score');
plt.legend(loc='upper right');

plt.subplot(4, 3, 2);
plt.hist(satact.SATEBRW, bins=51, alpha=0.5, label='SAT ebrw');
plt.title('SAT Evidence-Based Reading and Writing Avg Score');
plt.legend(loc='upper right');

plt.subplot(4, 3, 3);
plt.hist(satact.SATTOT, bins=51, alpha=0.5, label='SAT total');
plt.title('SAT Total Avg Score');
plt.legend(loc='upper right');

plt.subplot(4, 3, 4);
plt.hist(satact.SATPTN, bins=51, alpha=0.5, label='SAT ptn');
plt.title('SAT Participation Rate');
plt.legend(loc='upper right');

plt.subplot(4, 3, 5);
plt.hist(satact.ACTSCI, bins=51, alpha=0.5, label='ACT sci');
plt.title('ACT Science Avg Score');
plt.legend(loc='upper right');

plt.subplot(4, 3, 6);
plt.hist(satact.ACTMATH, bins=51, alpha=0.5, label='ACT math');
plt.title('ACT Math Avg Score');
plt.legend(loc='upper right');

plt.subplot(4, 3, 7);
plt.hist(satact.ACTENG, bins=51, alpha=0.5, label='ACT eng');
plt.title('ACT English Avg Score');
plt.legend(loc='upper right');

plt.subplot(4, 3, 8);
plt.hist(satact.ACTREAD, bins=51, alpha=0.5, label='ACT read');
plt.title('ACT Reading Avg Score');
plt.legend(loc='upper right');

plt.subplot(4, 3, 9);
plt.hist(satact.ACTCOMP, bins=51, alpha=0.5, label='ACT comp');
plt.title('ACT Composite Avg Score');
plt.legend(loc='upper right');

plt.subplot(4, 3, 10);
plt.hist(satact.ACTPTN, bins=51, alpha=0.5, label='ACT ptn');
plt.title('ACT Participation Rate');
plt.legend(loc='upper right');

plt.subplot(4, 3, 11);
plt.hist(x, bins=51, alpha=0.5, label='ACT English');
plt.hist(y, bins=51, alpha=0.5, label='ACT reading');
plt.title('ACT English/Reading');
plt.legend(loc='upper right');

plt.tight_layout();
plt.show();
```


![png](/images/final_for_blog_files/final_for_blog_45_0.png)



```python
import seaborn as sns
sns.regplot(x=satact['SATPTN'], y=satact['ACTPTN'], line_kws={"color":"r","alpha":0.7,"lw":5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a191e7358>




![png](/images/final_for_blog_files/final_for_blog_46_1.png)



```python
import seaborn as sns
sns.regplot(x=satact['SATTOT'], y=satact['ACTCOMP'], line_kws={"color":"r","alpha":0.7,"lw":5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a19883080>




![png](/images/final_for_blog_files/final_for_blog_47_1.png)


Here I did some hypothesis testing. 

H$_{0}$: The mean participation rates are the same.

H$_{A}$: The mean participation rates are not the same. 

$\alpha$: .05


```python
import scipy.stats as stats
result = stats.ttest_ind(satact.ACTPTN, satact.SATPTN)
result.statistic, result.pvalue, result
```




    (3.8085778908170544,
     0.00024134203698662353,
     Ttest_indResult(statistic=3.8085778908170544, pvalue=0.00024134203698662353))



Here, we see that when we compare the sample means of both our participation columns, we get a p of .0002, which is far less than our level of significance, which we set to be .05. Thus, there is only a .02% chance that our test result, 3.8, was due to randomness. Therefore, we have evidence that suggest our null hypothesis should be rejected, and so it seams that the participation rates are different. 

Generate and interpreting 95% confidence intervals for SAT and ACT participation rates.


```python
mean1, sigma1 = satact.ACTPTN.mean(), satact.ACTPTN.std()
mean2, sigma2 = satact.SATPTN.mean(), satact.SATPTN.std()

conf_int1 = stats.norm.interval(alpha=0.95, loc=mean1, scale=sigma1)
conf_int2 = stats.norm.interval(alpha=0.95, loc=mean2, scale=sigma2)
print('The 95% confidence interval for the ACT participation is {}, which means that, out of 100 trials, 95 of them will fall between the bounds of this interval.'.format(conf_int1))
print('The 95% confidence interval for the SAT participation is {}, which means that, out of 100 trials, 95 of them will fall between the bounds of this interval. Note that the lower bound of this interval is less than zero. Thus, we can assume that the true interval is (0.)'.format(conf_int2))
```

    The 95% confidence interval for the ACT participation is (0.02260009176854383, 1.2824979474471425), which means that, out of 100 trials, 95 of them will fall between the bounds of this interval.
    The 95% confidence interval for the SAT participation is (-0.29337007176461544, 1.0894485031371646), which means that, out of 100 trials, 95 of them will fall between the bounds of this interval. Note that the lower bound of this interval is less than zero. Thus, we can assume that the true interval is (0.)


That's it for the notebook. If you haven't, please look at the accompanying presentation to see my hypothetical recommendations to The College Board and check out the GitHub [repository](https://github.com/esjacobs/SAT_and_ACT).
. 

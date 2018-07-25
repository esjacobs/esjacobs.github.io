
# Ames Housing Data and House Price Prediction

Hi and welcome to my blog. This is my first real entry, and it basically entails a project I did in my data science bootcamp at General Assembly. This particular post will be heavy in the coding department and light in the writing department, but will give you a little glimpse into how I was thinking early on in the program. So, without further ado…

First we start off by importing anything and everything that might be helpful here.


```python
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
import warnings
warnings.simplefilter("ignore")

sns.set_style('darkgrid')

%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```

Next, we import out data files, first saving the names as variables.


```python
train_csv = '/Users/evanjacobs/dsi/DSI-US-4/project-2/train.csv'
test_csv = '/Users/evanjacobs/dsi/DSI-US-4/project-2/test.csv'
```

For now, we'll just import out training data, so we don't accidentally alter the precious testing data. 


```python
df = pd.read_csv(train_csv)
finaltest = pd.read_csv(test_csv)
```

First, we're going to do our test train split, and here we'll set our y to be our target, 'SalePrice'.


```python
X = df.drop(['SalePrice'], axis=1)
y = df.SalePrice.values
X_full = df.drop(['SalePrice'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

Let's have a look, shall we?


```python
X_train.head()
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
      <th>Id</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>484</th>
      <td>1875</td>
      <td>534201040</td>
      <td>20</td>
      <td>RL</td>
      <td>70.0</td>
      <td>8050</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2007</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>178</td>
      <td>902206040</td>
      <td>50</td>
      <td>RM</td>
      <td>50.0</td>
      <td>5500</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>20</td>
      <td>527302110</td>
      <td>20</td>
      <td>RL</td>
      <td>85.0</td>
      <td>13175</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2010</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>640</th>
      <td>2420</td>
      <td>528228280</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>3087</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
      <td>New</td>
    </tr>
    <tr>
      <th>811</th>
      <td>1448</td>
      <td>907202160</td>
      <td>80</td>
      <td>RL</td>
      <td>NaN</td>
      <td>10970</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Low</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>10</td>
      <td>2008</td>
      <td>WD</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
X_train.describe()
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
      <th>Id</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Mas Vnr Area</th>
      <th>...</th>
      <th>Garage Area</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1538.000000</td>
      <td>1.538000e+03</td>
      <td>1538.000000</td>
      <td>1294.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1521.000000</td>
      <td>...</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1469.118336</td>
      <td>7.148299e+08</td>
      <td>57.542263</td>
      <td>69.540958</td>
      <td>10179.084525</td>
      <td>6.109883</td>
      <td>5.571521</td>
      <td>1971.674252</td>
      <td>1984.081274</td>
      <td>99.113083</td>
      <td>...</td>
      <td>471.424577</td>
      <td>95.207412</td>
      <td>49.256177</td>
      <td>23.018856</td>
      <td>2.914174</td>
      <td>17.200260</td>
      <td>3.197659</td>
      <td>55.282835</td>
      <td>6.195709</td>
      <td>2007.784785</td>
    </tr>
    <tr>
      <th>std</th>
      <td>844.226713</td>
      <td>1.887552e+08</td>
      <td>43.351837</td>
      <td>22.987056</td>
      <td>7353.026485</td>
      <td>1.405082</td>
      <td>1.110848</td>
      <td>30.258868</td>
      <td>21.200024</td>
      <td>174.156041</td>
      <td>...</td>
      <td>216.396308</td>
      <td>132.411630</td>
      <td>69.244398</td>
      <td>60.037423</td>
      <td>27.776465</td>
      <td>59.571394</td>
      <td>43.605315</td>
      <td>617.362905</td>
      <td>2.753136</td>
      <td>1.313997</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>5.263011e+08</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1879.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>746.500000</td>
      <td>5.284567e+08</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7455.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1953.000000</td>
      <td>1964.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>316.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>2007.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1496.500000</td>
      <td>5.354546e+08</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9465.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1975.000000</td>
      <td>1993.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>480.000000</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2174.750000</td>
      <td>9.071855e+08</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11635.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2001.000000</td>
      <td>2004.000000</td>
      <td>162.000000</td>
      <td>...</td>
      <td>576.000000</td>
      <td>168.000000</td>
      <td>72.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2930.000000</td>
      <td>9.241520e+08</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>159000.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>...</td>
      <td>1418.000000</td>
      <td>1424.000000</td>
      <td>547.000000</td>
      <td>432.000000</td>
      <td>508.000000</td>
      <td>490.000000</td>
      <td>800.000000</td>
      <td>17000.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>




```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1538 entries, 484 to 1169
    Data columns (total 80 columns):
    Id                 1538 non-null int64
    PID                1538 non-null int64
    MS SubClass        1538 non-null int64
    MS Zoning          1538 non-null object
    Lot Frontage       1294 non-null float64
    Lot Area           1538 non-null int64
    Street             1538 non-null object
    Alley              110 non-null object
    Lot Shape          1538 non-null object
    Land Contour       1538 non-null object
    Utilities          1538 non-null object
    Lot Config         1538 non-null object
    Land Slope         1538 non-null object
    Neighborhood       1538 non-null object
    Condition 1        1538 non-null object
    Condition 2        1538 non-null object
    Bldg Type          1538 non-null object
    House Style        1538 non-null object
    Overall Qual       1538 non-null int64
    Overall Cond       1538 non-null int64
    Year Built         1538 non-null int64
    Year Remod/Add     1538 non-null int64
    Roof Style         1538 non-null object
    Roof Matl          1538 non-null object
    Exterior 1st       1538 non-null object
    Exterior 2nd       1538 non-null object
    Mas Vnr Type       1521 non-null object
    Mas Vnr Area       1521 non-null float64
    Exter Qual         1538 non-null object
    Exter Cond         1538 non-null object
    Foundation         1538 non-null object
    Bsmt Qual          1501 non-null object
    Bsmt Cond          1501 non-null object
    Bsmt Exposure      1498 non-null object
    BsmtFin Type 1     1501 non-null object
    BsmtFin SF 1       1538 non-null float64
    BsmtFin Type 2     1500 non-null object
    BsmtFin SF 2       1538 non-null float64
    Bsmt Unf SF        1538 non-null float64
    Total Bsmt SF      1538 non-null float64
    Heating            1538 non-null object
    Heating QC         1538 non-null object
    Central Air        1538 non-null object
    Electrical         1538 non-null object
    1st Flr SF         1538 non-null int64
    2nd Flr SF         1538 non-null int64
    Low Qual Fin SF    1538 non-null int64
    Gr Liv Area        1538 non-null int64
    Bsmt Full Bath     1537 non-null float64
    Bsmt Half Bath     1537 non-null float64
    Full Bath          1538 non-null int64
    Half Bath          1538 non-null int64
    Bedroom AbvGr      1538 non-null int64
    Kitchen AbvGr      1538 non-null int64
    Kitchen Qual       1538 non-null object
    TotRms AbvGrd      1538 non-null int64
    Functional         1538 non-null object
    Fireplaces         1538 non-null int64
    Fireplace Qu       798 non-null object
    Garage Type        1449 non-null object
    Garage Yr Blt      1449 non-null float64
    Garage Finish      1449 non-null object
    Garage Cars        1538 non-null float64
    Garage Area        1538 non-null float64
    Garage Qual        1449 non-null object
    Garage Cond        1449 non-null object
    Paved Drive        1538 non-null object
    Wood Deck SF       1538 non-null int64
    Open Porch SF      1538 non-null int64
    Enclosed Porch     1538 non-null int64
    3Ssn Porch         1538 non-null int64
    Screen Porch       1538 non-null int64
    Pool Area          1538 non-null int64
    Pool QC            9 non-null object
    Fence              286 non-null object
    Misc Feature       50 non-null object
    Misc Val           1538 non-null int64
    Mo Sold            1538 non-null int64
    Yr Sold            1538 non-null int64
    Sale Type          1538 non-null object
    dtypes: float64(11), int64(27), object(42)
    memory usage: 973.3+ KB



```python
X_train.columns
```




    Index(['Id', 'PID', 'MS SubClass', 'MS Zoning', 'Lot Frontage', 'Lot Area',
           'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities',
           'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1',
           'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual',
           'Overall Cond', 'Year Built', 'Year Remod/Add', 'Roof Style',
           'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
           'Mas Vnr Area', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual',
           'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1',
           'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
           'Heating', 'Heating QC', 'Central Air', 'Electrical', '1st Flr SF',
           '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath',
           'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr',
           'Kitchen AbvGr', 'Kitchen Qual', 'TotRms AbvGrd', 'Functional',
           'Fireplaces', 'Fireplace Qu', 'Garage Type', 'Garage Yr Blt',
           'Garage Finish', 'Garage Cars', 'Garage Area', 'Garage Qual',
           'Garage Cond', 'Paved Drive', 'Wood Deck SF', 'Open Porch SF',
           'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Pool QC',
           'Fence', 'Misc Feature', 'Misc Val', 'Mo Sold', 'Yr Sold', 'Sale Type'],
          dtype='object')



Closing up the column names so I can use dot notation.


```python
def colclean(column_list): 
    columns=[]
    for n in column_list:
        n = n.lower().replace(' ','')
        columns.append(n)
    return columns
colclean(df.columns)
X_train.columns = colclean(X_train.columns)
X_test.columns = colclean(X_test.columns)
X_train.columns
X_full.columns = colclean(X_full.columns)
finaltest.columns = colclean(finaltest.columns)
```

Checking for duplicate PIDs. 


```python
X_train.duplicated(subset='pid', keep='first').sum()
```




    0



Got any nulls lying around?


```python
X_train.isnull().sum()
```




    id                  0
    pid                 0
    mssubclass          0
    mszoning            0
    lotfrontage       244
    lotarea             0
    street              0
    alley            1428
    lotshape            0
    landcontour         0
    utilities           0
    lotconfig           0
    landslope           0
    neighborhood        0
    condition1          0
    condition2          0
    bldgtype            0
    housestyle          0
    overallqual         0
    overallcond         0
    yearbuilt           0
    yearremod/add       0
    roofstyle           0
    roofmatl            0
    exterior1st         0
    exterior2nd         0
    masvnrtype         17
    masvnrarea         17
    exterqual           0
    extercond           0
                     ... 
    fullbath            0
    halfbath            0
    bedroomabvgr        0
    kitchenabvgr        0
    kitchenqual         0
    totrmsabvgrd        0
    functional          0
    fireplaces          0
    fireplacequ       740
    garagetype         89
    garageyrblt        89
    garagefinish       89
    garagecars          0
    garagearea          0
    garagequal         89
    garagecond         89
    paveddrive          0
    wooddecksf          0
    openporchsf         0
    enclosedporch       0
    3ssnporch           0
    screenporch         0
    poolarea            0
    poolqc           1529
    fence            1252
    miscfeature      1488
    miscval             0
    mosold              0
    yrsold              0
    saletype            0
    Length: 80, dtype: int64



Here's a trick I learned.


```python
X_train.isna().sum()[X_train.isna().sum() !=0]
```




    lotfrontage      244
    alley           1428
    masvnrtype        17
    masvnrarea        17
    bsmtqual          37
    bsmtcond          37
    bsmtexposure      40
    bsmtfintype1      37
    bsmtfintype2      38
    bsmtfullbath       1
    bsmthalfbath       1
    fireplacequ      740
    garagetype        89
    garageyrblt       89
    garagefinish      89
    garagequal        89
    garagecond        89
    poolqc          1529
    fence           1252
    miscfeature     1488
    dtype: int64



Having a look at what the object column with null values looks like. 


```python
X_train.poolqc.unique()
```




    array([nan, 'TA', 'Gd', 'Ex', 'Fa'], dtype=object)



Just for the sake of time, going to fill all null values with their numerical averages for numerical columns.


```python
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
X_full = X_full.fillna(X_full.mean())
finaltest = finaltest.fillna(finaltest.mean())
```


```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1538 entries, 484 to 1169
    Data columns (total 80 columns):
    id               1538 non-null int64
    pid              1538 non-null int64
    mssubclass       1538 non-null int64
    mszoning         1538 non-null object
    lotfrontage      1538 non-null float64
    lotarea          1538 non-null int64
    street           1538 non-null object
    alley            110 non-null object
    lotshape         1538 non-null object
    landcontour      1538 non-null object
    utilities        1538 non-null object
    lotconfig        1538 non-null object
    landslope        1538 non-null object
    neighborhood     1538 non-null object
    condition1       1538 non-null object
    condition2       1538 non-null object
    bldgtype         1538 non-null object
    housestyle       1538 non-null object
    overallqual      1538 non-null int64
    overallcond      1538 non-null int64
    yearbuilt        1538 non-null int64
    yearremod/add    1538 non-null int64
    roofstyle        1538 non-null object
    roofmatl         1538 non-null object
    exterior1st      1538 non-null object
    exterior2nd      1538 non-null object
    masvnrtype       1521 non-null object
    masvnrarea       1538 non-null float64
    exterqual        1538 non-null object
    extercond        1538 non-null object
    foundation       1538 non-null object
    bsmtqual         1501 non-null object
    bsmtcond         1501 non-null object
    bsmtexposure     1498 non-null object
    bsmtfintype1     1501 non-null object
    bsmtfinsf1       1538 non-null float64
    bsmtfintype2     1500 non-null object
    bsmtfinsf2       1538 non-null float64
    bsmtunfsf        1538 non-null float64
    totalbsmtsf      1538 non-null float64
    heating          1538 non-null object
    heatingqc        1538 non-null object
    centralair       1538 non-null object
    electrical       1538 non-null object
    1stflrsf         1538 non-null int64
    2ndflrsf         1538 non-null int64
    lowqualfinsf     1538 non-null int64
    grlivarea        1538 non-null int64
    bsmtfullbath     1538 non-null float64
    bsmthalfbath     1538 non-null float64
    fullbath         1538 non-null int64
    halfbath         1538 non-null int64
    bedroomabvgr     1538 non-null int64
    kitchenabvgr     1538 non-null int64
    kitchenqual      1538 non-null object
    totrmsabvgrd     1538 non-null int64
    functional       1538 non-null object
    fireplaces       1538 non-null int64
    fireplacequ      798 non-null object
    garagetype       1449 non-null object
    garageyrblt      1538 non-null float64
    garagefinish     1449 non-null object
    garagecars       1538 non-null float64
    garagearea       1538 non-null float64
    garagequal       1449 non-null object
    garagecond       1449 non-null object
    paveddrive       1538 non-null object
    wooddecksf       1538 non-null int64
    openporchsf      1538 non-null int64
    enclosedporch    1538 non-null int64
    3ssnporch        1538 non-null int64
    screenporch      1538 non-null int64
    poolarea         1538 non-null int64
    poolqc           9 non-null object
    fence            286 non-null object
    miscfeature      50 non-null object
    miscval          1538 non-null int64
    mosold           1538 non-null int64
    yrsold           1538 non-null int64
    saletype         1538 non-null object
    dtypes: float64(11), int64(27), object(42)
    memory usage: 973.3+ KB


Breaking dataframes into two where one is all num values and one is all obj values so I can look at them more easily.


```python
X_tr_obj = X_train.select_dtypes(exclude=[np.number])
X_tr_num = X_train.select_dtypes(include=[np.number])
X_ts_obj = X_test.select_dtypes(exclude=[np.number])
X_ts_num = X_test.select_dtypes(include=[np.number])
X_full_obj = X_full.select_dtypes(exclude=[np.number])
X_full_num = X_full.select_dtypes(include=[np.number])
finaltest_obj = finaltest.select_dtypes(exclude=[np.number])
finaltest_num = finaltest.select_dtypes(include=[np.number])
```


```python
X_ts_num.head()
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
      <th>id</th>
      <th>pid</th>
      <th>mssubclass</th>
      <th>lotfrontage</th>
      <th>lotarea</th>
      <th>overallqual</th>
      <th>overallcond</th>
      <th>yearbuilt</th>
      <th>yearremod/add</th>
      <th>masvnrarea</th>
      <th>...</th>
      <th>wooddecksf</th>
      <th>openporchsf</th>
      <th>enclosedporch</th>
      <th>3ssnporch</th>
      <th>screenporch</th>
      <th>poolarea</th>
      <th>poolqc</th>
      <th>miscval</th>
      <th>mosold</th>
      <th>yrsold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>908</th>
      <td>2559</td>
      <td>534455080</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>5</td>
      <td>6</td>
      <td>1961</td>
      <td>1990</td>
      <td>0.0</td>
      <td>...</td>
      <td>144</td>
      <td>0</td>
      <td>205</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1619</th>
      <td>1947</td>
      <td>535375130</td>
      <td>50</td>
      <td>60.0</td>
      <td>10134</td>
      <td>5</td>
      <td>6</td>
      <td>1940</td>
      <td>1950</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>391</th>
      <td>81</td>
      <td>531453010</td>
      <td>20</td>
      <td>81.0</td>
      <td>9672</td>
      <td>6</td>
      <td>5</td>
      <td>1984</td>
      <td>1985</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>861</th>
      <td>2573</td>
      <td>535151130</td>
      <td>90</td>
      <td>70.0</td>
      <td>7728</td>
      <td>5</td>
      <td>6</td>
      <td>1962</td>
      <td>1962</td>
      <td>120.0</td>
      <td>...</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1270</th>
      <td>1569</td>
      <td>914476080</td>
      <td>90</td>
      <td>76.0</td>
      <td>10260</td>
      <td>5</td>
      <td>4</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>



Checking to make sure it worked. 


```python
print(X_train.shape)
print(X_tr_obj.shape)
print(X_tr_num.shape)
```

    (1538, 80)
    (1538, 42)
    (1538, 38)


Let's check for potential outliers.


```python
X_tr_num.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>1538.0</td>
      <td>1.469118e+03</td>
      <td>8.442267e+02</td>
      <td>1.0</td>
      <td>7.465000e+02</td>
      <td>1.496500e+03</td>
      <td>2.174750e+03</td>
      <td>2930.0</td>
    </tr>
    <tr>
      <th>pid</th>
      <td>1538.0</td>
      <td>7.148299e+08</td>
      <td>1.887552e+08</td>
      <td>526301100.0</td>
      <td>5.284567e+08</td>
      <td>5.354546e+08</td>
      <td>9.071855e+08</td>
      <td>924152030.0</td>
    </tr>
    <tr>
      <th>mssubclass</th>
      <td>1538.0</td>
      <td>5.754226e+01</td>
      <td>4.335184e+01</td>
      <td>20.0</td>
      <td>2.000000e+01</td>
      <td>5.000000e+01</td>
      <td>7.000000e+01</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>lotfrontage</th>
      <td>1538.0</td>
      <td>6.954096e+01</td>
      <td>2.108364e+01</td>
      <td>21.0</td>
      <td>6.000000e+01</td>
      <td>6.954096e+01</td>
      <td>7.900000e+01</td>
      <td>313.0</td>
    </tr>
    <tr>
      <th>lotarea</th>
      <td>1538.0</td>
      <td>1.017908e+04</td>
      <td>7.353026e+03</td>
      <td>1300.0</td>
      <td>7.455500e+03</td>
      <td>9.465000e+03</td>
      <td>1.163550e+04</td>
      <td>159000.0</td>
    </tr>
    <tr>
      <th>overallqual</th>
      <td>1538.0</td>
      <td>6.109883e+00</td>
      <td>1.405082e+00</td>
      <td>1.0</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>7.000000e+00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>overallcond</th>
      <td>1538.0</td>
      <td>5.571521e+00</td>
      <td>1.110848e+00</td>
      <td>1.0</td>
      <td>5.000000e+00</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>yearbuilt</th>
      <td>1538.0</td>
      <td>1.971674e+03</td>
      <td>3.025887e+01</td>
      <td>1879.0</td>
      <td>1.953000e+03</td>
      <td>1.975000e+03</td>
      <td>2.001000e+03</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>yearremod/add</th>
      <td>1538.0</td>
      <td>1.984081e+03</td>
      <td>2.120002e+01</td>
      <td>1950.0</td>
      <td>1.964000e+03</td>
      <td>1.993000e+03</td>
      <td>2.004000e+03</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>masvnrarea</th>
      <td>1538.0</td>
      <td>9.911308e+01</td>
      <td>1.731902e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.600000e+02</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>bsmtfinsf1</th>
      <td>1538.0</td>
      <td>4.402523e+02</td>
      <td>4.704420e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>3.610000e+02</td>
      <td>7.287500e+02</td>
      <td>5644.0</td>
    </tr>
    <tr>
      <th>bsmtfinsf2</th>
      <td>1538.0</td>
      <td>4.791873e+01</td>
      <td>1.636097e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1474.0</td>
    </tr>
    <tr>
      <th>bsmtunfsf</th>
      <td>1538.0</td>
      <td>5.709402e+02</td>
      <td>4.445010e+02</td>
      <td>0.0</td>
      <td>2.222500e+02</td>
      <td>4.800000e+02</td>
      <td>8.150000e+02</td>
      <td>2336.0</td>
    </tr>
    <tr>
      <th>totalbsmtsf</th>
      <td>1538.0</td>
      <td>1.059111e+03</td>
      <td>4.524990e+02</td>
      <td>0.0</td>
      <td>7.895000e+02</td>
      <td>9.945000e+02</td>
      <td>1.324000e+03</td>
      <td>6110.0</td>
    </tr>
    <tr>
      <th>1stflrsf</th>
      <td>1538.0</td>
      <td>1.165298e+03</td>
      <td>4.039276e+02</td>
      <td>334.0</td>
      <td>8.782500e+02</td>
      <td>1.092500e+03</td>
      <td>1.408500e+03</td>
      <td>5095.0</td>
    </tr>
    <tr>
      <th>2ndflrsf</th>
      <td>1538.0</td>
      <td>3.328362e+02</td>
      <td>4.238517e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>6.995000e+02</td>
      <td>1836.0</td>
    </tr>
    <tr>
      <th>lowqualfinsf</th>
      <td>1538.0</td>
      <td>5.667100e+00</td>
      <td>5.337538e+01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1064.0</td>
    </tr>
    <tr>
      <th>grlivarea</th>
      <td>1538.0</td>
      <td>1.503801e+03</td>
      <td>5.047836e+02</td>
      <td>334.0</td>
      <td>1.143000e+03</td>
      <td>1.452000e+03</td>
      <td>1.724000e+03</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <th>bsmtfullbath</th>
      <td>1538.0</td>
      <td>4.307092e-01</td>
      <td>5.182866e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>bsmthalfbath</th>
      <td>1538.0</td>
      <td>6.115810e-02</td>
      <td>2.476318e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>fullbath</th>
      <td>1538.0</td>
      <td>1.583225e+00</td>
      <td>5.445938e-01</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>halfbath</th>
      <td>1538.0</td>
      <td>3.719116e-01</td>
      <td>4.993598e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>bedroomabvgr</th>
      <td>1538.0</td>
      <td>2.843953e+00</td>
      <td>8.124554e-01</td>
      <td>0.0</td>
      <td>2.000000e+00</td>
      <td>3.000000e+00</td>
      <td>3.000000e+00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>kitchenabvgr</th>
      <td>1538.0</td>
      <td>1.041612e+00</td>
      <td>2.093097e-01</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>totrmsabvgrd</th>
      <td>1538.0</td>
      <td>6.445384e+00</td>
      <td>1.545643e+00</td>
      <td>2.0</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>7.000000e+00</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>fireplaces</th>
      <td>1538.0</td>
      <td>6.046814e-01</td>
      <td>6.481274e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>garageyrblt</th>
      <td>1538.0</td>
      <td>1.978795e+03</td>
      <td>2.501178e+01</td>
      <td>1895.0</td>
      <td>1.962000e+03</td>
      <td>1.978795e+03</td>
      <td>2.001000e+03</td>
      <td>2207.0</td>
    </tr>
    <tr>
      <th>garagecars</th>
      <td>1538.0</td>
      <td>1.774382e+00</td>
      <td>7.672165e-01</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>garagearea</th>
      <td>1538.0</td>
      <td>4.714246e+02</td>
      <td>2.163963e+02</td>
      <td>0.0</td>
      <td>3.162500e+02</td>
      <td>4.800000e+02</td>
      <td>5.760000e+02</td>
      <td>1418.0</td>
    </tr>
    <tr>
      <th>wooddecksf</th>
      <td>1538.0</td>
      <td>9.520741e+01</td>
      <td>1.324116e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.680000e+02</td>
      <td>1424.0</td>
    </tr>
    <tr>
      <th>openporchsf</th>
      <td>1538.0</td>
      <td>4.925618e+01</td>
      <td>6.924440e+01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>2.800000e+01</td>
      <td>7.200000e+01</td>
      <td>547.0</td>
    </tr>
    <tr>
      <th>enclosedporch</th>
      <td>1538.0</td>
      <td>2.301886e+01</td>
      <td>6.003742e+01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>432.0</td>
    </tr>
    <tr>
      <th>3ssnporch</th>
      <td>1538.0</td>
      <td>2.914174e+00</td>
      <td>2.777646e+01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>508.0</td>
    </tr>
    <tr>
      <th>screenporch</th>
      <td>1538.0</td>
      <td>1.720026e+01</td>
      <td>5.957139e+01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>490.0</td>
    </tr>
    <tr>
      <th>poolarea</th>
      <td>1538.0</td>
      <td>3.197659e+00</td>
      <td>4.360531e+01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>800.0</td>
    </tr>
    <tr>
      <th>miscval</th>
      <td>1538.0</td>
      <td>5.528283e+01</td>
      <td>6.173629e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>17000.0</td>
    </tr>
    <tr>
      <th>mosold</th>
      <td>1538.0</td>
      <td>6.195709e+00</td>
      <td>2.753136e+00</td>
      <td>1.0</td>
      <td>4.000000e+00</td>
      <td>6.000000e+00</td>
      <td>8.000000e+00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>yrsold</th>
      <td>1538.0</td>
      <td>2.007785e+03</td>
      <td>1.313997e+00</td>
      <td>2006.0</td>
      <td>2.007000e+03</td>
      <td>2.008000e+03</td>
      <td>2.009000e+03</td>
      <td>2010.0</td>
    </tr>
  </tbody>
</table>
</div>



Obviously, some of the maxes are large, but that's the way real estate works. Otherwise, nothing here pops out as being wroong. 

Okay, now that we have separated our dataframe into a numerical one and a categorical one, let's take a look-see at the numerical correlations.


```python
X_tr_num['sp']=y_train
```


```python
abs(X_tr_num.corr().sp)
```




    id               0.046963
    pid              0.243312
    mssubclass       0.104183
    lotfrontage      0.320966
    lotarea          0.301233
    overallqual      0.787963
    overallcond      0.094205
    yearbuilt        0.560274
    yearremod/add    0.531071
    masvnrarea       0.503817
    bsmtfinsf1       0.423380
    bsmtfinsf2       0.010237
    bsmtunfsf        0.180959
    totalbsmtsf      0.621630
    1stflrsf         0.616752
    2ndflrsf         0.244477
    lowqualfinsf     0.031802
    grlivarea        0.695442
    bsmtfullbath     0.279659
    bsmthalfbath     0.036105
    fullbath         0.541253
    halfbath         0.296527
    bedroomabvgr     0.150723
    kitchenabvgr     0.131280
    totrmsabvgrd     0.495609
    fireplaces       0.467371
    garageyrblt      0.499246
    garagecars       0.646358
    garagearea       0.654564
    wooddecksf       0.322195
    openporchsf      0.326194
    enclosedporch    0.113088
    3ssnporch        0.062105
    screenporch      0.128890
    poolarea         0.026498
    miscval          0.013014
    mosold           0.026836
    yrsold           0.014487
    sp               1.000000
    Name: sp, dtype: float64



For this first model, we're going to choose all the columns where the correlation coefficient with SalePrice is greater than or equal to some value I determine.


```python
vals = abs(X_tr_num.corr().sp).drop('sp').sort_values(ascending=False)
corr_cols = list(vals[vals >= 0.3].index)

X_tr_mod1 = X_tr_num[corr_cols]
X_ts_mod1 = X_ts_num[corr_cols]
X_full_mod1 = X_full_num[corr_cols]
finaltest_num = finaltest_num[corr_cols]

corr_cols
```




    ['overallqual',
     'grlivarea',
     'garagearea',
     'garagecars',
     'totalbsmtsf',
     '1stflrsf',
     'yearbuilt',
     'fullbath',
     'yearremod/add',
     'masvnrarea',
     'garageyrblt',
     'totrmsabvgrd',
     'fireplaces',
     'bsmtfinsf1',
     'openporchsf',
     'wooddecksf',
     'lotfrontage',
     'lotarea']



First, let's just notice that garageyrblt, yearbuilt should be correlated, as well as garagearea, garagecars, as well as totalbsmtsf, masvnrarea, grlivarea, 1stflrsf. So let's make some interaction variables. 


```python
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, interaction_only=False, 
                         include_bias=True)
pf.fit(X_tr_mod1)
X_tr_mod1 = pf.transform(X_tr_mod1)
X_ts_mod1 = pf.transform(X_ts_mod1)
X_full_mod1 = pf.transform(X_full_mod1)
finaltest_num = pf.transform(finaltest_num)
```

Okay, now let's use a standard scalar to make everything line up nicely. 


```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_tr_mod1 = ss.fit_transform(X_tr_mod1)
X_ts_mod1 = ss.transform(X_ts_mod1)
X_full_mod1 = ss.fit_transform(X_full_mod1)
finaltest_num = ss.transform(finaltest_num)
```

So, for lasso, ridge, and enet, I played with different alpha ranges, different numbers of iterations, and also different correlation thresholds. 

Let's try a lasso!


```python
l_alphas = np.arange(.001, .15, .0025)
lasso_model = LassoCV(alphas=l_alphas, max_iter=2000, cv=5)
# lasso_model = LassoCV(max_iter=10000, cv=5)

model_1 = lasso_model.fit(X_tr_mod1, y_train)
```

Great, let's score the lasso.


```python
print(model_1.score(X_ts_mod1, y_test))
```

    0.8775294669615423


Hey, that's not a bad score at all! What if we tried ridge? 


```python
ridge_alphas = np.logspace(0, 5, 200)

ridge_model = RidgeCV(alphas=ridge_alphas, cv=10)
# ridge_model = RidgeCV(cv=10)
ridge_model.fit(X_tr_mod1, y_train)
```




    RidgeCV(alphas=array([1.00000e+00, 1.05956e+00, ..., 9.43788e+04, 1.00000e+05]),
        cv=10, fit_intercept=True, gcv_mode=None, normalize=False,
        scoring=None, store_cv_values=False)




```python
ridge = Ridge(alpha=ridge_model.alpha_)

ridge_scores = cross_val_score(ridge, X_ts_mod1, y_test, cv=15)

print(ridge_scores)
print(np.mean(ridge_scores))
```

Hmmm. And last, everyone's favorite, the elastic net. 


```python
l1_ratios = np.linspace(0.01, 1.0, 25)

enet = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=100, cv=10,
                            verbose=0)
# enet = ElasticNetCV(cv=10, verbose=0)
enet.fit(X_tr_mod1, y_train)

print(enet.alpha_)
print(enet.l1_ratio_)

```


```python
enet = ElasticNet(alpha=enet.alpha_, l1_ratio=enet.l1_ratio_)

enet_scores = cross_val_score(enet, X_ts_mod1, y_test, cv=10)

print(enet_scores)
print(np.mean(enet_scores))

```

It's basically the same. But I have discovered that as I decreased my cut off for correlation, my lasso score remained largely the same, but my ridge and enet scores went up a tiny bit, culminating with my pulling an R-squared on .89 from Elastic Net. 


```python
# l_alphas = np.arange(.001, .15, .0025)
# lasso_model_final = LassoCV(alphas=l_alphas, cv=5)
# model_1_final = lasso_model.fit(X_full_mod1, y)

enet = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=100, cv=10,
                            verbose=0)
model_1_final = enet.fit(X_full_mod1, y)

```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


This is how I send my model's predictions to a file.


```python
evansubmission1 = pd.DataFrame(data = model_1_final.predict(finaltest_num), columns = ['SalePrice'], index=finaltest['id'])
evansubmission1.to_csv('./evansubmission1.csv')
```

Plotting my test predictions vs. my test y for a nice visualization of the efficacy of my model. 


```python
predictions = model_1_final.predict(X_ts_mod1)
y = y_test

# Plot the model
plt.figure(figsize=(8,8))
plt.scatter(predictions, y, s=30, c='g', marker='*', zorder=10)
plt.xlabel("Predicted Values of Price From My Horrible Model")
plt.ylabel("Actual Values of Price")

plt.plot([0, np.max(y)], [0, np.max(y)], c = 'k')

plt.show()
score = cross_val_score(model_1_final, X_ts_mod1, y_test, cv=10)
print("score: ", score.mean())
```


![png](/images/EvanProject2Model1of3toblog_files/EvanProject2Model1of3toblog_60_0.png)


    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    score:  0.8773479828283166


And that's it for the first model! Don't forget to look at the other two model files by checking out the GitHub [repository](https://github.com/esjacobs/Predicting-Ames-Housing-Prices). I've included the data dictionary below. 


```python
# evansubmission1 = pd.DataFrame(data = model_1.predict(X_ts_mod1), columns = ['SalePrice'], index=y_test['Id'])
# evansubmission1.to_csv('./evansubmission1.csv')
```


```python
# There are three files:

# train.csv -- this data contains all of the training data for your model.
# The target variable (SalePrice) is removed from the test set!
# test.csv -- this data contains the test data for your model. You will feed this data into your regression model to make predictions.
# sample_sub_reg.csv -- An example of a correctly formatted submission for this challenge (with a random number provided as predictions for SalePrice. Please ensure that your submission to Kaggle matches this format.
# Codebook / Data Dictionary:

# SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict for this challenge.
# MSSubClass: The building class
#     20 1-STORY 1946 & NEWER ALL STYLES
#     30 1-STORY 1945 & OLDER
#     40 1-STORY W/FINISHED ATTIC ALL AGES
#     45 1-1/2 STORY - UNFINISHED ALL AGES
#     50 1-1/2 STORY FINISHED ALL AGES
#     60 2-STORY 1946 & NEWER
#     70 2-STORY 1945 & OLDER
#     75 2-1/2 STORY ALL AGES
#     80 SPLIT OR MULTI-LEVEL
#     85 SPLIT FOYER
#     90 DUPLEX - ALL STYLES AND AGES
#     120 1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#     150 1-1/2 STORY PUD - ALL AGES
#     160 2-STORY PUD - 1946 & NEWER
#     180 PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#     190 2 FAMILY CONVERSION - ALL STYLES AND AGES
# MSZoning: Identifies the general zoning classification of the sale.
#     A Agriculture
#     C Commercial
#     FV Floating Village Residential
#     I Industrial
#     RH Residential High Density
#     RL Residential Low Density
#     RP Residential Low Density Park
#     RM Residential Medium Density
# LotFrontage: Linear feet of street connected to property
# LotArea: Lot size in square feet
# Street: Type of road access to property
#     Grvl Gravel
#     Pave Paved
# Alley: Type of alley access to property
#     Grvl Gravel
#     Pave Paved
#     NA No alley access
# LotShape: General shape of property
#     Reg Regular
#     IR1 Slightly irregular
#     IR2 Moderately Irregular
#     IR3 Irregular
# LandContour: Flatness of the property
#     Lvl Near Flat/Level
#     Bnk Banked - Quick and significant rise from street grade to building
#     HLS Hillside - Significant slope from side to side
#     Low Depression
# Utilities: Type of utilities available
#     AllPub All public Utilities (E,G,W,& S)
#     NoSewr Electricity, Gas, and Water (Septic Tank)
#     NoSeWa Electricity and Gas Only
#     ELO Electricity only
# LotConfig: Lot configuration
#     Inside Inside lot
#     Corner Corner lot
#     CulDSac Cul-de-sac
#     FR2 Frontage on 2 sides of property
#     FR3 Frontage on 3 sides of property
# LandSlope: Slope of property
#     Gtl Gentle slope
#     Mod Moderate Slope
#     Sev Severe Slope
# Neighborhood: Physical locations within Ames city limits
#     Blmngtn Bloomington Heights
#     Blueste Bluestem
#     BrDale Briardale
#     BrkSide Brookside
#     ClearCr Clear Creek
#     CollgCr College Creek
#     Crawfor Crawford
#     Edwards Edwards
#     Gilbert Gilbert
#     IDOTRR Iowa DOT and Rail Road
#     MeadowV Meadow Village
#     Mitchel Mitchell
#     Names North Ames
#     NoRidge Northridge
#     NPkVill Northpark Villa
#     NridgHt Northridge Heights
#     NWAmes Northwest Ames
#     OldTown Old Town
#     SWISU South & West of Iowa State University
#     Sawyer Sawyer
#     SawyerW Sawyer West
#     Somerst Somerset
#     StoneBr Stone Brook
#     Timber Timberland
#     Veenker Veenker
# Condition1: Proximity to main road or railroad
#     Artery Adjacent to arterial street
#     Feedr Adjacent to feeder street
#     Norm Normal
#     RRNn Within 200' of North-South Railroad
#     RRAn Adjacent to North-South Railroad
#     PosN Near positive off-site feature--park, greenbelt, etc.
#     PosA Adjacent to postive off-site feature
#     RRNe Within 200' of East-West Railroad
#     RRAe Adjacent to East-West Railroad
# Condition2: Proximity to main road or railroad (if a second is present)
#     Artery Adjacent to arterial street
#     Feedr Adjacent to feeder street
#     Norm Normal
#     RRNn Within 200' of North-South Railroad
#     RRAn Adjacent to North-South Railroad
#     PosN Near positive off-site feature--park, greenbelt, etc.
#     PosA Adjacent to postive off-site feature
#     RRNe Within 200' of East-West Railroad
#     RRAe Adjacent to East-West Railroad
# BldgType: Type of dwelling
#     1Fam Single-family Detached
#     2FmCon Two-family Conversion; originally built as one-family dwelling
#     Duplx Duplex
#     TwnhsE Townhouse End Unit
#     TwnhsI Townhouse Inside Unit
# HouseStyle: Style of dwelling
#     1Story One story
#     1.5Fin One and one-half story: 2nd level finished
#     1.5Unf One and one-half story: 2nd level unfinished
#     2Story Two story
#     2.5Fin Two and one-half story: 2nd level finished
#     2.5Unf Two and one-half story: 2nd level unfinished
#     SFoyer Split Foyer
#     SLvl Split Level
# OverallQual: Overall material and finish quality
#     10 Very Excellent
#     9 Excellent
#     8 Very Good
#     7 Good
#     6 Above Average
#     5 Average
#     4 Below Average
#     3 Fair
#     2 Poor
#     1 Very Poor
# OverallCond: Overall condition rating
#     10 Very Excellent
#     9 Excellent
#     8 Very Good
#     7 Good
#     6 Above Average
#     5 Average
#     4 Below Average
#     3 Fair
#     2 Poor
#     1 Very Poor
# YearBuilt: Original construction date
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
# RoofStyle: Type of roof
#     Flat Flat
#     Gable Gable
#     Gambrel Gabrel (Barn)
#     Hip Hip
#     Mansard Mansard
#     Shed Shed
# RoofMatl: Roof material
#     ClyTile Clay or Tile
#     CompShg Standard (Composite) Shingle
#     Membran Membrane
#     Metal Metal
#     Roll Roll
#     Tar&Grv Gravel & Tar
#     WdShake Wood Shakes
#     WdShngl Wood Shingles
# Exterior1st: Exterior covering on house
#     AsbShng Asbestos Shingles
#     AsphShn Asphalt Shingles
#     BrkComm Brick Common
#     BrkFace Brick Face
#     CBlock Cinder Block
#     CemntBd Cement Board
#     HdBoard Hard Board
#     ImStucc Imitation Stucco
#     MetalSd Metal Siding
#     Other Other
#     Plywood Plywood
#     PreCast PreCast
#     Stone Stone
#     Stucco Stucco
#     VinylSd Vinyl Siding
#     Wd Sdng Wood Siding
#     WdShing Wood Shingles
# Exterior2nd: Exterior covering on house (if more than one material)
#     AsbShng Asbestos Shingles
#     AsphShn Asphalt Shingles
#     BrkComm Brick Common
#     BrkFace Brick Face
#     CBlock Cinder Block
#     CemntBd Cement Board
#     HdBoard Hard Board
#     ImStucc Imitation Stucco
#     MetalSd Metal Siding
#     Other Other
#     Plywood Plywood
#     PreCast PreCast
#     Stone Stone
#     Stucco Stucco
#     VinylSd Vinyl Siding
#     Wd Sdng Wood Siding
#     WdShing Wood Shingles
# MasVnrType: Masonry veneer type
#     BrkCmn Brick Common
#     BrkFace Brick Face
#     CBlock Cinder Block
#     None None
#     Stone Stone
# MasVnrArea: Masonry veneer area in square feet
# ExterQual: Exterior material quality
#     Ex Excellent
#     Gd Good
#     TA Average/Typical
#     Fa Fair
#     Po Poor
# ExterCond: Present condition of the material on the exterior
#     Ex Excellent
#     Gd Good
#     TA Average/Typical
#     Fa Fair
#     Po Poor
# Foundation: Type of foundation
#     BrkTil Brick & Tile
#     CBlock Cinder Block
#     PConc Poured Contrete
#     Slab Slab
#     Stone Stone
#     Wood Wood
# BsmtQual: Height of the basement
#     Ex Excellent (100+ inches)
#     Gd Good (90-99 inches)
#     TA Typical (80-89 inches)
#     Fa Fair (70-79 inches)
#     Po Poor (<70 inches)
#     NA No Basement
# BsmtCond: General condition of the basement
#     Ex Excellent
#     Gd Good
#     TA Typical - slight dampness allowed
#     Fa Fair - dampness or some cracking or settling
#     Po Poor - Severe cracking, settling, or wetness
#     NA No Basement
# BsmtExposure: Walkout or garden level basement walls
#     Gd Good Exposure
#     Av Average Exposure (split levels or foyers typically score average or above)
#     Mn Mimimum Exposure
#     No No Exposure
#     NA No Basement
# BsmtFinType1: Quality of basement finished area
#     GLQ Good Living Quarters
#     ALQ Average Living Quarters
#     BLQ Below Average Living Quarters
#     Rec Average Rec Room
#     LwQ Low Quality
#     Unf Unfinshed
#     NA No Basement
# BsmtFinSF1: Type 1 finished square feet
# BsmtFinType2: Quality of second finished area (if present)
#     GLQ Good Living Quarters
#     ALQ Average Living Quarters
#     BLQ Below Average Living Quarters
#     Rec Average Rec Room
#     LwQ Low Quality
#     Unf Unfinshed
#     NA No Basement
# BsmtFinSF2: Type 2 finished square feet
# BsmtUnfSF: Unfinished square feet of basement area
# TotalBsmtSF: Total square feet of basement area
# Heating: Type of heating
#     Floor Floor Furnace
#     GasA Gas forced warm air furnace
#     GasW Gas hot water or steam heat
#     Grav Gravity furnace
#     OthW Hot water or steam heat other than gas
#     Wall Wall furnace
# HeatingQC: Heating quality and condition
#     Ex Excellent
#     Gd Good
#     TA Average/Typical
#     Fa Fair
#     Po Poor
# CentralAir: Central air conditioning
#     N No
#     Y Yes
# Electrical: Electrical system
#     SBrkr Standard Circuit Breakers & Romex
#     FuseA Fuse Box over 60 AMP and all Romex wiring (Average)
#     FuseF 60 AMP Fuse Box and mostly Romex wiring (Fair)
#     FuseP 60 AMP Fuse Box and mostly knob & tube wiring (poor)
#     Mix Mixed
# 1stFlrSF: First Floor square feet
# 2ndFlrSF: Second floor square feet
# LowQualFinSF: Low quality finished square feet (all floors)
# GrLivArea: Above grade (ground) living area square feet
# BsmtFullBath: Basement full bathrooms
# BsmtHalfBath: Basement half bathrooms
# FullBath: Full bathrooms above grade
# HalfBath: Half baths above grade
# Bedroom: Number of bedrooms above basement level
# Kitchen: Number of kitchens
# KitchenQual: Kitchen quality
#     Ex Excellent
#     Gd Good
#     TA Typical/Average
#     Fa Fair
#     Po Poor
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# Functional: Home functionality rating
#     Typ Typical Functionality
#     Min1 Minor Deductions 1
#     Min2 Minor Deductions 2
#     Mod Moderate Deductions
#     Maj1 Major Deductions 1
#     Maj2 Major Deductions 2
#     Sev Severely Damaged
#     Sal Salvage only
# Fireplaces: Number of fireplaces
# FireplaceQu: Fireplace quality
#     Ex Excellent - Exceptional Masonry Fireplace
#     Gd Good - Masonry Fireplace in main level
#     TA Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#     Fa Fair - Prefabricated Fireplace in basement
#     Po Poor - Ben Franklin Stove
#     NA No Fireplace
# GarageType: Garage location
#     2Types More than one type of garage
#     Attchd Attached to home
#     Basment Basement Garage
#     BuiltIn Built-In (Garage part of house - typically has room above garage)
#     CarPort Car Port
#     Detchd Detached from home
#     NA No Garage
# GarageYrBlt: Year garage was built
# GarageFinish: Interior finish of the garage
#     Fin Finished
#     RFn Rough Finished
#     Unf Unfinished
#     NA No Garage
# GarageCars: Size of garage in car capacity
# GarageArea: Size of garage in square feet
# GarageQual: Garage quality
#     Ex Excellent
#     Gd Good
#     TA Typical/Average
#     Fa Fair
#     Po Poor
#     NA No Garage
#     GarageCond: Garage condition
#     Ex Excellent
#     Gd Good
#     TA Typical/Average
#     Fa Fair
#     Po Poor
#     NA No Garage
# PavedDrive: Paved driveway
#     Y Paved
#     P Partial Pavement
#     N Dirt/Gravel
# WoodDeckSF: Wood deck area in square feet
# OpenPorchSF: Open porch area in square feet
# EnclosedPorch: Enclosed porch area in square feet
# 3SsnPorch: Three season porch area in square feet
# ScreenPorch: Screen porch area in square feet
# PoolArea: Pool area in square feet
# PoolQC: Pool quality
#     Ex Excellent
#     Gd Good
#     TA Average/Typical
#     Fa Fair
#     NA No Pool
# Fence: Fence quality
#     GdPrv Good Privacy
#     MnPrv Minimum Privacy
#     GdWo Good Wood
#     MnWw Minimum Wood/Wire
#     NA No Fence
# MiscFeature: Miscellaneous feature not covered in other categories
#     Elev Elevator
#     Gar2 2nd Garage (if not described in garage section)
#     Othr Other
#     Shed Shed (over 100 SF)
#     TenC Tennis Court
#     NA None
# MiscVal: $Value of miscellaneous feature
# MoSold: Month Sold
# YrSold: Year Sold
# SaleType: Type of sale
#     WD Warranty Deed - Conventional
#     CWD Warranty Deed - Cash
#     VWD Warranty Deed - VA Loan
#     New Home just constructed and sold
#     COD Court Officer Deed/Estate
#     Con Contract 15% Down payment regular terms
#     ConLw Contract Low Down payment and low interest
#     ConLI Contract Low Interest
#     ConLD Contract Low Down
#     Oth Other

```


# Project 2: The Ames Housing Data and House Price Prediction
## By Evan Jacobs, data genius

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



sns.set_style('darkgrid')

%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```

Next, we import out data files, first saving the names as variables. 


```python
train_csv = './train.csv'
test_csv = './test.csv'
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
      <th>1018</th>
      <td>1780</td>
      <td>528431030</td>
      <td>20</td>
      <td>RL</td>
      <td>76.0</td>
      <td>10612</td>
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
      <td>1</td>
      <td>2007</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>1931</th>
      <td>1118</td>
      <td>528431120</td>
      <td>60</td>
      <td>RL</td>
      <td>73.0</td>
      <td>9801</td>
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
      <td>7</td>
      <td>2008</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>1079</th>
      <td>2014</td>
      <td>903231090</td>
      <td>50</td>
      <td>RM</td>
      <td>NaN</td>
      <td>6240</td>
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
      <th>374</th>
      <td>184</td>
      <td>902305110</td>
      <td>70</td>
      <td>RM</td>
      <td>60.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>Grvl</td>
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
      <th>978</th>
      <td>536</td>
      <td>531363050</td>
      <td>20</td>
      <td>RL</td>
      <td>63.0</td>
      <td>7500</td>
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
      <td>8</td>
      <td>2009</td>
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
      <td>1290.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1538.000000</td>
      <td>1518.000000</td>
      <td>...</td>
      <td>1537.000000</td>
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
      <td>1466.042263</td>
      <td>7.122693e+08</td>
      <td>57.093628</td>
      <td>68.985271</td>
      <td>10055.761378</td>
      <td>6.122887</td>
      <td>5.555917</td>
      <td>1972.218466</td>
      <td>1984.492848</td>
      <td>100.474967</td>
      <td>...</td>
      <td>475.744307</td>
      <td>93.388166</td>
      <td>46.691808</td>
      <td>21.791938</td>
      <td>2.695709</td>
      <td>16.674252</td>
      <td>3.197659</td>
      <td>52.877113</td>
      <td>6.237971</td>
      <td>2007.788036</td>
    </tr>
    <tr>
      <th>std</th>
      <td>842.579820</td>
      <td>1.887129e+08</td>
      <td>42.431456</td>
      <td>23.782384</td>
      <td>7293.459355</td>
      <td>1.434716</td>
      <td>1.077870</td>
      <td>29.941785</td>
      <td>20.836171</td>
      <td>178.317469</td>
      <td>...</td>
      <td>217.255068</td>
      <td>129.897941</td>
      <td>65.047772</td>
      <td>59.122929</td>
      <td>23.490991</td>
      <td>57.274350</td>
      <td>43.605315</td>
      <td>573.162359</td>
      <td>2.761255</td>
      <td>1.314278</td>
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
      <td>1875.000000</td>
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
      <td>744.500000</td>
      <td>5.284581e+08</td>
      <td>20.000000</td>
      <td>58.000000</td>
      <td>7424.250000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1966.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>336.000000</td>
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
      <td>1492.500000</td>
      <td>5.354531e+08</td>
      <td>50.000000</td>
      <td>68.000000</td>
      <td>9314.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1975.000000</td>
      <td>1993.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>480.000000</td>
      <td>0.000000</td>
      <td>27.000000</td>
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
      <td>2186.000000</td>
      <td>9.071758e+08</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11425.750000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2001.000000</td>
      <td>2004.000000</td>
      <td>163.000000</td>
      <td>...</td>
      <td>576.000000</td>
      <td>168.000000</td>
      <td>68.000000</td>
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
      <td>523.000000</td>
      <td>368.000000</td>
      <td>323.000000</td>
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
    Int64Index: 1538 entries, 1018 to 1609
    Data columns (total 80 columns):
    Id                 1538 non-null int64
    PID                1538 non-null int64
    MS SubClass        1538 non-null int64
    MS Zoning          1538 non-null object
    Lot Frontage       1290 non-null float64
    Lot Area           1538 non-null int64
    Street             1538 non-null object
    Alley              109 non-null object
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
    Mas Vnr Type       1518 non-null object
    Mas Vnr Area       1518 non-null float64
    Exter Qual         1538 non-null object
    Exter Cond         1538 non-null object
    Foundation         1538 non-null object
    Bsmt Qual          1492 non-null object
    Bsmt Cond          1492 non-null object
    Bsmt Exposure      1490 non-null object
    BsmtFin Type 1     1492 non-null object
    BsmtFin SF 1       1537 non-null float64
    BsmtFin Type 2     1492 non-null object
    BsmtFin SF 2       1537 non-null float64
    Bsmt Unf SF        1537 non-null float64
    Total Bsmt SF      1537 non-null float64
    Heating            1538 non-null object
    Heating QC         1538 non-null object
    Central Air        1538 non-null object
    Electrical         1538 non-null object
    1st Flr SF         1538 non-null int64
    2nd Flr SF         1538 non-null int64
    Low Qual Fin SF    1538 non-null int64
    Gr Liv Area        1538 non-null int64
    Bsmt Full Bath     1536 non-null float64
    Bsmt Half Bath     1536 non-null float64
    Full Bath          1538 non-null int64
    Half Bath          1538 non-null int64
    Bedroom AbvGr      1538 non-null int64
    Kitchen AbvGr      1538 non-null int64
    Kitchen Qual       1538 non-null object
    TotRms AbvGrd      1538 non-null int64
    Functional         1538 non-null object
    Fireplaces         1538 non-null int64
    Fireplace Qu       786 non-null object
    Garage Type        1450 non-null object
    Garage Yr Blt      1449 non-null float64
    Garage Finish      1449 non-null object
    Garage Cars        1537 non-null float64
    Garage Area        1537 non-null float64
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
    Fence              290 non-null object
    Misc Feature       49 non-null object
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
    lotfrontage       248
    lotarea             0
    street              0
    alley            1429
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
    masvnrtype         20
    masvnrarea         20
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
    fireplacequ       752
    garagetype         88
    garageyrblt        89
    garagefinish       89
    garagecars          1
    garagearea          1
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
    fence            1248
    miscfeature      1489
    miscval             0
    mosold              0
    yrsold              0
    saletype            0
    Length: 80, dtype: int64



Here's a trick I learned.


```python
X_train.isna().sum()[X_train.isna().sum() !=0]
```




    lotfrontage      248
    alley           1429
    masvnrtype        20
    masvnrarea        20
    bsmtqual          46
    bsmtcond          46
    bsmtexposure      48
    bsmtfintype1      46
    bsmtfinsf1         1
    bsmtfintype2      46
    bsmtfinsf2         1
    bsmtunfsf          1
    totalbsmtsf        1
    bsmtfullbath       2
    bsmthalfbath       2
    fireplacequ      752
    garagetype        88
    garageyrblt       89
    garagefinish      89
    garagecars         1
    garagearea         1
    garagequal        89
    garagecond        89
    poolqc          1529
    fence           1248
    miscfeature     1489
    dtype: int64



Having a look at what the object column with null values looks like. 


```python
X_train.poolqc.unique()
```




    array([nan, 'Gd', 'Fa', 'TA', 'Ex'], dtype=object)



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
    Int64Index: 1538 entries, 1018 to 1609
    Data columns (total 80 columns):
    id               1538 non-null int64
    pid              1538 non-null int64
    mssubclass       1538 non-null int64
    mszoning         1538 non-null object
    lotfrontage      1538 non-null float64
    lotarea          1538 non-null int64
    street           1538 non-null object
    alley            109 non-null object
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
    masvnrtype       1518 non-null object
    masvnrarea       1538 non-null float64
    exterqual        1538 non-null object
    extercond        1538 non-null object
    foundation       1538 non-null object
    bsmtqual         1492 non-null object
    bsmtcond         1492 non-null object
    bsmtexposure     1490 non-null object
    bsmtfintype1     1492 non-null object
    bsmtfinsf1       1538 non-null float64
    bsmtfintype2     1492 non-null object
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
    fireplacequ      786 non-null object
    garagetype       1450 non-null object
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
    fence            290 non-null object
    miscfeature      49 non-null object
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
X_ts_num
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
      <th>997</th>
      <td>2492</td>
      <td>532376180</td>
      <td>20</td>
      <td>65.000000</td>
      <td>8450</td>
      <td>5</td>
      <td>6</td>
      <td>1968</td>
      <td>1968</td>
      <td>90.0</td>
      <td>...</td>
      <td>0</td>
      <td>155</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>704</th>
      <td>1298</td>
      <td>902134100</td>
      <td>30</td>
      <td>60.000000</td>
      <td>6756</td>
      <td>5</td>
      <td>6</td>
      <td>1910</td>
      <td>1950</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>854</th>
      <td>2873</td>
      <td>910200020</td>
      <td>30</td>
      <td>50.000000</td>
      <td>7288</td>
      <td>5</td>
      <td>6</td>
      <td>1942</td>
      <td>1950</td>
      <td>0.0</td>
      <td>...</td>
      <td>160</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>1401</td>
      <td>905352070</td>
      <td>20</td>
      <td>82.000000</td>
      <td>20270</td>
      <td>7</td>
      <td>6</td>
      <td>1979</td>
      <td>1979</td>
      <td>0.0</td>
      <td>...</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1832</th>
      <td>364</td>
      <td>527166010</td>
      <td>60</td>
      <td>69.264501</td>
      <td>10762</td>
      <td>7</td>
      <td>5</td>
      <td>1999</td>
      <td>1999</td>
      <td>344.0</td>
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
      <td>2009</td>
    </tr>
    <tr>
      <th>1838</th>
      <td>935</td>
      <td>909451140</td>
      <td>160</td>
      <td>24.000000</td>
      <td>1612</td>
      <td>6</td>
      <td>6</td>
      <td>1980</td>
      <td>1980</td>
      <td>0.0</td>
      <td>...</td>
      <td>154</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1258</th>
      <td>2575</td>
      <td>535152280</td>
      <td>20</td>
      <td>70.000000</td>
      <td>8400</td>
      <td>5</td>
      <td>5</td>
      <td>1957</td>
      <td>1957</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>187</th>
      <td>1779</td>
      <td>528429110</td>
      <td>20</td>
      <td>49.000000</td>
      <td>15256</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>84.0</td>
      <td>...</td>
      <td>168</td>
      <td>160</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>951</th>
      <td>1319</td>
      <td>902330010</td>
      <td>70</td>
      <td>50.000000</td>
      <td>5250</td>
      <td>8</td>
      <td>5</td>
      <td>1872</td>
      <td>1987</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>54</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1406</th>
      <td>1228</td>
      <td>534479150</td>
      <td>20</td>
      <td>63.000000</td>
      <td>7584</td>
      <td>5</td>
      <td>5</td>
      <td>1953</td>
      <td>1953</td>
      <td>88.0</td>
      <td>...</td>
      <td>120</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>908</th>
      <td>2559</td>
      <td>534455080</td>
      <td>20</td>
      <td>80.000000</td>
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
      <th>886</th>
      <td>999</td>
      <td>527107210</td>
      <td>60</td>
      <td>57.000000</td>
      <td>21872</td>
      <td>7</td>
      <td>5</td>
      <td>1996</td>
      <td>1997</td>
      <td>0.0</td>
      <td>...</td>
      <td>264</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1151</th>
      <td>1844</td>
      <td>533213140</td>
      <td>160</td>
      <td>34.000000</td>
      <td>3230</td>
      <td>6</td>
      <td>5</td>
      <td>1999</td>
      <td>1999</td>
      <td>1129.0</td>
      <td>...</td>
      <td>0</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>623</th>
      <td>457</td>
      <td>528176030</td>
      <td>20</td>
      <td>100.000000</td>
      <td>14836</td>
      <td>10</td>
      <td>5</td>
      <td>2004</td>
      <td>2005</td>
      <td>730.0</td>
      <td>...</td>
      <td>226</td>
      <td>235</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1907</th>
      <td>1236</td>
      <td>535150280</td>
      <td>20</td>
      <td>71.000000</td>
      <td>9204</td>
      <td>5</td>
      <td>5</td>
      <td>1963</td>
      <td>1963</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>88</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>189</th>
      <td>2334</td>
      <td>527212040</td>
      <td>60</td>
      <td>82.000000</td>
      <td>12438</td>
      <td>8</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>466.0</td>
      <td>...</td>
      <td>324</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>415</th>
      <td>2045</td>
      <td>904100100</td>
      <td>70</td>
      <td>107.000000</td>
      <td>12888</td>
      <td>7</td>
      <td>8</td>
      <td>1937</td>
      <td>1980</td>
      <td>0.0</td>
      <td>...</td>
      <td>521</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1429</th>
      <td>1547</td>
      <td>910202050</td>
      <td>30</td>
      <td>40.000000</td>
      <td>3636</td>
      <td>4</td>
      <td>4</td>
      <td>1922</td>
      <td>1950</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>1871</td>
      <td>534175010</td>
      <td>90</td>
      <td>69.264501</td>
      <td>11500</td>
      <td>5</td>
      <td>6</td>
      <td>1976</td>
      <td>1976</td>
      <td>164.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>627</th>
      <td>1439</td>
      <td>907187030</td>
      <td>60</td>
      <td>57.000000</td>
      <td>8924</td>
      <td>7</td>
      <td>5</td>
      <td>1998</td>
      <td>1999</td>
      <td>0.0</td>
      <td>...</td>
      <td>120</td>
      <td>155</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>931</th>
      <td>1036</td>
      <td>527402130</td>
      <td>20</td>
      <td>68.000000</td>
      <td>10295</td>
      <td>4</td>
      <td>6</td>
      <td>1969</td>
      <td>1969</td>
      <td>72.0</td>
      <td>...</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>370</th>
      <td>2567</td>
      <td>535103070</td>
      <td>80</td>
      <td>69.264501</td>
      <td>12700</td>
      <td>6</td>
      <td>5</td>
      <td>1964</td>
      <td>1964</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>69</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1224</th>
      <td>1631</td>
      <td>527175130</td>
      <td>20</td>
      <td>160.000000</td>
      <td>18160</td>
      <td>6</td>
      <td>6</td>
      <td>1964</td>
      <td>1964</td>
      <td>138.0</td>
      <td>...</td>
      <td>0</td>
      <td>108</td>
      <td>246</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>547</th>
      <td>2677</td>
      <td>903231190</td>
      <td>50</td>
      <td>69.264501</td>
      <td>6240</td>
      <td>5</td>
      <td>4</td>
      <td>1936</td>
      <td>1950</td>
      <td>0.0</td>
      <td>...</td>
      <td>200</td>
      <td>114</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1479</th>
      <td>653</td>
      <td>535354070</td>
      <td>20</td>
      <td>60.000000</td>
      <td>9600</td>
      <td>5</td>
      <td>6</td>
      <td>1950</td>
      <td>1950</td>
      <td>0.0</td>
      <td>...</td>
      <td>126</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>778</th>
      <td>38</td>
      <td>528112020</td>
      <td>20</td>
      <td>98.000000</td>
      <td>11478</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2008</td>
      <td>200.0</td>
      <td>...</td>
      <td>0</td>
      <td>50</td>
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
      <th>1552</th>
      <td>403</td>
      <td>527450280</td>
      <td>160</td>
      <td>21.000000</td>
      <td>1680</td>
      <td>6</td>
      <td>5</td>
      <td>1973</td>
      <td>1973</td>
      <td>504.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>2815</td>
      <td>907420070</td>
      <td>60</td>
      <td>65.000000</td>
      <td>8461</td>
      <td>6</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1067</th>
      <td>1630</td>
      <td>527165170</td>
      <td>60</td>
      <td>69.264501</td>
      <td>7655</td>
      <td>6</td>
      <td>5</td>
      <td>1993</td>
      <td>1994</td>
      <td>0.0</td>
      <td>...</td>
      <td>290</td>
      <td>84</td>
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
      <th>1320</th>
      <td>2104</td>
      <td>906380070</td>
      <td>60</td>
      <td>69.000000</td>
      <td>9588</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>270.0</td>
      <td>...</td>
      <td>0</td>
      <td>148</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1199</th>
      <td>2272</td>
      <td>916475020</td>
      <td>20</td>
      <td>85.000000</td>
      <td>14536</td>
      <td>8</td>
      <td>5</td>
      <td>2002</td>
      <td>2003</td>
      <td>236.0</td>
      <td>...</td>
      <td>0</td>
      <td>252</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>608</th>
      <td>2868</td>
      <td>909425120</td>
      <td>20</td>
      <td>90.000000</td>
      <td>14115</td>
      <td>6</td>
      <td>7</td>
      <td>1956</td>
      <td>2004</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>280</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2039</th>
      <td>2288</td>
      <td>923228220</td>
      <td>160</td>
      <td>21.000000</td>
      <td>1495</td>
      <td>4</td>
      <td>6</td>
      <td>1970</td>
      <td>1970</td>
      <td>189.0</td>
      <td>...</td>
      <td>0</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1539</th>
      <td>1381</td>
      <td>905104110</td>
      <td>20</td>
      <td>68.000000</td>
      <td>10265</td>
      <td>5</td>
      <td>7</td>
      <td>1967</td>
      <td>2005</td>
      <td>0.0</td>
      <td>...</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>600</td>
      <td>7</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>313</th>
      <td>527</td>
      <td>528480130</td>
      <td>60</td>
      <td>65.000000</td>
      <td>8125</td>
      <td>7</td>
      <td>5</td>
      <td>2006</td>
      <td>2007</td>
      <td>100.0</td>
      <td>...</td>
      <td>0</td>
      <td>168</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>905</th>
      <td>1574</td>
      <td>916380060</td>
      <td>20</td>
      <td>74.000000</td>
      <td>11563</td>
      <td>8</td>
      <td>5</td>
      <td>2006</td>
      <td>2007</td>
      <td>258.0</td>
      <td>...</td>
      <td>0</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>695</th>
      <td>1450</td>
      <td>907202240</td>
      <td>20</td>
      <td>40.000000</td>
      <td>14330</td>
      <td>5</td>
      <td>6</td>
      <td>1975</td>
      <td>2001</td>
      <td>0.0</td>
      <td>...</td>
      <td>140</td>
      <td>0</td>
      <td>239</td>
      <td>0</td>
      <td>227</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>798</th>
      <td>312</td>
      <td>914476520</td>
      <td>20</td>
      <td>129.000000</td>
      <td>9196</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>0.0</td>
      <td>...</td>
      <td>100</td>
      <td>150</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>1789</th>
      <td>1197</td>
      <td>534225120</td>
      <td>90</td>
      <td>74.000000</td>
      <td>13101</td>
      <td>5</td>
      <td>5</td>
      <td>1965</td>
      <td>1965</td>
      <td>108.0</td>
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
    <tr>
      <th>1949</th>
      <td>486</td>
      <td>528280230</td>
      <td>60</td>
      <td>69.264501</td>
      <td>12224</td>
      <td>6</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>40.0</td>
      <td>...</td>
      <td>24</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>1597</td>
      <td>923225310</td>
      <td>180</td>
      <td>21.000000</td>
      <td>1974</td>
      <td>4</td>
      <td>7</td>
      <td>1973</td>
      <td>2006</td>
      <td>0.0</td>
      <td>...</td>
      <td>120</td>
      <td>101</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>207</th>
      <td>323</td>
      <td>923201020</td>
      <td>20</td>
      <td>85.000000</td>
      <td>15300</td>
      <td>5</td>
      <td>5</td>
      <td>1965</td>
      <td>1977</td>
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
      <td>6</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>284</th>
      <td>1629</td>
      <td>527165100</td>
      <td>80</td>
      <td>69.264501</td>
      <td>9125</td>
      <td>7</td>
      <td>5</td>
      <td>1992</td>
      <td>1992</td>
      <td>170.0</td>
      <td>...</td>
      <td>100</td>
      <td>25</td>
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
      <th>1640</th>
      <td>725</td>
      <td>902405120</td>
      <td>50</td>
      <td>60.000000</td>
      <td>5400</td>
      <td>6</td>
      <td>6</td>
      <td>1920</td>
      <td>1950</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>146</th>
      <td>888</td>
      <td>908128060</td>
      <td>85</td>
      <td>64.000000</td>
      <td>7301</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>500.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>177</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>444</td>
      <td>528142090</td>
      <td>20</td>
      <td>107.000000</td>
      <td>11362</td>
      <td>8</td>
      <td>5</td>
      <td>2004</td>
      <td>2005</td>
      <td>42.0</td>
      <td>...</td>
      <td>125</td>
      <td>185</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>867</th>
      <td>1115</td>
      <td>528429080</td>
      <td>20</td>
      <td>75.000000</td>
      <td>14587</td>
      <td>9</td>
      <td>5</td>
      <td>2008</td>
      <td>2008</td>
      <td>284.0</td>
      <td>...</td>
      <td>0</td>
      <td>174</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1902</th>
      <td>2087</td>
      <td>905478220</td>
      <td>50</td>
      <td>60.000000</td>
      <td>11100</td>
      <td>5</td>
      <td>6</td>
      <td>1951</td>
      <td>1994</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>68</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>293</th>
      <td>1470</td>
      <td>907290250</td>
      <td>120</td>
      <td>69.264501</td>
      <td>4426</td>
      <td>6</td>
      <td>5</td>
      <td>2004</td>
      <td>2004</td>
      <td>205.0</td>
      <td>...</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1393</th>
      <td>1200</td>
      <td>534250370</td>
      <td>60</td>
      <td>69.264501</td>
      <td>8963</td>
      <td>8</td>
      <td>9</td>
      <td>1976</td>
      <td>1996</td>
      <td>289.0</td>
      <td>...</td>
      <td>0</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>738</th>
      <td>1979</td>
      <td>902110010</td>
      <td>30</td>
      <td>56.000000</td>
      <td>3153</td>
      <td>5</td>
      <td>6</td>
      <td>1920</td>
      <td>1990</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1037</th>
      <td>28</td>
      <td>527425090</td>
      <td>20</td>
      <td>70.000000</td>
      <td>10500</td>
      <td>4</td>
      <td>5</td>
      <td>1971</td>
      <td>1971</td>
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
      <td>4</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>829</th>
      <td>1301</td>
      <td>902201140</td>
      <td>20</td>
      <td>100.000000</td>
      <td>12000</td>
      <td>5</td>
      <td>7</td>
      <td>1948</td>
      <td>2005</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>1056</td>
      <td>528110080</td>
      <td>20</td>
      <td>107.000000</td>
      <td>13891</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2008</td>
      <td>436.0</td>
      <td>...</td>
      <td>0</td>
      <td>102</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1602</th>
      <td>320</td>
      <td>916455170</td>
      <td>60</td>
      <td>80.000000</td>
      <td>11316</td>
      <td>7</td>
      <td>5</td>
      <td>2002</td>
      <td>2003</td>
      <td>44.0</td>
      <td>...</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>2289</td>
      <td>923228250</td>
      <td>160</td>
      <td>21.000000</td>
      <td>2001</td>
      <td>4</td>
      <td>5</td>
      <td>1970</td>
      <td>1970</td>
      <td>80.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1873</th>
      <td>349</td>
      <td>527110020</td>
      <td>80</td>
      <td>69.264501</td>
      <td>8530</td>
      <td>7</td>
      <td>5</td>
      <td>1995</td>
      <td>1996</td>
      <td>22.0</td>
      <td>...</td>
      <td>120</td>
      <td>72</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>700</td>
      <td>5</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>2637</td>
      <td>902100130</td>
      <td>70</td>
      <td>57.000000</td>
      <td>9906</td>
      <td>4</td>
      <td>4</td>
      <td>1925</td>
      <td>1950</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>172</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>188</th>
      <td>2569</td>
      <td>535125060</td>
      <td>60</td>
      <td>88.000000</td>
      <td>14200</td>
      <td>7</td>
      <td>6</td>
      <td>1966</td>
      <td>1966</td>
      <td>309.0</td>
      <td>...</td>
      <td>105</td>
      <td>66</td>
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
      <th>319</th>
      <td>1108</td>
      <td>528365080</td>
      <td>60</td>
      <td>91.000000</td>
      <td>10010</td>
      <td>7</td>
      <td>5</td>
      <td>1993</td>
      <td>1994</td>
      <td>320.0</td>
      <td>...</td>
      <td>385</td>
      <td>99</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
<p>513 rows × 39 columns</p>
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
      <td>1.466042e+03</td>
      <td>8.425798e+02</td>
      <td>1.0</td>
      <td>7.445000e+02</td>
      <td>1.492500e+03</td>
      <td>2.186000e+03</td>
      <td>2930.0</td>
    </tr>
    <tr>
      <th>pid</th>
      <td>1538.0</td>
      <td>7.122693e+08</td>
      <td>1.887129e+08</td>
      <td>526301100.0</td>
      <td>5.284581e+08</td>
      <td>5.354531e+08</td>
      <td>9.071758e+08</td>
      <td>924152030.0</td>
    </tr>
    <tr>
      <th>mssubclass</th>
      <td>1538.0</td>
      <td>5.709363e+01</td>
      <td>4.243146e+01</td>
      <td>20.0</td>
      <td>2.000000e+01</td>
      <td>5.000000e+01</td>
      <td>7.000000e+01</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>lotfrontage</th>
      <td>1538.0</td>
      <td>6.898527e+01</td>
      <td>2.177935e+01</td>
      <td>21.0</td>
      <td>6.000000e+01</td>
      <td>6.898527e+01</td>
      <td>7.800000e+01</td>
      <td>313.0</td>
    </tr>
    <tr>
      <th>lotarea</th>
      <td>1538.0</td>
      <td>1.005576e+04</td>
      <td>7.293459e+03</td>
      <td>1300.0</td>
      <td>7.424250e+03</td>
      <td>9.314500e+03</td>
      <td>1.142575e+04</td>
      <td>159000.0</td>
    </tr>
    <tr>
      <th>overallqual</th>
      <td>1538.0</td>
      <td>6.122887e+00</td>
      <td>1.434716e+00</td>
      <td>1.0</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>7.000000e+00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>overallcond</th>
      <td>1538.0</td>
      <td>5.555917e+00</td>
      <td>1.077870e+00</td>
      <td>1.0</td>
      <td>5.000000e+00</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>yearbuilt</th>
      <td>1538.0</td>
      <td>1.972218e+03</td>
      <td>2.994178e+01</td>
      <td>1875.0</td>
      <td>1.954000e+03</td>
      <td>1.975000e+03</td>
      <td>2.001000e+03</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>yearremod/add</th>
      <td>1538.0</td>
      <td>1.984493e+03</td>
      <td>2.083617e+01</td>
      <td>1950.0</td>
      <td>1.966000e+03</td>
      <td>1.993000e+03</td>
      <td>2.004000e+03</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>masvnrarea</th>
      <td>1538.0</td>
      <td>1.004750e+02</td>
      <td>1.771535e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.600000e+02</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>bsmtfinsf1</th>
      <td>1538.0</td>
      <td>4.453084e+02</td>
      <td>4.672316e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>3.695000e+02</td>
      <td>7.280000e+02</td>
      <td>5644.0</td>
    </tr>
    <tr>
      <th>bsmtfinsf2</th>
      <td>1538.0</td>
      <td>4.942550e+01</td>
      <td>1.676362e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1393.0</td>
    </tr>
    <tr>
      <th>bsmtunfsf</th>
      <td>1538.0</td>
      <td>5.643012e+02</td>
      <td>4.422569e+02</td>
      <td>0.0</td>
      <td>2.162500e+02</td>
      <td>4.750000e+02</td>
      <td>8.107500e+02</td>
      <td>2336.0</td>
    </tr>
    <tr>
      <th>totalbsmtsf</th>
      <td>1538.0</td>
      <td>1.059035e+03</td>
      <td>4.604695e+02</td>
      <td>0.0</td>
      <td>7.922500e+02</td>
      <td>9.915000e+02</td>
      <td>1.329750e+03</td>
      <td>6110.0</td>
    </tr>
    <tr>
      <th>1stflrsf</th>
      <td>1538.0</td>
      <td>1.168811e+03</td>
      <td>4.054156e+02</td>
      <td>334.0</td>
      <td>8.750000e+02</td>
      <td>1.100500e+03</td>
      <td>1.406500e+03</td>
      <td>5095.0</td>
    </tr>
    <tr>
      <th>2ndflrsf</th>
      <td>1538.0</td>
      <td>3.308225e+02</td>
      <td>4.237303e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>6.930000e+02</td>
      <td>1862.0</td>
    </tr>
    <tr>
      <th>lowqualfinsf</th>
      <td>1538.0</td>
      <td>4.007802e+00</td>
      <td>3.951297e+01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>528.0</td>
    </tr>
    <tr>
      <th>grlivarea</th>
      <td>1538.0</td>
      <td>1.503642e+03</td>
      <td>5.075192e+02</td>
      <td>334.0</td>
      <td>1.142000e+03</td>
      <td>1.443000e+03</td>
      <td>1.730750e+03</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <th>bsmtfullbath</th>
      <td>1538.0</td>
      <td>4.218750e-01</td>
      <td>5.156150e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>bsmthalfbath</th>
      <td>1538.0</td>
      <td>6.705729e-02</td>
      <td>2.602400e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>fullbath</th>
      <td>1538.0</td>
      <td>1.585826e+00</td>
      <td>5.453840e-01</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>halfbath</th>
      <td>1538.0</td>
      <td>3.693108e-01</td>
      <td>4.960688e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>bedroomabvgr</th>
      <td>1538.0</td>
      <td>2.851105e+00</td>
      <td>8.284558e-01</td>
      <td>0.0</td>
      <td>2.000000e+00</td>
      <td>3.000000e+00</td>
      <td>3.000000e+00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>kitchenabvgr</th>
      <td>1538.0</td>
      <td>1.040962e+00</td>
      <td>2.047252e-01</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>totrmsabvgrd</th>
      <td>1538.0</td>
      <td>6.448635e+00</td>
      <td>1.572876e+00</td>
      <td>2.0</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
      <td>7.000000e+00</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>fireplaces</th>
      <td>1538.0</td>
      <td>5.864759e-01</td>
      <td>6.335854e-01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>garageyrblt</th>
      <td>1538.0</td>
      <td>1.979240e+03</td>
      <td>2.460375e+01</td>
      <td>1895.0</td>
      <td>1.963000e+03</td>
      <td>1.979240e+03</td>
      <td>2.002000e+03</td>
      <td>2207.0</td>
    </tr>
    <tr>
      <th>garagecars</th>
      <td>1538.0</td>
      <td>1.782043e+00</td>
      <td>7.711386e-01</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>garagearea</th>
      <td>1538.0</td>
      <td>4.757443e+02</td>
      <td>2.171844e+02</td>
      <td>0.0</td>
      <td>3.360000e+02</td>
      <td>4.800000e+02</td>
      <td>5.760000e+02</td>
      <td>1418.0</td>
    </tr>
    <tr>
      <th>wooddecksf</th>
      <td>1538.0</td>
      <td>9.338817e+01</td>
      <td>1.298979e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.680000e+02</td>
      <td>1424.0</td>
    </tr>
    <tr>
      <th>openporchsf</th>
      <td>1538.0</td>
      <td>4.669181e+01</td>
      <td>6.504777e+01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>2.700000e+01</td>
      <td>6.800000e+01</td>
      <td>523.0</td>
    </tr>
    <tr>
      <th>enclosedporch</th>
      <td>1538.0</td>
      <td>2.179194e+01</td>
      <td>5.912293e+01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>368.0</td>
    </tr>
    <tr>
      <th>3ssnporch</th>
      <td>1538.0</td>
      <td>2.695709e+00</td>
      <td>2.349099e+01</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>323.0</td>
    </tr>
    <tr>
      <th>screenporch</th>
      <td>1538.0</td>
      <td>1.667425e+01</td>
      <td>5.727435e+01</td>
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
      <td>5.287711e+01</td>
      <td>5.731624e+02</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>17000.0</td>
    </tr>
    <tr>
      <th>mosold</th>
      <td>1538.0</td>
      <td>6.237971e+00</td>
      <td>2.761255e+00</td>
      <td>1.0</td>
      <td>4.000000e+00</td>
      <td>6.000000e+00</td>
      <td>8.000000e+00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>yrsold</th>
      <td>1538.0</td>
      <td>2.007788e+03</td>
      <td>1.314278e+00</td>
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
X_tr_num
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/ipykernel/__main__.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':





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
      <th>miscval</th>
      <th>mosold</th>
      <th>yrsold</th>
      <th>sp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1018</th>
      <td>1780</td>
      <td>528431030</td>
      <td>20</td>
      <td>76.000000</td>
      <td>10612</td>
      <td>8</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>248.000000</td>
      <td>...</td>
      <td>168</td>
      <td>46</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2007</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>1931</th>
      <td>1118</td>
      <td>528431120</td>
      <td>60</td>
      <td>73.000000</td>
      <td>9801</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>156.000000</td>
      <td>...</td>
      <td>144</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
      <td>257000</td>
    </tr>
    <tr>
      <th>1079</th>
      <td>2014</td>
      <td>903231090</td>
      <td>50</td>
      <td>68.985271</td>
      <td>6240</td>
      <td>6</td>
      <td>5</td>
      <td>1938</td>
      <td>1950</td>
      <td>0.000000</td>
      <td>...</td>
      <td>225</td>
      <td>0</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2007</td>
      <td>126000</td>
    </tr>
    <tr>
      <th>374</th>
      <td>184</td>
      <td>902305110</td>
      <td>70</td>
      <td>60.000000</td>
      <td>9600</td>
      <td>8</td>
      <td>9</td>
      <td>1900</td>
      <td>2003</td>
      <td>0.000000</td>
      <td>...</td>
      <td>54</td>
      <td>228</td>
      <td>246</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>150000</td>
    </tr>
    <tr>
      <th>978</th>
      <td>536</td>
      <td>531363050</td>
      <td>20</td>
      <td>63.000000</td>
      <td>7500</td>
      <td>6</td>
      <td>5</td>
      <td>2004</td>
      <td>2004</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
      <td>143500</td>
    </tr>
    <tr>
      <th>587</th>
      <td>975</td>
      <td>923225490</td>
      <td>20</td>
      <td>62.000000</td>
      <td>9858</td>
      <td>5</td>
      <td>6</td>
      <td>1968</td>
      <td>1968</td>
      <td>0.000000</td>
      <td>...</td>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>600</td>
      <td>11</td>
      <td>2009</td>
      <td>130000</td>
    </tr>
    <tr>
      <th>412</th>
      <td>1549</td>
      <td>910203020</td>
      <td>30</td>
      <td>71.000000</td>
      <td>6900</td>
      <td>5</td>
      <td>6</td>
      <td>1940</td>
      <td>1955</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>120500</td>
    </tr>
    <tr>
      <th>166</th>
      <td>1320</td>
      <td>902401010</td>
      <td>50</td>
      <td>68.985271</td>
      <td>5700</td>
      <td>7</td>
      <td>7</td>
      <td>1926</td>
      <td>1950</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
      <td>116900</td>
    </tr>
    <tr>
      <th>1779</th>
      <td>400</td>
      <td>527405180</td>
      <td>20</td>
      <td>70.000000</td>
      <td>8120</td>
      <td>4</td>
      <td>7</td>
      <td>1970</td>
      <td>1970</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2009</td>
      <td>124500</td>
    </tr>
    <tr>
      <th>2035</th>
      <td>2862</td>
      <td>909279040</td>
      <td>30</td>
      <td>80.000000</td>
      <td>11600</td>
      <td>4</td>
      <td>5</td>
      <td>1922</td>
      <td>1950</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>67</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
      <td>137500</td>
    </tr>
    <tr>
      <th>454</th>
      <td>104</td>
      <td>533223100</td>
      <td>160</td>
      <td>68.985271</td>
      <td>2403</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>147110</td>
    </tr>
    <tr>
      <th>924</th>
      <td>677</td>
      <td>535450160</td>
      <td>90</td>
      <td>60.000000</td>
      <td>8544</td>
      <td>3</td>
      <td>4</td>
      <td>1950</td>
      <td>1950</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2009</td>
      <td>92900</td>
    </tr>
    <tr>
      <th>975</th>
      <td>1596</td>
      <td>923225200</td>
      <td>160</td>
      <td>41.000000</td>
      <td>2665</td>
      <td>5</td>
      <td>7</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.000000</td>
      <td>...</td>
      <td>92</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2008</td>
      <td>129500</td>
    </tr>
    <tr>
      <th>561</th>
      <td>956</td>
      <td>916176030</td>
      <td>20</td>
      <td>68.985271</td>
      <td>14375</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>541.000000</td>
      <td>...</td>
      <td>0</td>
      <td>118</td>
      <td>0</td>
      <td>0</td>
      <td>233</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2009</td>
      <td>137500</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>2872</td>
      <td>909475020</td>
      <td>20</td>
      <td>68.985271</td>
      <td>16381</td>
      <td>6</td>
      <td>5</td>
      <td>1969</td>
      <td>1969</td>
      <td>312.000000</td>
      <td>...</td>
      <td>0</td>
      <td>73</td>
      <td>216</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2006</td>
      <td>223000</td>
    </tr>
    <tr>
      <th>2029</th>
      <td>969</td>
      <td>921128050</td>
      <td>20</td>
      <td>85.000000</td>
      <td>12633</td>
      <td>9</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>290.000000</td>
      <td>...</td>
      <td>308</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2009</td>
      <td>425000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2827</td>
      <td>908186070</td>
      <td>180</td>
      <td>35.000000</td>
      <td>3675</td>
      <td>6</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>82.000000</td>
      <td>...</td>
      <td>0</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>813</th>
      <td>142</td>
      <td>535152150</td>
      <td>20</td>
      <td>70.000000</td>
      <td>10552</td>
      <td>5</td>
      <td>5</td>
      <td>1959</td>
      <td>1959</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>165500</td>
    </tr>
    <tr>
      <th>920</th>
      <td>273</td>
      <td>907414030</td>
      <td>20</td>
      <td>65.000000</td>
      <td>8773</td>
      <td>7</td>
      <td>5</td>
      <td>2004</td>
      <td>2004</td>
      <td>98.000000</td>
      <td>...</td>
      <td>132</td>
      <td>105</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>185500</td>
    </tr>
    <tr>
      <th>973</th>
      <td>2588</td>
      <td>535325340</td>
      <td>20</td>
      <td>65.000000</td>
      <td>11050</td>
      <td>5</td>
      <td>5</td>
      <td>1956</td>
      <td>1956</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>288</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
      <td>133500</td>
    </tr>
    <tr>
      <th>862</th>
      <td>1793</td>
      <td>528445070</td>
      <td>60</td>
      <td>75.000000</td>
      <td>8778</td>
      <td>8</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2007</td>
      <td>221300</td>
    </tr>
    <tr>
      <th>912</th>
      <td>731</td>
      <td>903201080</td>
      <td>50</td>
      <td>55.000000</td>
      <td>7264</td>
      <td>7</td>
      <td>7</td>
      <td>1925</td>
      <td>2007</td>
      <td>0.000000</td>
      <td>...</td>
      <td>74</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>144</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
      <td>205000</td>
    </tr>
    <tr>
      <th>150</th>
      <td>420</td>
      <td>527455280</td>
      <td>20</td>
      <td>68.985271</td>
      <td>10710</td>
      <td>5</td>
      <td>7</td>
      <td>1966</td>
      <td>2004</td>
      <td>165.000000</td>
      <td>...</td>
      <td>0</td>
      <td>162</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1200</td>
      <td>7</td>
      <td>2009</td>
      <td>148800</td>
    </tr>
    <tr>
      <th>1274</th>
      <td>2368</td>
      <td>527450460</td>
      <td>160</td>
      <td>21.000000</td>
      <td>1890</td>
      <td>6</td>
      <td>7</td>
      <td>1972</td>
      <td>1972</td>
      <td>380.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
      <td>116000</td>
    </tr>
    <tr>
      <th>1642</th>
      <td>1216</td>
      <td>534426040</td>
      <td>20</td>
      <td>72.000000</td>
      <td>10007</td>
      <td>5</td>
      <td>7</td>
      <td>1959</td>
      <td>2006</td>
      <td>54.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
      <td>145500</td>
    </tr>
    <tr>
      <th>213</th>
      <td>1390</td>
      <td>905201020</td>
      <td>20</td>
      <td>94.000000</td>
      <td>9239</td>
      <td>5</td>
      <td>8</td>
      <td>1963</td>
      <td>2003</td>
      <td>0.000000</td>
      <td>...</td>
      <td>168</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2008</td>
      <td>144900</td>
    </tr>
    <tr>
      <th>911</th>
      <td>2300</td>
      <td>923252080</td>
      <td>20</td>
      <td>69.000000</td>
      <td>7599</td>
      <td>4</td>
      <td>6</td>
      <td>1982</td>
      <td>2006</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
      <td>129500</td>
    </tr>
    <tr>
      <th>1826</th>
      <td>744</td>
      <td>903231130</td>
      <td>30</td>
      <td>51.000000</td>
      <td>6120</td>
      <td>5</td>
      <td>7</td>
      <td>1931</td>
      <td>1993</td>
      <td>0.000000</td>
      <td>...</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>2009</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>365</th>
      <td>164</td>
      <td>535454070</td>
      <td>50</td>
      <td>71.000000</td>
      <td>8520</td>
      <td>5</td>
      <td>4</td>
      <td>1952</td>
      <td>1952</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>166000</td>
    </tr>
    <tr>
      <th>1426</th>
      <td>476</td>
      <td>528235090</td>
      <td>60</td>
      <td>68.985271</td>
      <td>8068</td>
      <td>6</td>
      <td>5</td>
      <td>2002</td>
      <td>2002</td>
      <td>0.000000</td>
      <td>...</td>
      <td>120</td>
      <td>46</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2009</td>
      <td>200000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>1306</td>
      <td>902207120</td>
      <td>20</td>
      <td>103.000000</td>
      <td>12205</td>
      <td>3</td>
      <td>1</td>
      <td>1949</td>
      <td>1992</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>65000</td>
    </tr>
    <tr>
      <th>808</th>
      <td>2109</td>
      <td>906382020</td>
      <td>20</td>
      <td>75.000000</td>
      <td>11750</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>204.000000</td>
      <td>...</td>
      <td>144</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
      <td>217000</td>
    </tr>
    <tr>
      <th>870</th>
      <td>1552</td>
      <td>910206110</td>
      <td>30</td>
      <td>60.000000</td>
      <td>7392</td>
      <td>5</td>
      <td>7</td>
      <td>1930</td>
      <td>1995</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>90</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2008</td>
      <td>99500</td>
    </tr>
    <tr>
      <th>1117</th>
      <td>1764</td>
      <td>528327060</td>
      <td>20</td>
      <td>68.985271</td>
      <td>11400</td>
      <td>10</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>705.000000</td>
      <td>...</td>
      <td>314</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2007</td>
      <td>466500</td>
    </tr>
    <tr>
      <th>1385</th>
      <td>2556</td>
      <td>534451120</td>
      <td>30</td>
      <td>51.000000</td>
      <td>5900</td>
      <td>4</td>
      <td>7</td>
      <td>1923</td>
      <td>1958</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2006</td>
      <td>85500</td>
    </tr>
    <tr>
      <th>1270</th>
      <td>1569</td>
      <td>914476080</td>
      <td>90</td>
      <td>76.000000</td>
      <td>10260</td>
      <td>5</td>
      <td>4</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>2008</td>
      <td>100000</td>
    </tr>
    <tr>
      <th>1263</th>
      <td>496</td>
      <td>528321010</td>
      <td>60</td>
      <td>174.000000</td>
      <td>15138</td>
      <td>8</td>
      <td>5</td>
      <td>1995</td>
      <td>1996</td>
      <td>506.000000</td>
      <td>...</td>
      <td>0</td>
      <td>146</td>
      <td>202</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2009</td>
      <td>403000</td>
    </tr>
    <tr>
      <th>229</th>
      <td>2033</td>
      <td>903453080</td>
      <td>80</td>
      <td>120.000000</td>
      <td>13200</td>
      <td>6</td>
      <td>6</td>
      <td>1963</td>
      <td>1963</td>
      <td>234.000000</td>
      <td>...</td>
      <td>0</td>
      <td>110</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2007</td>
      <td>202500</td>
    </tr>
    <tr>
      <th>1685</th>
      <td>443</td>
      <td>528142040</td>
      <td>60</td>
      <td>74.000000</td>
      <td>8834</td>
      <td>9</td>
      <td>5</td>
      <td>2004</td>
      <td>2005</td>
      <td>216.000000</td>
      <td>...</td>
      <td>184</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2009</td>
      <td>350000</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>2810</td>
      <td>907410100</td>
      <td>60</td>
      <td>70.000000</td>
      <td>8400</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
      <td>0.000000</td>
      <td>...</td>
      <td>144</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2006</td>
      <td>195800</td>
    </tr>
    <tr>
      <th>880</th>
      <td>2455</td>
      <td>528429060</td>
      <td>60</td>
      <td>75.000000</td>
      <td>12447</td>
      <td>8</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>192.000000</td>
      <td>...</td>
      <td>200</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2006</td>
      <td>252000</td>
    </tr>
    <tr>
      <th>338</th>
      <td>1800</td>
      <td>528458150</td>
      <td>60</td>
      <td>112.000000</td>
      <td>12217</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>100.474967</td>
      <td>...</td>
      <td>168</td>
      <td>127</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2007</td>
      <td>310013</td>
    </tr>
    <tr>
      <th>1313</th>
      <td>15</td>
      <td>527182190</td>
      <td>120</td>
      <td>68.985271</td>
      <td>6820</td>
      <td>8</td>
      <td>5</td>
      <td>1985</td>
      <td>1985</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>54</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>212000</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>513</td>
      <td>528441020</td>
      <td>60</td>
      <td>75.000000</td>
      <td>9675</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2009</td>
      <td>253000</td>
    </tr>
    <tr>
      <th>537</th>
      <td>1244</td>
      <td>535180100</td>
      <td>20</td>
      <td>75.000000</td>
      <td>9464</td>
      <td>6</td>
      <td>7</td>
      <td>1958</td>
      <td>1958</td>
      <td>135.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>130</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>136000</td>
    </tr>
    <tr>
      <th>814</th>
      <td>1515</td>
      <td>909101060</td>
      <td>30</td>
      <td>45.000000</td>
      <td>8248</td>
      <td>3</td>
      <td>3</td>
      <td>1914</td>
      <td>1950</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>67000</td>
    </tr>
    <tr>
      <th>656</th>
      <td>2659</td>
      <td>902305090</td>
      <td>70</td>
      <td>60.000000</td>
      <td>9600</td>
      <td>4</td>
      <td>2</td>
      <td>1900</td>
      <td>1950</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>90</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2006</td>
      <td>87000</td>
    </tr>
    <tr>
      <th>246</th>
      <td>2328</td>
      <td>527190050</td>
      <td>160</td>
      <td>44.000000</td>
      <td>5306</td>
      <td>7</td>
      <td>7</td>
      <td>1987</td>
      <td>1987</td>
      <td>0.000000</td>
      <td>...</td>
      <td>441</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>239000</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>473</td>
      <td>528228455</td>
      <td>120</td>
      <td>43.000000</td>
      <td>3182</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>16.000000</td>
      <td>...</td>
      <td>100</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
      <td>180000</td>
    </tr>
    <tr>
      <th>756</th>
      <td>1015</td>
      <td>527252080</td>
      <td>120</td>
      <td>60.000000</td>
      <td>8118</td>
      <td>9</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>178.000000</td>
      <td>...</td>
      <td>156</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
      <td>334000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1325</td>
      <td>902406090</td>
      <td>50</td>
      <td>81.000000</td>
      <td>12150</td>
      <td>5</td>
      <td>5</td>
      <td>1954</td>
      <td>1954</td>
      <td>335.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>2008</td>
      <td>131500</td>
    </tr>
    <tr>
      <th>475</th>
      <td>2900</td>
      <td>916475100</td>
      <td>20</td>
      <td>85.000000</td>
      <td>14331</td>
      <td>8</td>
      <td>5</td>
      <td>2002</td>
      <td>2002</td>
      <td>630.000000</td>
      <td>...</td>
      <td>270</td>
      <td>78</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2006</td>
      <td>312500</td>
    </tr>
    <tr>
      <th>1431</th>
      <td>2372</td>
      <td>527451460</td>
      <td>160</td>
      <td>21.000000</td>
      <td>1680</td>
      <td>6</td>
      <td>5</td>
      <td>1972</td>
      <td>1972</td>
      <td>504.000000</td>
      <td>...</td>
      <td>352</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>2006</td>
      <td>108000</td>
    </tr>
    <tr>
      <th>1286</th>
      <td>478</td>
      <td>528240030</td>
      <td>60</td>
      <td>62.000000</td>
      <td>7984</td>
      <td>7</td>
      <td>5</td>
      <td>2004</td>
      <td>2005</td>
      <td>200.000000</td>
      <td>...</td>
      <td>120</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2009</td>
      <td>189500</td>
    </tr>
    <tr>
      <th>247</th>
      <td>2583</td>
      <td>535301150</td>
      <td>20</td>
      <td>70.000000</td>
      <td>8092</td>
      <td>6</td>
      <td>8</td>
      <td>1954</td>
      <td>2000</td>
      <td>176.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>156000</td>
    </tr>
    <tr>
      <th>1252</th>
      <td>244</td>
      <td>905478190</td>
      <td>20</td>
      <td>60.000000</td>
      <td>11100</td>
      <td>4</td>
      <td>7</td>
      <td>1946</td>
      <td>2006</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>84900</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1177</td>
      <td>533236070</td>
      <td>160</td>
      <td>24.000000</td>
      <td>2645</td>
      <td>8</td>
      <td>5</td>
      <td>1999</td>
      <td>2000</td>
      <td>456.000000</td>
      <td>...</td>
      <td>169</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>200000</td>
    </tr>
    <tr>
      <th>132</th>
      <td>414</td>
      <td>527454130</td>
      <td>160</td>
      <td>24.000000</td>
      <td>2349</td>
      <td>6</td>
      <td>5</td>
      <td>1977</td>
      <td>1977</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2009</td>
      <td>137900</td>
    </tr>
    <tr>
      <th>1953</th>
      <td>299</td>
      <td>909455030</td>
      <td>120</td>
      <td>35.000000</td>
      <td>3907</td>
      <td>8</td>
      <td>6</td>
      <td>1989</td>
      <td>1989</td>
      <td>0.000000</td>
      <td>...</td>
      <td>141</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>185000</td>
    </tr>
    <tr>
      <th>1609</th>
      <td>1657</td>
      <td>527357020</td>
      <td>20</td>
      <td>128.000000</td>
      <td>13001</td>
      <td>6</td>
      <td>5</td>
      <td>1971</td>
      <td>1971</td>
      <td>176.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>249</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2007</td>
      <td>170000</td>
    </tr>
  </tbody>
</table>
<p>1538 rows × 39 columns</p>
</div>




```python
abs(X_tr_num.corr().sp)
```




    id               0.047547
    pid              0.267597
    mssubclass       0.095592
    lotfrontage      0.318741
    lotarea          0.301374
    overallqual      0.799418
    overallcond      0.114950
    yearbuilt        0.577287
    yearremod/add    0.540100
    masvnrarea       0.519581
    bsmtfinsf1       0.417304
    bsmtfinsf2       0.024065
    bsmtunfsf        0.210097
    totalbsmtsf      0.633981
    1stflrsf         0.620790
    2ndflrsf         0.251470
    lowqualfinsf     0.046227
    grlivarea        0.702253
    bsmtfullbath     0.289372
    bsmthalfbath     0.043568
    fullbath         0.535411
    halfbath         0.293439
    bedroomabvgr     0.145406
    kitchenabvgr     0.129250
    totrmsabvgrd     0.517462
    fireplaces       0.470631
    garageyrblt      0.506175
    garagecars       0.642726
    garagearea       0.638193
    wooddecksf       0.341659
    openporchsf      0.323471
    enclosedporch    0.127035
    3ssnporch        0.070299
    screenporch      0.134954
    poolarea         0.025787
    miscval          0.007560
    mosold           0.055951
    yrsold           0.020305
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
     'garagecars',
     'garagearea',
     'totalbsmtsf',
     '1stflrsf',
     'yearbuilt',
     'yearremod/add',
     'fullbath',
     'masvnrarea',
     'totrmsabvgrd',
     'garageyrblt',
     'fireplaces',
     'bsmtfinsf1',
     'wooddecksf',
     'openporchsf',
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

Let's try a lasso!


```python
l_alphas = np.arange(.001, .15, .0025)
lasso_model = LassoCV(alphas=l_alphas, max_iter=2000, cv=5)
# lasso_model = LassoCV(max_iter=10000, cv=5)

model_1 = lasso_model.fit(X_tr_mod1, y_train)
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


Great, let's score the lasso.


```python
print(model_1.score(X_ts_mod1, y_test))
```

    0.8531253453633401


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



```python
evansubmission1 = pd.DataFrame(data = model_1_final.predict(finaltest_num), columns = ['SalePrice'], index=finaltest['id'])
evansubmission1.to_csv('./evansubmission1.csv')

# submission_poly = pd.concat([test.Id, pd.Series(y_hat_poly[len(train):])], axis=1)
# submission_poly_column_rename = {0 : 'SalePrice'}
# submission_poly = submission_poly.rename(columns=submission_poly_column_rename)
# submission_poly.to_csv("submission_poly", index=False)
```


```python
# lm = linear_model.LinearRegression()

# df['intercept'] = 1

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


![png](/images/EvansProject2Model1of3_files/EvansProject2Model1of3_57_0.png)


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



```python
# plt.figure(figsize=(10,20))

# i=1
# for name in df1.columns:
#     try: 
#         plt.subplot(3, 1, i)
#         sns.regplot(x=df1[name],y=df2['price']); #!!!! you need to use one-dimension object for both X and Y
#         plt.yticks(np.arange(df2['price'].min(), df2['price'].max(), 100000))
#         plt.xlabel(name,fontsize=18)
#         plt.ylabel('price',fontsize=18)
#         i += 1
#     except:
#         pass
# plt.tight_layout()
```


```python
# Joe's way of concatenating and then unconcatenating
# def get_dummied(train, test):
#     list = ['MS SubClass']
#     for i in train.columns:
#         if train[i].dtype == object:
#             list.append(i)
            
#     full_data = pd.concat([train, test], axis=0)
#     full_data = pd.get_dummies(full_data, columns=list, drop_first=True)
    
#     X_dummied = full_data[:len(train)]
#     test_dummied = full_data[len(train):]

#     return X_dummied, test_dummied 

# Xd, testd = get_dummied(X, test)


```


```python
# evansubmission1 = pd.DataFrame(data = model_1.predict(X_ts_mod1), columns = ['SalePrice'], index=y_test['Id'])
# evansubmission1.to_csv('./evansubmission1.csv')
```

Project 2 - Ames Housing Data and Kaggle Challenge
Due Date: May 18, 2018

Welcome to Project 2! It's time to start modeling.

Creating and iteratively refining a regression model
Using Kaggle to practice the modeling process
You are tasked with creating a regression model based on the Ames Housing Dataset. This model will predict the price of a house at sale.

The Ames Housing Dataset is an exceptionally detailed and robust dataset with over 70 columns of different features relating to houses.

Secondly, we are hosting a competition on Kaggle to give you the opportunity to practice the following skills:

Refining models over time
Use of train-test split, cross-validation, and data with unknown values for the target to simulate the modeling process
The use of Kaggle as a place to practice data science
Set-up
Before you begin working on this project, please do the following:

Sign up for an account on Kaggle
IMPORTANT: Click this link (Regression Challenge Sign Up) to join the competition (otherwise you will not be able to make submissions!)
Review the material on the DSI-US-4 Regression Challenge
The Modeling Process
The train dataset has all of the columns that you will need to generate and refine your models. The test dataset has all of those columns except for the target that you are trying to predict in your Regression model.
Generate your regression model using the training data. We expect that within this process, you'll be making use of:
train-test split
cross-validation / grid searching for hyperparameters
strong exploratory data analysis to question correlation and relationship across predictive variables
code that reproducibly and consistently applies feature transformation (such as the preprocessing library)
Predict the values for your target column in the test dataset and submit your predictions to Kaggle to see how your model does against unknown data.
Note: Kaggle expects to see your submissions in a specific format. Check the challenge's page to make sure you are formatting your files correctly!
Submission Checklist
We expect the following to be submitted by end of day on the due date.

Your code for the regression model, including your exploratory data analysis. Add your (well organized!) notebooks to this repository and submit a pull request.
At least one successful prediction submission on DSI-US-4 Regression Challenge -- you should see your name in the "Leaderboard" tab.
Check the Project Feedback + Evaluation section (below) to ensure that you know what will factor into the evaluation of your work.
Project Feedback + Evaluation
For all projects, students will be evaluated on a simple 4 point scale (0-3 inclusive). Instructors will use this rubric when scoring student performance on each of the core project requirements:

Score	Expectations
0	Does not meet expectations. Try again.
1	Approaching expectations. Getting there...
2	Meets expecations. Great job.
3	Surpasses expectations. Brilliant!
For Project 2 the evaluation categories are as follows:

Organization:	Clearly commented, annotated and sectioned Jupyter notebook or Python script. Comments and annotations add clarity, explanation and intent to the work. Notebook is well-structured with title, author and sections. Assumptions are stated and justified.
Presentation: The goal, methodology and results of your work are presented in a clear, concise and thorough manner. The presentation is appropriate for the specified audience, and includes relevant and enlightening visual aides as appropriate.
Data Structures: Python data structures including lists, dictionaries and imported structures (e.g. DataFrames), are created and used correctly. The appropriate data structures are used in context. Data structures are created and accessed using appropriate mechanisms such as comprehensions, slices, filters and copies.
Python Syntax and Control Flow: Python code is written correctly and follows standard style guidelines and best practices. There are no runtime errors. The code is expressive while being reasonably concise.
Modeling: Data is appropriately prepared for modeling. Model choice matches the context of the data and the analysis. Model hyperparameters are optimized. Model evaluation is robust. Model results are extracted and explained either visually, numerically or narratively.
Regression Challenge Submission: Student has made at least one successful submission to the DSI-US-4 Regression Challenge
Your final assessment ("grade" if you will) will be calculated based on a topical rubric. For each category, you will receive a score of 0-3. From the rubric you can see descriptions of each score and what is needed to attain those scores.

Welcome the Kaggle challenge for Project 2! As part of a successful submission for Project 2, we will expect you to make at least one (and hopefully, multiple!) submissions towards this regression challenge.

In this challenge, you will use the well known Ames housing data to create a regression model that predicts the price of houses in Ames, IA. You should feel free to use any and all features that are present in this dataset.

Goal
It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable.

Evaluation
Kaggle leaderboard standings will be determined by root mean squared error (RMSE).


RMSE=∑(ŷ i−yi)2n‾‾‾‾‾‾‾‾‾‾‾‾‾√
Submission File Format
The file should contain a header and have the following format:.

<!-- Begin Base Rules -->
<h3>One account per participant</h3>
<p>You cannot sign up to Kaggle from multiple accounts and therefore you cannot submit from multiple accounts.</p>
<h3>No private sharing outside teams</h3>
<p>
    Privately sharing code or data outside of teams is not permitted.
    It's okay to share code if made available to all participants on the forums.
</p>
<h3>Team Mergers</h3>
<p>Team mergers are not allowed in this competition.</p>

<h3>Team Limits</h3>
<p>There is no maximum team size.</p>
<h3>Submission Limits</h3>
<p>You may submit a maximum of 2 entries per day.</p>
<p>You may select up to 2 final submissions for judging.</p>
<h3>Competition Timeline</h3>
<p>Start Date: <strong>TBA</strong></p>
<p>Merger Deadline: <strong>None</strong></p>
<p>Entry Deadline: <strong>None</strong></p>
<p>End Date: <strong>2/1/2014 12:00 AM UTC</strong></p>
<!-- End Base Rules -->


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

Adam's presentation notes. Take some time to put markdown selss in. Six point thing. Evaluate the model. Summary of answering hte question. You can have two notebooks. Clean code. Presentation. Technical. Intended for other data scientists. 

Presentation: 
-Outline. Question/talk. A prediction is useless if it's jsut a prediciton. I want to make the best prediciton ever, to make a busines strategy or whatever. So you want to make sure you have the best model out there. I'm interesting to see which features were important (Evan, do this one.) What do you find interesting in this question? I.e., your model's accuracy doesn't have to be anything. 
-Talk about Data cleaning. (Outliers? Null Values)
-Talk about feature selection and engineering. 
-Model
-Model Selection
-Answer Q
-Next Steps/Additional Data


# Modeling: The Movie

(Go to the READ.ME of this repository for the entire write-up.)

For modeling, we took the practice of throwing everything at the wall and seeing what worked. We imported many different models, including linear regression, lasso, SGD regressor, bagging regressor, random forrest regressor, SVR, and adaboost regressor, as well as classifiers including logistic regression, random forest classifier, adaboost classifier, k-nearest neighbors classifier, decision tree classifier, and even a neural network. 


```python
import imdb
import re
import pandas as pd
import numpy as np
import ast
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, SGDRegressor
from sklearn.feature_selection import RFE
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib inline
```

We brought in our six dataframes:
1. 1 df 2 = directors and actors weighted, , deleted columns with 1 or fewer terms
2. 2 df 2 = directors and actors weighted, deleted columns with 1 or fewer terms
3. 3 df 2 = directors and actors and writers weighted, deleted columns with 1 or fewer terms
4. 1 df 3 = directors and actors weighted, , deleted columns with 2 or fewer terms
5. 2 df 2 = directors and actors weighted, deleted columns with 2 or fewer terms
6. 3 df 2 = directors and actors and writers weighted, deleted columns with 2 or fewer terms


```python
# Pre-made dataframes with directors weighted

# X_train = pd.read_csv('train_everything_director_weights_df2.csv') # 1
# X_test = pd.read_csv('test_everything_director_weights_df2.csv') # 1
# X_train = pd.read_csv('train_everything_director_actor_weights_df2.csv') # 2
# X_test = pd.read_csv('test_everything_director_actor_weights_df2.csv') # 2 
X_train = pd.read_csv('train_everything_director_actor_writer_weights_df2.csv') # 3
X_test = pd.read_csv('test_everything_director_actor_writer_weights_df2.csv') # 3
# X_train = pd.read_csv('train_everything_director_weights_df3.csv') # 4
# X_test = pd.read_csv('test_everything_director_weights_df3.csv') # 4
# X_train = pd.read_csv('train_everything_director_actor_weights_df3.csv') # 5
# X_test = pd.read_csv('test_everything_director_actor_weights_df3.csv') # 5
# X_train = pd.read_csv('train_everything_director_actor_writer_weights_df3.csv') # 6
# X_test = pd.read_csv('test_everything_director_actor_writer_weights_df3.csv') # 6
```

We then fed the dataframes through the following cell, which gave us three regressor scores, then transformed our y variable for classification (based on median Metacritic score) and fed that through three classifiers. Throughout this process many models were attempted and thrown out. Dataframes were changed and had to be saved again and reloaded. At the end of the day we decided on the following models:

- Regression
    - Bagging Regressor
    - Random Forest Regressor
    - LASSO
- Classification
    - Logistic Regression
    - Bagging Classifier
    - Random Forest Classifier
    
Except for LASSO and logistic regression, there wasn't much rhyme or reason for modeling choices. These just gave us the best relative scores (of the ones we tried), and also didn't take a huge amount of time. Also, the bagging regressor and classifier, which didn't seem to ever give us scores that were as good as the other models, still worked quickly and served as a veritable canary in a coal mine, warning us if something had gone wrong with the models. 


```python
y_train = X_train.Metascore
y_test = X_test.Metascore

X_train.drop(['Metascore'], axis=1, inplace=True)
X_test.drop(['Metascore'], axis=1, inplace=True)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
```


```python
br = BaggingRegressor()
br.fit(X_train, y_train)
# print('br train score: ')
# print(br.score(X_train, y_train))
print('br test score: ')
print(br.score(X_test, y_test))
print()

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
# print('rf train score: ')
# print(rf.score(X_train, y_train))
print('rf test score: ')
print(rf.score(X_test, y_test))
print()

lasso = Lasso(.15)
lasso.fit(X_train, y_train)
# print('rf train score: ')
# print(rf.score(X_train, y_train))
print('lasso test score: ')
print(lasso.score(X_test, y_test))
print()

median = np.median(y_train)

new_y = []
for n in y_train:
    if n > median:
        new_y.append(1)
    else:
        new_y.append(0)
y_train = new_y

new_y = []
for n in y_test:
    if n > median:
        new_y.append(1)
    else:
        new_y.append(0)
y_test = new_y

logreg = LogisticRegression() 
logreg.fit(X_train, y_train)
# print('logreg train score: ')
# print(logreg.score(X_train, y_train))
print('logreg test score: ')
print(logreg.score(X_test, y_test))
print()

br = BaggingClassifier()
br.fit(X_train, y_train)
# print('br train score: ')
# print(br.score(X_train, y_train))
print('br test score: ')
print(br.score(X_test, y_test))
print()

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
# print('rf train score: ')
# print(rf.score(X_train, y_train))
print('rf test score: ')
print(rf.score(X_test, y_test))
print()

# 1 reg 

# br test score: 
# 0.038342711196289625

# rf test score: 
# 0.11832620794676674

# lasso test score: 
# 0.19244316790430385

# 1 class 

# logreg test score: 
# 0.6736577181208053

# br test score: 
# 0.662751677852349

# rf test score: 
# 0.6753355704697986

# 2 reg 

# br test score: 
# 0.006896130293002622

# rf test score: 
# 0.07139091002869702

# lasso test score: 
# 0.1924431679043039

# 2 class

# logreg test score: 
# 0.6736577181208053

# br test score: 
# 0.6375838926174496

# rf test score: 
# 0.6585570469798657

# 3 reg

# br test score: 
# 0.05994540328342234

# rf test score: 
# -0.03186605837286138

# lasso test score: 
# 0.1924431679043039

# 3 class

# logreg test score: 
# 0.6736577181208053

# br test score: 
# 0.6384228187919463

# rf test score: 
# 0.6719798657718121

# 4 reg

# br test score: 
# 0.023266042810753954

# rf test score: 
# 0.07619378931494514

# lasso test score: 
# 0.21460320119560472

# 4 class 

# logreg test score: 
# 0.6854026845637584

# br test score: 
# 0.6434563758389261

# rf test score: 
# 0.6375838926174496

# 5 reg

# br test score: 
# 0.005276011558945859

# rf test score: 
# 0.03497975713168888

# lasso test score: 
# 0.21460320119560497

# 5 class 

# logreg test score: 
# 0.6854026845637584

# br test score: 
# 0.6518456375838926

# rf test score: 
# 0.6610738255033557

# 6 reg

# br test score: 
# 0.03739589734130877

# rf test score: 
# -0.02765041735558893

# lasso test score: 
# 0.21460320119560503

# 6 class

# logreg test score: 
# 0.6854026845637584

# br test score: 
# 0.6593959731543624

# rf test score: 
# 0.6753355704697986
```

    br test score: 
    0.03739589734130877
    
    rf test score: 
    -0.02765041735558893
    
    lasso test score: 
    0.21460320119560503
    
    logreg test score: 
    0.6854026845637584
    
    br test score: 
    0.6593959731543624
    
    rf test score: 
    0.6753355704697986
    



```python
cap_mods = pd.read_csv('capstone_models_1.csv')
ap_mods.columns = ['', '1 df 3', '2 df 3', '3 df 3', '1 df 2', '2 df2 ',
       '3 df 2 ']
cap_mods = cap_mods.set_index('')
cap_mods_class = cap_mods.iloc[3:,:].copy()
cap_mods_reg = cap_mods.iloc[:3,:].copy()
```


```python
sns.set_style("darkgrid",{"xtick.color":"black", "ytick.color":"black"})
plt.figure(figsize=(10,5))
sns.heatmap(cap_mods_reg, annot = True, cmap="Greens")
# plt.tick_params(color='white', labelcolor='white');
```


```python
# sns.set_style("dark",{"xtick.color":"white", "ytick.color":"white"})
plt.figure(figsize=(10,5))
sns.heatmap(cap_mods_class, annot = True, cmap = "Blues")
# plt.tick_params(color='white', labelcolor='white');
```

After analyzing the output from our models, we decided to use the 3 df 2 dataframe, aka, # 3, to tune hyperparameters on. Similarly, we tuned classifers on random forest, logreg, and LASSO, omitting the others for time. Frankly, the differences between performance is largely negligible, but we had might as well take the .02 bump provided by our best models. 


```python
cap_mods
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
      <th>1 df 3</th>
      <th>2 df 3</th>
      <th>3 df 3</th>
      <th>1 df 2</th>
      <th>2 df2</th>
      <th>3 df 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>br reg</td>
      <td>0.038343</td>
      <td>0.006896</td>
      <td>0.059945</td>
      <td>0.023266</td>
      <td>0.005276</td>
      <td>0.037396</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rf reg</td>
      <td>0.118326</td>
      <td>0.071391</td>
      <td>-0.031866</td>
      <td>0.076194</td>
      <td>0.034980</td>
      <td>-0.027650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lasso reg</td>
      <td>0.192443</td>
      <td>0.192443</td>
      <td>0.192443</td>
      <td>0.214603</td>
      <td>0.214603</td>
      <td>0.214603</td>
    </tr>
    <tr>
      <th>3</th>
      <td>logreg class</td>
      <td>0.673658</td>
      <td>0.673658</td>
      <td>0.673658</td>
      <td>0.685403</td>
      <td>0.685403</td>
      <td>0.685403</td>
    </tr>
    <tr>
      <th>4</th>
      <td>br class</td>
      <td>0.662752</td>
      <td>0.637584</td>
      <td>0.638423</td>
      <td>0.643456</td>
      <td>0.651846</td>
      <td>0.659396</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rf class</td>
      <td>0.675336</td>
      <td>0.658557</td>
      <td>0.671980</td>
      <td>0.637584</td>
      <td>0.661074</td>
      <td>0.675336</td>
    </tr>
  </tbody>
</table>
</div>




```python
# y_train = X_train.Metascore
# y_test = X_test.Metascore

# X_train.drop(['Metascore'], axis=1, inplace=True)
# X_test.drop(['Metascore'], axis=1, inplace=True)

# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_test = ss.transform(X_test)

# median = np.median(y_train)

# new_y = []
# for n in y_train:
#     if n > median:
#         new_y.append(1)
#     else:
#         new_y.append(0)
# y_train = new_y

# new_y = []
# for n in y_test:
#     if n > median:
#         new_y.append(1)
#     else:
#         new_y.append(0)
# y_test = new_y

rf_params = {
    'max_depth': [None],
    'n_estimators': [200],
    'max_features': [2, 10],
}

gs = GridSearchCV(rf, param_grid=rf_params)
gs.fit(X_train, y_train)
print(gs.score(X_test, y_test))
print(gs.best_score_)
print(gs.best_params_)qwa

# 0.7030201342281879
# 0.667
# {'max_depth': None, 'max_features': 10, 'n_estimators': 200}

# 0.697986577181208
# 0.6808888888888889
# {'max_depth': 1000, 'max_features': 2, 'n_estimators': 200}
```

    0.697986577181208
    0.6808888888888889
    {'max_depth': 1000, 'max_features': 2, 'n_estimators': 200}



```python
lasso = LassoCV()
lasso.fit(X_train, y_train)
lasso.score(X_test, y_test)
```




    0.21850824761335874




```python
lasso = Lasso()
lasso_params = {
    'alphas': [None, .15],
}

gs = GridSearchCV(lasso, param_grid=lasso_params)
gs.fit(X_train, y_train)
print(gs.score(X_test, y_test))
print(gs.best_score_)
print(gs.best_params_)

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-88-53d7a684e4b4> in <module>()
          5 
          6 gs = GridSearchCV(lasso, param_grid=lasso_params)
    ----> 7 gs.fit(X_train, y_train)
          8 print(gs.score(X_test, y_test))
          9 print(gs.best_score_)


    /anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_search.py in fit(self, X, y, groups, **fit_params)
        637                                   error_score=self.error_score)
        638           for parameters, (train, test) in product(candidate_params,
    --> 639                                                    cv.split(X, y, groups)))
        640 
        641         # if one choose to see train score, "out" will contain train score info


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in __call__(self, iterable)
        777             # was dispatched. In particular this covers the edge
        778             # case of Parallel used with an exhausted iterator.
    --> 779             while self.dispatch_one_batch(iterator):
        780                 self._iterating = True
        781             else:


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in dispatch_one_batch(self, iterator)
        623                 return False
        624             else:
    --> 625                 self._dispatch(tasks)
        626                 return True
        627 


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in _dispatch(self, batch)
        586         dispatch_timestamp = time.time()
        587         cb = BatchCompletionCallBack(dispatch_timestamp, len(batch), self)
    --> 588         job = self._backend.apply_async(batch, callback=cb)
        589         self._jobs.append(job)
        590 


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py in apply_async(self, func, callback)
        109     def apply_async(self, func, callback=None):
        110         """Schedule a func to be run"""
    --> 111         result = ImmediateResult(func)
        112         if callback:
        113             callback(result)


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py in __init__(self, batch)
        330         # Don't delay the application, to avoid keeping the input
        331         # arguments in memory
    --> 332         self.results = batch()
        333 
        334     def get(self):


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in __call__(self)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
        132 
        133     def __len__(self):


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in <listcomp>(.0)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
        132 
        133     def __len__(self):


    /anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, error_score)
        442     train_scores = {}
        443     if parameters is not None:
    --> 444         estimator.set_params(**parameters)
        445 
        446     start_time = time.time()


    /anaconda3/lib/python3.6/site-packages/sklearn/base.py in set_params(self, **params)
        272                                  'Check the list of available parameters '
        273                                  'with `estimator.get_params().keys()`.' %
    --> 274                                  (key, self))
        275 
        276             if delim:


    ValueError: Invalid parameter alphas for estimator Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False). Check the list of available parameters with `estimator.get_params().keys()`.



```python
logreg_params = {
    'penalty': ['l1'],
    'C': [10, 100],
}

gs = GridSearchCV(logreg, param_grid=logreg_params)
gs.fit(X_train, y_train)
print(gs.score(X_test, y_test))
print(gs.best_score_)
print(gs.best_params_)

# 0.6971476510067114
# 0.6945555555555556
# {'C': 10, 'penalty': 'l1'}

# 0.7055369127516778
# 0.6921111111111111
# {'C': 10, 'penalty': 'l1'}
```

    /anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_split.py:605: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of members in any class cannot be less than n_splits=3.
      % (min_groups, self.n_splits)), Warning)
    ERROR:root:Internal Python error in the inspect module.
    Below is the traceback from this internal error.
    
    ERROR:root:Internal Python error in the inspect module.
    Below is the traceback from this internal error.
    


    Traceback (most recent call last):
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 625, in dispatch_one_batch
        self._dispatch(tasks)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 588, in _dispatch
        job = self._backend.apply_async(batch, callback=cb)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py", line 111, in apply_async
        result = ImmediateResult(func)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py", line 332, in __init__
        self.results = batch()
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 131, in __call__
        return [func(*args, **kwargs) for func, args, kwargs in self.items]
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 131, in <listcomp>
        return [func(*args, **kwargs) for func, args, kwargs in self.items]
      File "/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_validation.py", line 448, in _fit_and_score
        X_train, y_train = _safe_split(estimator, X, y, train)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/utils/metaestimators.py", line 200, in _safe_split
        X_subset = safe_indexing(X, indices)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/utils/__init__.py", line 160, in safe_indexing
        return X.take(indices, axis=0)
    KeyboardInterrupt
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2910, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "<ipython-input-87-325d164ab35a>", line 7, in <module>
        gs.fit(X_train, y_train)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_search.py", line 639, in fit
        cv.split(X, y, groups)))
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 779, in __call__
        while self.dispatch_one_batch(iterator):
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 625, in dispatch_one_batch
        self._dispatch(tasks)
    KeyboardInterrupt
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 1828, in showtraceback
        stb = value._render_traceback_()
    AttributeError: 'KeyboardInterrupt' object has no attribute '_render_traceback_'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py", line 1090, in get_records
        return _fixed_getinnerframes(etb, number_of_lines_of_context, tb_offset)
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py", line 311, in wrapped
        return f(*args, **kwargs)
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py", line 345, in _fixed_getinnerframes
        records = fix_frame_records_filenames(inspect.getinnerframes(etb, context))
      File "/anaconda3/lib/python3.6/inspect.py", line 1483, in getinnerframes
        frameinfo = (tb.tb_frame,) + getframeinfo(tb, context)
      File "/anaconda3/lib/python3.6/inspect.py", line 1441, in getframeinfo
        filename = getsourcefile(frame) or getfile(frame)
      File "/anaconda3/lib/python3.6/inspect.py", line 696, in getsourcefile
        if getattr(getmodule(object, filename), '__loader__', None) is not None:
      File "/anaconda3/lib/python3.6/inspect.py", line 732, in getmodule
        for modname, module in list(sys.modules.items()):
    KeyboardInterrupt
    Traceback (most recent call last):
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 625, in dispatch_one_batch
        self._dispatch(tasks)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 588, in _dispatch
        job = self._backend.apply_async(batch, callback=cb)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py", line 111, in apply_async
        result = ImmediateResult(func)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py", line 332, in __init__
        self.results = batch()
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 131, in __call__
        return [func(*args, **kwargs) for func, args, kwargs in self.items]
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 131, in <listcomp>
        return [func(*args, **kwargs) for func, args, kwargs in self.items]
      File "/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_validation.py", line 448, in _fit_and_score
        X_train, y_train = _safe_split(estimator, X, y, train)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/utils/metaestimators.py", line 200, in _safe_split
        X_subset = safe_indexing(X, indices)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/utils/__init__.py", line 160, in safe_indexing
        return X.take(indices, axis=0)
    KeyboardInterrupt
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2910, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "<ipython-input-87-325d164ab35a>", line 7, in <module>
        gs.fit(X_train, y_train)
      File "/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_search.py", line 639, in fit
        cv.split(X, y, groups)))
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 779, in __call__
        while self.dispatch_one_batch(iterator):
      File "/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 625, in dispatch_one_batch
        self._dispatch(tasks)
    KeyboardInterrupt
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 1828, in showtraceback
        stb = value._render_traceback_()
    AttributeError: 'KeyboardInterrupt' object has no attribute '_render_traceback_'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2850, in run_ast_nodes
        if self.run_code(code, result):
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2927, in run_code
        self.showtraceback(running_compiled_code=True)
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 1831, in showtraceback
        value, tb, tb_offset=tb_offset)
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py", line 1371, in structured_traceback
        self, etype, value, tb, tb_offset, number_of_lines_of_context)
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py", line 1279, in structured_traceback
        self, etype, value, tb, tb_offset, number_of_lines_of_context
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py", line 1140, in structured_traceback
        formatted_exceptions += self.prepare_chained_exception_message(evalue.__cause__)
    TypeError: must be str, not list
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 1828, in showtraceback
        stb = value._render_traceback_()
    AttributeError: 'TypeError' object has no attribute '_render_traceback_'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py", line 1090, in get_records
        return _fixed_getinnerframes(etb, number_of_lines_of_context, tb_offset)
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py", line 311, in wrapped
        return f(*args, **kwargs)
      File "/anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py", line 345, in _fixed_getinnerframes
        records = fix_frame_records_filenames(inspect.getinnerframes(etb, context))
      File "/anaconda3/lib/python3.6/inspect.py", line 1483, in getinnerframes
        frameinfo = (tb.tb_frame,) + getframeinfo(tb, context)
      File "/anaconda3/lib/python3.6/inspect.py", line 1441, in getframeinfo
        filename = getsourcefile(frame) or getfile(frame)
      File "/anaconda3/lib/python3.6/inspect.py", line 696, in getsourcefile
        if getattr(getmodule(object, filename), '__loader__', None) is not None:
      File "/anaconda3/lib/python3.6/inspect.py", line 733, in getmodule
        if ismodule(module) and hasattr(module, '__file__'):
      File "/anaconda3/lib/python3.6/inspect.py", line 71, in ismodule
        return isinstance(object, types.ModuleType)
    KeyboardInterrupt



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in dispatch_one_batch(self, iterator)
        624             else:
    --> 625                 self._dispatch(tasks)
        626                 return True


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in _dispatch(self, batch)
        587         cb = BatchCompletionCallBack(dispatch_timestamp, len(batch), self)
    --> 588         job = self._backend.apply_async(batch, callback=cb)
        589         self._jobs.append(job)


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py in apply_async(self, func, callback)
        110         """Schedule a func to be run"""
    --> 111         result = ImmediateResult(func)
        112         if callback:


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py in __init__(self, batch)
        331         # arguments in memory
    --> 332         self.results = batch()
        333 


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in __call__(self)
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
        132 


    /anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in <listcomp>(.0)
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
        132 


    /anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, error_score)
        447 
    --> 448     X_train, y_train = _safe_split(estimator, X, y, train)
        449     X_test, y_test = _safe_split(estimator, X, y, test, train)


    /anaconda3/lib/python3.6/site-packages/sklearn/utils/metaestimators.py in _safe_split(estimator, X, y, indices, train_indices)
        199     else:
    --> 200         X_subset = safe_indexing(X, indices)
        201 


    /anaconda3/lib/python3.6/site-packages/sklearn/utils/__init__.py in safe_indexing(X, indices)
        159             # This is often substantially faster than X[indices]
    --> 160             return X.take(indices, axis=0)
        161         else:


    KeyboardInterrupt: 

    
    During handling of the above exception, another exception occurred:


    
    During handling of the above exception, another exception occurred:


    AttributeError                            Traceback (most recent call last)

    /anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py in showtraceback(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)
       1827                         # in the engines. This should return a list of strings.
    -> 1828                         stb = value._render_traceback_()
       1829                     except Exception:


    AttributeError: 'KeyboardInterrupt' object has no attribute '_render_traceback_'

    
    During handling of the above exception, another exception occurred:


    TypeError                                 Traceback (most recent call last)

    /anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py in run_code(self, code_obj, result)
       2925             if result is not None:
       2926                 result.error_in_exec = sys.exc_info()[1]
    -> 2927             self.showtraceback(running_compiled_code=True)
       2928         else:
       2929             outflag = False


    /anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py in showtraceback(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)
       1829                     except Exception:
       1830                         stb = self.InteractiveTB.structured_traceback(etype,
    -> 1831                                             value, tb, tb_offset=tb_offset)
       1832 
       1833                     self._showtraceback(etype, value, stb)


    /anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py in structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)
       1369         self.tb = tb
       1370         return FormattedTB.structured_traceback(
    -> 1371             self, etype, value, tb, tb_offset, number_of_lines_of_context)
       1372 
       1373 


    /anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py in structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)
       1277             # Verbose modes need a full traceback
       1278             return VerboseTB.structured_traceback(
    -> 1279                 self, etype, value, tb, tb_offset, number_of_lines_of_context
       1280             )
       1281         else:


    /anaconda3/lib/python3.6/site-packages/IPython/core/ultratb.py in structured_traceback(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)
       1138             exception = self.get_parts_of_chained_exception(evalue)
       1139             if exception:
    -> 1140                 formatted_exceptions += self.prepare_chained_exception_message(evalue.__cause__)
       1141                 etype, evalue, etb = exception
       1142             else:


    TypeError: must be str, not list


Our best classifier (logreg) accuracy was 

0.6945555555555556 

using C = 10 with an l1 penalty. 

And our best regression R$^2$ score was 

0.21460320119560503

with an $\alpha$ = .15

There is no reason we shouldn't be able to achieve better than this given more time in the future. 

Future recommendations are numerous. There are many different ways possible to make this score better, the only constraint being time. 

In terms of data collection, there are several other large databases to access, including imdb's itself as well as Metacritic's. It is entirely possible we have all the metacritic scores, but we could always use more. Plus, Metacritic has statistics such as whether the movie is part of a franchise and how well the previous film did. We can, of course, make that data ourselves, but again, time is a factor here.

We would also like access to more of the cast and crew including producers, cinematographers, composers, editors, and more of the cast. After all, the theory underlying this entire endeavous is that people make movies and people are consistent in their product. 

Finally, we could impute null values, especially with things like box office revenue, opening weekend box office revenue, Rotten Tomatoes scores, which could all replace Metacritic scores as the target variable. It would then be a simple mapping from one to the other. There could easily be more Rotten Tomatoes scores than Metacritic.

In terms of feature engineering, there are always more columns to make. We could use polynomial features on our numerical data. We could just use directors and writers. We could run more n-grams on the titles. We could change our min_dfs per column. We could sift down out list of actor weights. We could go back and try to get the actors averages like before. 

Finally, there are more models for us to use. Several will allow us to tune hyperarameters to eek out better scores. There are models that work better with NLP. We can try a neural network for both classification and regression. With can try a passive aggressive classifer. And we'll do all that and we'll predict movie scores and eventually they'll make a movie about us. 

And that's my capstone! Wasn't it great? 

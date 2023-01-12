# 1. [Exploratory Data Analysis](#Exploratory-Data-Analysis-(EDA))

## 1.1. [Check if there is a duplicated row](#Check-if-there-is-a-duplicated-row)
### 1.1.1. [duplicated()](#duplicated())
### 1.1.2. [drop_duplicates()](#drop_duplicates())

## 1.2. [Check if there is a duplicated row](#Check-if-there-is-a-duplicated-row)


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sonuçlarda çıkan warning'leri ignore etmek için;
from warnings import filterwarnings
filterwarnings("ignore")

# dataframe'de kaç satır ve sütun gösterilsin;
#pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#Jupyter notebook satırlarını genişletir
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% ! important; }<style>"))

#örnek veri setimiz;
df = sns.load_dataset('iris')
df.head()
```


<style>.container { width:80% ! important; }<style>





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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



<a id='Exploratory-Data-Analysis-(EDA)'></a>
# 1.Exploratory Data Analysis (EDA)

<a id='Check-if-there-is-a-duplicated-row'></a>
## 1.1. Check if there is a duplicated row

* Spot the duplicated observations in the dataset and discard them
* Duplicated datas dont contribute anything so we dont need them
* **duplicated()** ve **drop_duplicates()** fonksiyonları kullanılır.

<a id='duplicated()'></a>
### 1.1.1. duplicated()

1. Returns True, if any row duplicates.


2. Combination with any() function, shows duplicity at once for a dataframe or selected features.


3. Available to check a column of a dataframe, multiple columns of a dataframe or a whole dataframe with using its **subset=["column_name"]** parameter.


4. This function randomly shows one of the duplicated observations(if any exists). To show the all duplicated rows use; **keep=False**


```python
# 1
df = sns.load_dataset('iris')
df.duplicated()
```




    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    145    False
    146    False
    147    False
    148    False
    149    False
    Length: 150, dtype: bool




```python
# 2
df = sns.load_dataset('iris')
df.duplicated().any()
```




    True




```python
# 3
df = sns.load_dataset('iris')
df.duplicated(subset=["sepal_width", "sepal_length"])
```




    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    145     True
    146     True
    147     True
    148    False
    149     True
    Length: 150, dtype: bool




```python
# 4
df = sns.load_dataset('iris')
df[df.duplicated(keep=False)]
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>142</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
</div>



<a id='drop_duplicates()'></a>
### 1.1.2. drop_duplicates()

1. Discards duplicated rows and leaves one of them in dataframe.


2. **Inplace=True** ; makes permenant this transaction


3. **ignore_index=True** ; Ignores indexes of discarded duplicated rows and sorts index number from 0 again.


```python
df = sns.load_dataset('iris')
df.drop_duplicates(inplace=True, ignore_index=True)
```

## 1.2. Have an initial inspection on the dataset

Check the given proporties of your dataset to have an inspire.

1. Size of your dataset >>> **df.shape**
2. Variable types >>> **df.info()**
3. Descriptive statistics >>> **df.describe()**
4. Get frequency of classes or values in each feature >>> **value_counts()**
5. Get unique classes or values in each feature >>> **unique()**
6. Get how many unique values each feature has >>> **nunique()**

### 1.2.1. df.shape


```python
df = sns.load_dataset('iris')
df.shape
```




    (150, 5)



### 1.2.2. df.info()


```python
df = sns.load_dataset('iris')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   sepal_length  150 non-null    float64
     1   sepal_width   150 non-null    float64
     2   petal_length  150 non-null    float64
     3   petal_width   150 non-null    float64
     4   species       150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB
    

### 1.2.3. df.describe()

* describe() only shows statistics for numerical variable with default usage. Use **include="all"** to get statistics for categorical variables as well.
* It's also available to return statistics for different classes in a feature/variable


```python
df = sns.load_dataset('iris')
df.describe(include="all").T
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
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
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
      <th>sepal_length</th>
      <td>150.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.843333</td>
      <td>0.828066</td>
      <td>4.3</td>
      <td>5.1</td>
      <td>5.8</td>
      <td>6.4</td>
      <td>7.9</td>
    </tr>
    <tr>
      <th>sepal_width</th>
      <td>150.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.057333</td>
      <td>0.435866</td>
      <td>2.0</td>
      <td>2.8</td>
      <td>3.0</td>
      <td>3.3</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>petal_length</th>
      <td>150.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.758</td>
      <td>1.765298</td>
      <td>1.0</td>
      <td>1.6</td>
      <td>4.35</td>
      <td>5.1</td>
      <td>6.9</td>
    </tr>
    <tr>
      <th>petal_width</th>
      <td>150.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.199333</td>
      <td>0.762238</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>1.3</td>
      <td>1.8</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>species</th>
      <td>150</td>
      <td>3</td>
      <td>setosa</td>
      <td>50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = sns.load_dataset('iris')
df.groupby("species")["petal_length"].describe().T
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
      <th>species</th>
      <th>setosa</th>
      <th>versicolor</th>
      <th>virginica</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.462000</td>
      <td>4.260000</td>
      <td>5.552000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.173664</td>
      <td>0.469911</td>
      <td>0.551895</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.400000</td>
      <td>4.000000</td>
      <td>5.100000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.500000</td>
      <td>4.350000</td>
      <td>5.550000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.575000</td>
      <td>4.600000</td>
      <td>5.875000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.900000</td>
      <td>5.100000</td>
      <td>6.900000</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2.4. value_counts()

* Useful parameters of this function;

1. **normalize=** >>> default selection is False, if True then it returns frequency of each classes as percentage
2. **ascending=** >>> default selection is False, if True then it returns frequencies in descending order.
3. **dropna=** >>> default selection is False, if True then it includes frequency of NaN values.


```python
df = sns.load_dataset('iris')
df.species.value_counts()
```




    setosa        50
    versicolor    50
    virginica     50
    Name: species, dtype: int64




```python
df.species.value_counts(normalize=True)
```




    setosa        0.333333
    versicolor    0.333333
    virginica     0.333333
    Name: species, dtype: float64



### 1.2.5. unique()


```python
df = sns.load_dataset('iris')
df.species.unique()
```




    array(['setosa', 'versicolor', 'virginica'], dtype=object)



### 1.2.6. nunique()


```python
df = sns.load_dataset('iris')
df.species.nunique()
```




    3



## 1.4. Split the data to train/val(test) in an appropriate way

1. It's a necessary step to avoid data leakage and overfitting problems.


2. You should better not leak any information from your train set to test set or from test set to train set. Data leakage biases the predictions, that is not desired in Machine Learning.


3. You should decide what proportion that you would like to use for your train and test sets in data splitting. Note that the train set can't be lower than %50. ***Try and observe the performance of different train test proportions while creating a machine learning model***


4. You should also choose a proper data splitting method to create a powerful model. There are 2 options; **"Regular(Normal) splitting"** and **"Stratified splitting"**


5. Regular Splitting; Available to use for both regression and classification problems.


6. Stratified Splitting; Available to use in classification problems, especially if the target feature is imbalanced. **It keeps the percentage of target features same in both train and test sets.**


7. ***Note that; check distributions of all the features after splitting. It's especially may be complicated with categorical features if a value is existing in on of the data group but not existing in the other one.***



*Helpful Soruce;*
* https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/

### 1.4.1. Regular(Normal) Splitting;


```python
# the library and the function that we use;
from sklearn.model_selection import train_test_split

df = sns.load_dataset('iris')
df.head(3)
```


```python
# "species" is our target feature

# I would like to have %60 of the dataset as train set

# Shuffle=True;
# shuffle all the observations while splitting, necessary to provide randomness while splitting

# random_state;
# The function splits the data randomly,
# so it's good to specify an random_state id to get identical splitting results after everytime you have to activate it.

X_train, X_test, y_train, y_test = train_test_split(df.drop(["species"], axis=1), 
                                                    df.species,
                                                    test_size=0.40,
                                                    shuffle=True,
                                                    random_state=22)

# Shuffle özelliğini açtığımız için index'ler karışmış halde olacaktır. Indexleri düzene sokmak için şu kodları yaz;
for i in [X_train, X_test, y_train, y_test]:
    i.reset_index(inplace=True, drop=True)
```

### 1.4.2. Stratified Splitting;

* This time we additionally use the **stratify=** parameter.


```python
# the library and the function that we use;
from sklearn.model_selection import train_test_split

df = sns.load_dataset('iris')
df.head(3)
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
# This time we additionally use the stratify= parameter.
X_train, X_test, y_train, y_test = train_test_split(df.drop(["species"], axis=1), 
                                                    df.species,
                                                    test_size=0.40,
                                                    shuffle=True,
                                                    random_state=22,
                                                    stratify= df.species)

# Shuffle özelliğini açtığımız için index'ler karışmış halde olacaktır. Indexleri düzene sokmak için şu kodları yaz;
for i in [X_train, X_test, y_train, y_test]:
    i.reset_index(inplace=True, drop=True)
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

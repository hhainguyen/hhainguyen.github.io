---
title: "What are the most important patterns in human names?"
layout: post
date: 2018-04-30 13:45
image: https://raw.githubusercontent.com/hhainguyen/indigo/gh-pages/assets/top100-uknames.png
headerImage: true
tag:
- python
- tree-based models
- random forest
- extra trees
- xgboost
- name classification
star: false
category: blog
author: hai
description: In this blog post, we will look at the importance of such syntatic features (sub-words) in name classification. We are going to do this by using some extensions of tree bagging models (*Random Forest* and *Extremely Randomized Trees*) to classify the datasets and then using feature importance of each model to see which parterns are more important patterns in classifying names.
---
## Introduction
In this series of blog posts, I am gonna go through a process of finding syntatic patterns of personal names. This is relevant to my "Names" project, in that I was looking at building a flexible model for name classification tasks. However in these posts we are only interested in finding the syntatic/linguistic patterns of personal names and do not try to build a working model.

### Dataset
We gonna use a dataset consisting Olympics athletes' name and countries from the last 2 Olympics Games (2012 and 2016). To try not to make the posts too length, I have collected and cleaned a bit of the dataset so it is ready to use. 

### Skill sets
* Basic skills in this series of posts include **tree-based models (bagging and boosting)**, **vector-based text modelling** and (a bit of) **neural networks**.
* Most code are written in **Python**; however I will occasionally move to **R** in some tasks due to the excellent R package's `ggplot2`.

## Part 1: What are the most important syntatic patterns in names?

In this post, we are trying to follow human instinct while guessing where a person coming from. 

* When I told people my first name,`Hai`, most of the time they would say I was from China. Not quite.

* If I told them my middle name, *removed for privacy*, then they would have another guess, from Korea or China.

* Finally, when they knew my last name, `Nguyen`, 75% of them made a correct prediction.

So which features in a name that enable us to correctly guess where a person come from, or even their gender, age, etc?

The first and easy answer would be linguistic/syntatic features. Such names or sub-names are somehow indexed in our brain so that every time we see a person with similar patterns/sub-names/names we can derive where they are from, which religion they follow and so on...

In this first part, we will look at the importance of such syntatic features (sub-words) in name classification. We are going to do this by using some extensions of tree bagging models (*Random Forest* and *Extremely Randomized Trees*) to classify the datasets and then using feature importance of each model to see which parterns are more important features in classifying names.



```python
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

```

Firstly we will load the dataset into a Pandas's `DataFrame` and have some initial look. 


```python
names = pd.read_csv('data/fullname_olympics_1216.csv')
names
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fullname</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#jesus $garcia</td>
      <td>ES</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#lam $shin</td>
      <td>KR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#aaron $brown</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#aaron $cook</td>
      <td>MD</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#aaron $gate</td>
      <td>NZ</td>
    </tr>
    <tr>
      <th>5</th>
      <td>#aaron $royle</td>
      <td>AU</td>
    </tr>
    <tr>
      <th>6</th>
      <td>#aaron $russell</td>
      <td>US</td>
    </tr>
    <tr>
      <th>7</th>
      <td>#aaron $younger</td>
      <td>AU</td>
    </tr>
    <tr>
      <th>8</th>
      <td>#aauri #lorena $bokesa</td>
      <td>ES</td>
    </tr>
    <tr>
      <th>9</th>
      <td>#ababel $yeshaneh</td>
      <td>ET</td>
    </tr>
    <tr>
      <th>10</th>
      <td>#abadi $hadis</td>
      <td>ET</td>
    </tr>
    <tr>
      <th>11</th>
      <td>#abbas #abubakar $abbas</td>
      <td>BH</td>
    </tr>
    <tr>
      <th>12</th>
      <td>#abbey $d'agostino</td>
      <td>US</td>
    </tr>
    <tr>
      <th>13</th>
      <td>#abbey $weitzeil</td>
      <td>US</td>
    </tr>
    <tr>
      <th>14</th>
      <td>#abbie $brown</td>
      <td>GB</td>
    </tr>
    <tr>
      <th>15</th>
      <td>#abbos $rakhmonov</td>
      <td>UZ</td>
    </tr>
    <tr>
      <th>16</th>
      <td>#abbubaker $mobara</td>
      <td>ZA</td>
    </tr>
    <tr>
      <th>17</th>
      <td>#abby $erceg</td>
      <td>NZ</td>
    </tr>
    <tr>
      <th>18</th>
      <td>#abd #elhalim #mohamed $abou</td>
      <td>EG</td>
    </tr>
    <tr>
      <th>19</th>
      <td>#abdalaati $iguider</td>
      <td>MA</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22823</th>
      <td>#kai $zou</td>
      <td>CN</td>
    </tr>
    <tr>
      <th>22824</th>
      <td>#shiming $zou</td>
      <td>CN</td>
    </tr>
    <tr>
      <th>22825</th>
      <td>#rizlen $zouak</td>
      <td>MA</td>
    </tr>
    <tr>
      <th>22833</th>
      <td>#steven $zuber</td>
      <td>CH</td>
    </tr>
    <tr>
      <th>22834</th>
      <td>#petra $zublasing</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>22835</th>
      <td>#egor $zubovich</td>
      <td>BY</td>
    </tr>
    <tr>
      <th>22836</th>
      <td>#pietro $zucchetti</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>22837</th>
      <td>#anastasia $zueva</td>
      <td>RU</td>
    </tr>
    <tr>
      <th>22838</th>
      <td>#nenad $zugaj</td>
      <td>HR</td>
    </tr>
    <tr>
      <th>22839</th>
      <td>#neven $zugaj</td>
      <td>HR</td>
    </tr>
    <tr>
      <th>22840</th>
      <td>#nadine $zumkehr</td>
      <td>CH</td>
    </tr>
    <tr>
      <th>22841</th>
      <td>#alejandro $zuniga</td>
      <td>CL</td>
    </tr>
    <tr>
      <th>22842</th>
      <td>#romana $zupan</td>
      <td>HR</td>
    </tr>
    <tr>
      <th>22843</th>
      <td>#kelita $zupancic</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>22844</th>
      <td>#emily $zurrer</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>22845</th>
      <td>#armands $zvirbulis</td>
      <td>LV</td>
    </tr>
    <tr>
      <th>22846</th>
      <td>#vera $zvonareva</td>
      <td>RU</td>
    </tr>
    <tr>
      <th>22847</th>
      <td>#krzysztof #maciej $zwarycz</td>
      <td>PL</td>
    </tr>
    <tr>
      <th>22848</th>
      <td>#daniel $zwickl</td>
      <td>HU</td>
    </tr>
    <tr>
      <th>22849</th>
      <td>#marc $zwiebler</td>
      <td>DE</td>
    </tr>
    <tr>
      <th>22850</th>
      <td>#viktoriya $zyabkina</td>
      <td>KZ</td>
    </tr>
    <tr>
      <th>22851</th>
      <td>#dominik $zycki</td>
      <td>PL</td>
    </tr>
    <tr>
      <th>22852</th>
      <td>#lukasz $zygadlo</td>
      <td>PL</td>
    </tr>
  </tbody>
</table>
<p>22853 rows × 2 columns</p>
</div>



Let's plot the distribution of the names by country.


```python
plt.rcParams['figure.figsize'] = [10, 25]
names['country'].value_counts()[names['country'].value_counts()>=15].plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f98be95f0b8>




![png](https://raw.githubusercontent.com/hhainguyen/indigo/gh-pages/assets/namepatterns_1_output_6_1.png)


There are many US names in this dataset as the US is among the top sports country. However, a big issue is that it is also a immigration country (hosting the most immigrants according to the latest UN report). Let's have a look at this country's names.


```python
names[names.country=='US']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fullname</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>#aaron $russell</td>
      <td>US</td>
    </tr>
    <tr>
      <th>12</th>
      <td>#abbey $d'agostino</td>
      <td>US</td>
    </tr>
    <tr>
      <th>13</th>
      <td>#abbey $weitzeil</td>
      <td>US</td>
    </tr>
    <tr>
      <th>60</th>
      <td>#abigail $johnston</td>
      <td>US</td>
    </tr>
    <tr>
      <th>100</th>
      <td>#adeline #maria $gray</td>
      <td>US</td>
    </tr>
    <tr>
      <th>132</th>
      <td>#adrienne $martelli</td>
      <td>US</td>
    </tr>
    <tr>
      <th>533</th>
      <td>#allison $schmitt</td>
      <td>US</td>
    </tr>
    <tr>
      <th>536</th>
      <td>#allyson $felix</td>
      <td>US</td>
    </tr>
    <tr>
      <th>562</th>
      <td>#alyssa $manley</td>
      <td>US</td>
    </tr>
    <tr>
      <th>563</th>
      <td>#alyssa $naeher</td>
      <td>US</td>
    </tr>
    <tr>
      <th>578</th>
      <td>#amanda $elmore</td>
      <td>US</td>
    </tr>
    <tr>
      <th>583</th>
      <td>#amanda $polk</td>
      <td>US</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22325</th>
      <td>#todd $wells</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22329</th>
      <td>#scott $weltz</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22332</th>
      <td>#lauren $wenger</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22339</th>
      <td>#russell $westbrook</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22345</th>
      <td>#lindsay $whalen</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22346</th>
      <td>#andrew $wheating</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22348</th>
      <td>#mary $whipple</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22356</th>
      <td>#ryan $whiting</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22368</th>
      <td>#jordyn $wieber</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22397</th>
      <td>#deron $williams</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22400</th>
      <td>#james $williams</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22401</th>
      <td>#jesse $williams</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22403</th>
      <td>#lauryn $williams</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22485</th>
      <td>#erica $wu</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22496</th>
      <td>#jacob $wukie</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22614</th>
      <td>#donald $young</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22615</th>
      <td>#isiah $young</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22616</th>
      <td>#jason $young</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22645</th>
      <td>#rachel $yurkovich</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22662</th>
      <td>#mariel $zagunis</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22706</th>
      <td>#sarah $zelenka</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22714</th>
      <td>#julie $zetlin</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22738</th>
      <td>#lily $zhang</td>
      <td>US</td>
    </tr>
    <tr>
      <th>22801</th>
      <td>#kate $ziegler</td>
      <td>US</td>
    </tr>
  </tbody>
</table>
<p>1101 rows × 2 columns</p>
</div>



This US group doesn't look very good. Names like *"karwoski"*, *"naddour"*, *"weltz"*, *"young"*, *"yurkovich"* will bring a lot of noise to our models. So I decided to get rid of it (for now). Sometimes having more data doesn't mean it's gonna be better. We have similar problems in immigration countries like Australia, Canada, and New Zealand so I might need to also filter out these countries.


```python
filtered_names = names[~names.country.isin(['CA','US','NZ','AU'])]
print(len(filtered_names['country'].value_counts()[names['country'].value_counts()>=15]))
filtered_names['country'].value_counts()[names['country'].value_counts()>=15].plot.barh()
```

    125


![png](https://raw.githubusercontent.com/hhainguyen/indigo/gh-pages/assets/namepatterns_1_output_10_2.png)

Note that we've removed some countries with less samples so that the total country count now become 125.

```python
filtered_names = filtered_names[filtered_names['country'].isin(filtered_names['country'].value_counts()[filtered_names['country'].value_counts()>=15].index)]
```

## Extracting N-grams Features
To extract n-grams from names, the easiest way is to use `sklearn.feature_extraction.text.*` packages such as `CountVectorizer`, `HashingVectorizer`, `TfidfVectorizer`... The first two are mainly for count and n-gram features while the last is count features with *TF-IDF* weights applied to it (**TF** is term frequency and **IDF** is the inversed-document frequency weights indicating the importance of a word within a corpus). In this post I used `CountVectorizer` as it is more intuitive and straight-forward to lookup features. Next we create a vectorizer and feed the names dataframe into it. 

To be able to choose the number of ngrams (vocabulary size in memory), let just see how much the vocabulary size changes w.r.t. the size of ngrams. This is quite important as the vocabulary of features in `CountVectorizer` are stored in memory and having too many sparse but not really necessary features can bring more noise to your models and at the same time occupied unnecessary memory. In theory n-ngram with n characters would take up to (approx) 26^n features, but in practice this number is much smaller. Let's have a look at this:


```python
from sklearn.feature_extraction.text import *
data = []
for i in range(1,7):
    ngram_vectorizer = CountVectorizer(lowercase=False, analyzer = 'char_wb', ngram_range = (i,i))
    ngram_vectorizer.fit(names['fullname'])
    data.append([len(ngram_vectorizer.vocabulary_),26**i])
print(data)
plt.semilogy(data)
plt.grid(True)
plt.show() a 
```

    [[35, 26], [791, 676], [8367, 17576], [38506, 456976], [72654, 11881376], [84350, 308915776]]



![png](https://raw.githubusercontent.com/hhainguyen/indigo/gh-pages/assets/namepatterns_1_output_13_1.png)


Let's choose the vocabulary size of 100K as this would cover most of the features with n-ngram ranging from 3 to 6 characters since we are intertested in patterns, not performance... In practice I recommend using much lower vocab size as it will make the model more roburst and save your a lot more time. Note that we use the analyzer `char_wb` to count n-grams of characters and remove unnecessary ngrams in the border of words (with white spaces). I also removed token whose count less than 3 (possibly just typoes).


```python
from sklearn.feature_extraction.text import *
ngram_vectorizer = CountVectorizer(lowercase=False, analyzer = 'char_wb', ngram_range = (3,6),max_features=100000,min_df=3)
ngram_vectorizer.fit(filtered_names['fullname'])
print(len(ngram_vectorizer.get_feature_names()))
ngram_vectorizer.get_feature_names()[:10]
```

    50461





    [' #a',
     ' #aa',
     ' #aar',
     ' #aaro',
     ' #ab',
     ' #abb',
     ' #abba',
     ' #abbo',
     ' #abd',
     ' #abda']



Now we only have around 50K patterns left, which is much better. The next step is to *transform*. Note that in machine learning, building a model is called "fit", and doing something with the built model is called either *transform*, *predict*, or ocassionally *evaluate*. The differences are:
* `fit`: to build a model. This requires a set of input feature values. If there is also labels for such input feature values, you are building a supervised model. Otherwise it is an unsupervised models.
* `transform`: to convert an input item into another output item. Normally this is for data transformation and only a single step of the chain. You can have multiple transformers and formed a pipeline of tasks.
* `predict`: to convert from an input (list of feature values) into an output. This is for predictive models, where you want to predict output for unseen input. This could be the last step of the chain, when you use the transformed/prepared input features to predict a numeric value (regression problem) or a class (classification problem).
    + `predict_proba` is a special case of `predict` where you want not only the results, but also the probability/confidence level of all classes. Say, if the probability of the predicted class is very low you might not want to use them.
* `evaluate`: to evaluate the supervised model using some metrics. This will need a transformed features as `predict` together with a set of ground_truth (labels) to measure the performance of the model. Without this step, you cannot say which supervised model is better than another.

Okay, next let's try to transform the names into vectors of n-grams. Storing a matrix of 20Kx50K would take a lot of memory (number of cells would be 1 billion); however luckily in sklearn this was stored in a sparse format (a special format that helps to save space and memory when you have many duplicated values likes zeroes or constants).


```python
data =  ngram_vectorizer.transform(filtered_names['fullname'])
label = filtered_names['country']
# since xgboost package doesn't accept string labels, we need to encode them into numeric values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le = le.fit(label)
encoded_label = le.transform(label)

```

We need to also transform the labels into number (int) as xgboost classifier doesn't accept the string labels. In `sklearn`, you can use `LabelEncoder` for this task. This is very useful as they can do the inversed mapping by calling `inversed_transform` to convert from the integer into the string labels.

## Building the Models to Find Most Important Patterns

In this section we will build the tree-based models from the `sklearn.ensemble` package and `xgboost` in python. We will also look at the most important patterns in each model and see if they are overlapping each other.

Firstly we gonna try Random Forest.


```python
from sklearn.ensemble import *      
rf = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
rf.fit( data,encoded_label)

```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                oob_score=False, random_state=42, verbose=0, warm_start=False)




```python
rf.feature_importances_
```




    array([1.04790809e-03, 1.41188986e-05, 1.46378045e-06, ...,
           0.00000000e+00, 1.03540043e-06, 1.50053925e-07])




```python
sorted_features_idx = np.argsort(rf.feature_importances_)
top_pattern_size = 500
rf_sorted_feature_names = [ngram_vectorizer.get_feature_names()[i] for i in sorted_features_idx[-top_pattern_size:]]
# get top patterns sorted by importance
top_features = list(zip(rf_sorted_feature_names, sorted_features_idx[-top_pattern_size:], np.sort((100*rf.feature_importances_),axis=0)[-top_pattern_size:]))
# print top patterns after removing 1 character pattern
[(feature_name,idx,importance) for (feature_name,idx,importance) in top_features if len(feature_name.replace('#','').replace('$','').strip())>1][-100:]
```




    [('le ', 33438, 0.04256709267150044),
     (' #mar', 1644, 0.04258213268086082),
     ('eng ', 24779, 0.04262375164158329),
     ('do ', 23107, 0.04282082862440401),
     ('nen ', 36925, 0.042835584379132124),
     ('man', 34973, 0.043349771882060475),
     ('ski', 44557, 0.04355082990677741),
     (' $li', 4854, 0.043928302930578526),
     ('sso', 44941, 0.044402791546340734),
     ('ya ', 49279, 0.04459564301601647),
     (' #van ', 2638, 0.04490113182903145),
     (' #van', 2637, 0.04519745763711816),
     ('yan', 49348, 0.04528871782337996),
     ('hang', 27696, 0.045428160340949056),
     ('$li ', 13506, 0.04591235536916939),
     ('ina', 30171, 0.046019498611105984),
     ('$kim', 13213, 0.04657805945605272),
     ('ta ', 45402, 0.04666833850224667),
     ('ina ', 30172, 0.047264536672328536),
     ('to ', 46269, 0.047383285931620105),
     ('ro ', 42669, 0.04794552706502739),
     ('re ', 41678, 0.04826265617364551),
     ('im ', 30038, 0.048949333368040256),
     (' $ma', 4929, 0.04958944424667826),
     ('ana', 17801, 0.049703771524317),
     ('#mar', 9085, 0.05023404120660694),
     ('eng', 24778, 0.05040909401827986),
     ('kov', 32723, 0.050522958228282316),
     ('ra ', 41085, 0.0509436145115542),
     ('ari', 18746, 0.051004046648833345),
     ('ti ', 46050, 0.05108503637397418),
     ('$kim ', 13214, 0.05170896955625455),
     ('son', 44741, 0.05235705164822516),
     (' $zh', 6592, 0.052605871323785566),
     ('ll ', 34171, 0.052749233066232915),
     ('son ', 44742, 0.052789962538949006),
     ('#an', 6834, 0.05305741800979369),
     ('io ', 30456, 0.053265769242590716),
     (' #an', 133, 0.05404485972156847),
     ('ian', 28916, 0.05456095864274689),
     ('mi ', 35603, 0.05483018552219813),
     ('nov', 37725, 0.05494090341377208),
     ('hi ', 28144, 0.055253552105131845),
     ('ey ', 25997, 0.05592314426062085),
     ('la ', 33004, 0.05683920281740606),
     ('and', 17895, 0.057061272212139046),
     ('wsk', 49187, 0.05789662354906709),
     ('$zh', 15812, 0.05797579982964542),
     ('ong ', 39120, 0.05867314631355296),
     ('$ma', 13598, 0.0587948035051236),
     ('ka ', 31887, 0.059459329387066384),
     ('un ', 47593, 0.05998631478363998),
     ('ou ', 39883, 0.06003970244107707),
     ('sson', 44947, 0.060044796847437173),
     ('el ', 24248, 0.06069505303142678),
     ('vic', 48606, 0.06355939592418759),
     ('mar', 35044, 0.0653457221818258),
     ('es ', 25517, 0.06540978829144324),
     ('os ', 39657, 0.06595570741349535),
     ('ne ', 36853, 0.06603573155762736),
     ('ong', 39119, 0.06667002053532751),
     (' $kim ', 4622, 0.06683974892344077),
     ('ung', 47641, 0.06720736826501691),
     ('aka', 16969, 0.06739991533030858),
     ('#ma', 9008, 0.06802895026722539),
     ('is ', 30745, 0.06913430585474722),
     ('sson ', 44948, 0.0695179850590573),
     ('in ', 30169, 0.07072107115817944),
     ('yan ', 49349, 0.0713067438112722),
     ('sch', 43816, 0.07159314638001407),
     ('ang ', 18027, 0.07203828832901753),
     ('ko ', 32568, 0.07242379777703797),
     ('ova ', 40079, 0.07328658554746752),
     ('tsu', 46621, 0.0745778896769875),
     ('sz ', 45345, 0.07544415960750227),
     ('on ', 39021, 0.07574168242916489),
     (' $kim', 4621, 0.07694494368684282),
     ('ni ', 37231, 0.07794505768104643),
     (' #ma', 1594, 0.07824036305877351),
     ('ie ', 29318, 0.08017624819418206),
     ('ki ', 32366, 0.08267168432832733),
     ('sen ', 43989, 0.08303344850697861),
     ('ia ', 28854, 0.08522454669653055),
     ('vic ', 48607, 0.08951583645721485),
     ('ev ', 25870, 0.09419395024731282),
     ('as ', 19147, 0.09523495854619825),
     ('ang', 18026, 0.100927095328532),
     ('na ', 36311, 0.10576984180437138),
     ('en ', 24662, 0.105837511042199),
     ('ao ', 18387, 0.10610172663972099),
     ('an ', 17793, 0.10845955968255647),
     ('sen', 43988, 0.1095635006092101),
     ('shi', 44195, 0.1186171411838533),
     ('ova', 40078, 0.12612620261085633),
     ('ez ', 26046, 0.13921838947549992),
     ('er ', 25116, 0.16281919553354393),
     ('va ', 48274, 0.1632053337312486),
     ('ic ', 29101, 0.19939663183970113),
     ('ng ', 37022, 0.2004263345017378),
     ('ov ', 40077, 0.20494469651043337)]



Looking good. We would expect something very similar with [*ExtraTrees*](https://link.springer.com/article/10.1007/s10994-006-6226-1).


```python
ee = ExtraTreesClassifier(n_estimators=100,random_state=42,n_jobs=-1,verbose = 1)
ee.fit( data,encoded_label)
sorted_features_idx = np.argsort(ee.feature_importances_)
top_pattern_size = 500
ee_sorted_feature_names = [ngram_vectorizer.get_feature_names()[i] for i in sorted_features_idx[-top_pattern_size:]]
# get top patterns sorted by importance
top_features = list(zip(ee_sorted_feature_names, sorted_features_idx[-top_pattern_size:], np.sort((100*ee.feature_importances_),axis=0)[-top_pattern_size:]))
# print top patterns after removing noise (1 character pattern)
[(feature_name,idx,importance) for (feature_name,idx,importance) in top_features if len(feature_name.replace('#','').replace('$','').strip())>1][-100:]
```

    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    8.6s finished





    [('#mar', 9085, 0.044955145118303765),
     ('ovic ', 40124, 0.045136766771534),
     ('ani', 18071, 0.045246059065985125),
     ('eng ', 24779, 0.045337170609674016),
     ('hi ', 28144, 0.0456506078305743),
     (' #van', 2637, 0.04597501527166446),
     ('mi ', 35603, 0.045996194267933437),
     ('zhang', 50154, 0.046042686446211206),
     ('kim', 32409, 0.04608196107823113),
     ('ey ', 25997, 0.04612203772113337),
     ('do ', 23107, 0.04675759108651651),
     ('ana', 17801, 0.046919908945627115),
     ('ri ', 42085, 0.04712047542168313),
     ('nen ', 36925, 0.04725806366359407),
     ('son', 44741, 0.04738321975890253),
     ('yama', 49336, 0.047471478217776836),
     ('#hy', 8157, 0.04788588748350777),
     ('$zh', 15812, 0.04791668718642465),
     (' $li ', 4855, 0.0479596148743877),
     ('ina ', 30172, 0.04809007976986185),
     ('ar ', 18522, 0.048376799707571076),
     ('ta ', 45402, 0.04887720506193443),
     ('son ', 44742, 0.04916290327250859),
     (' $li', 4854, 0.049191499895848555),
     ('ro ', 42669, 0.049460723678886674),
     ('#an', 6834, 0.04953844376506084),
     ('$li ', 13506, 0.04958076675064566),
     (' #an', 133, 0.04966248613601018),
     ('ina', 30171, 0.04987780036318775),
     ('re ', 41678, 0.050133447120492315),
     (' #xi', 2752, 0.05015254559774518),
     ('io ', 30456, 0.05021444884939952),
     ('nov', 37725, 0.05074997966308795),
     ('un ', 47593, 0.05076489542378371),
     ('ka ', 31887, 0.05122317821094081),
     ('sson ', 44948, 0.051255810864940854),
     ('man', 34973, 0.05169403017113321),
     ('ian', 28916, 0.05273215646315537),
     ('kov', 32723, 0.05279763282691419),
     ('sz ', 45345, 0.05328525618022896),
     ('ti ', 46050, 0.05339648168048596),
     ('ang', 18026, 0.05372515331898624),
     ('ini ', 30318, 0.05467477781416105),
     (' $kim', 4621, 0.05501894665992591),
     ('enko', 24862, 0.0551396048749669),
     ('#van ', 10571, 0.05550271989315009),
     (' $zh', 6592, 0.05581340948870025),
     ('and', 17895, 0.055949956057978235),
     ('ari', 18746, 0.0562675486817043),
     ('tsu', 46621, 0.0569305367089678),
     ('ki ', 32366, 0.05783251580886696),
     ('ll ', 34171, 0.05965239914134089),
     ('ong ', 39120, 0.05966519338249538),
     ('sso', 44941, 0.059909193956821746),
     ('ra ', 41085, 0.059973953122648006),
     ('la ', 33004, 0.060453135603371025),
     ('ovic', 40123, 0.06052187270240952),
     (' $ma', 4929, 0.06074711817546006),
     ('kim ', 32410, 0.061184123596058265),
     ('ne ', 36853, 0.06145662004907431),
     ('$ma', 13598, 0.06254860691967477),
     ('ou ', 39883, 0.06363465057037956),
     ('owsk', 40188, 0.06365706069229628),
     ('$kim', 13213, 0.0637418490404201),
     ('es ', 25517, 0.06443104693898041),
     ('ko ', 32568, 0.0651487864220205),
     ('sch', 43816, 0.06576288439901287),
     ('ong', 39119, 0.06668997205250095),
     ('ni ', 37231, 0.06710424208942982),
     ('os ', 39657, 0.06781863953286224),
     ('ie ', 29318, 0.0704003367163709),
     ('on ', 39021, 0.07085292111275938),
     ('ao ', 18387, 0.07094680285901557),
     ('mar', 35044, 0.07116285006572898),
     ('is ', 30745, 0.07151996030836691),
     ('el ', 24248, 0.07360989316763965),
     ('#ma', 9008, 0.07517917449360398),
     ('yan ', 49349, 0.07528667308031196),
     ('sson', 44947, 0.07623018797490469),
     ('en ', 24662, 0.07722349995756884),
     (' #ma', 1594, 0.07814313932127245),
     ('ev ', 25870, 0.07933254640989479),
     ('sen', 43988, 0.07993051968600741),
     ('shi', 44195, 0.08097183213741163),
     ('in ', 30169, 0.08110750038316603),
     ('ang ', 18027, 0.0869085338311959),
     ('ova ', 40079, 0.08872467813630665),
     ('as ', 19147, 0.08881302372416354),
     ('ia ', 28854, 0.0888160690850886),
     ('ova', 40078, 0.10105395896840833),
     ('ez ', 26046, 0.10765161976804602),
     ('na ', 36311, 0.10796082306967256),
     ('vic ', 48607, 0.10807446223769633),
     ('an ', 17793, 0.11214521989802909),
     ('sen ', 43989, 0.12211872837731705),
     ('va ', 48274, 0.1273201187308604),
     ('ng ', 37022, 0.13157069547920763),
     ('er ', 25116, 0.13579010957546528),
     ('ov ', 40077, 0.1591217190628161),
     ('ic ', 29101, 0.17477829732484954)]




```python
print('#patterns in intersection',len(list(set(rf_sorted_feature_names)&set(ee_sorted_feature_names))))
print('#patterns in rf model only',len(list(set(rf_sorted_feature_names) - set(ee_sorted_feature_names))))
print(pd.DataFrame(list(set(rf_sorted_feature_names) - set(ee_sorted_feature_names))))
print('#patterns in ee model only',len(list(set(ee_sorted_feature_names) - set(rf_sorted_feature_names))))
print(pd.DataFrame(list(set(ee_sorted_feature_names) - set(rf_sorted_feature_names))))
```

    #patterns in intersection 441
    #patterns in rf model only 59
             0
    0      yun
    1      tti
    2      yam
    3     hang
    4      #va
    5     aite
    6      $ko
    7     park
    8       #z
    9      jia
    10     aud
    11    nov 
    12    ska 
    13     rzy
    14     #sh
    15     $sz
    16     #se
    17     nse
    18     mon
    19     min
    20     lva
    21     ira
    22     czy
    23     #ju
    24     ori
    25    $mar
    26     ame
    27     #el
    28   $silv
    29     kip
    30     iel
    31     nna
    32    ille
    33     uli
    34     bou
    35     ska
    36     tsi
    37     gue
    38    med 
    39     rti
    40     uan
    41     nen
    42    iia 
    43     hvi
    44    naka
    45    $sch
    46  $wang 
    47     us 
    48    $sch
    49     #le
    50     erg
    51    eong
    52    #and
    53    yte 
    54    mura
    55     #ca
    56     nko
    57     ist
    58     xia
    #patterns in ee model only 59
             0
    0    ensen
    1      sar
    2    park 
    3    $zhan
    4     ira 
    5   $zhang
    6      ert
    7     suke
    8     yeon
    9     ara 
    10    ler 
    11     #ar
    12    oshi
    13   zhang
    14   mura 
    15    yang
    16     ena
    17     #ar
    18    wang
    19    #jia
    20   ingh 
    21     avi
    22      $u
    23     arr
    24     #el
    25     aro
    26     ga 
    27     ere
    28     szt
    29     cz 
    30    $lee
    31    iya 
    32    umi 
    33    atsu
    34     $ko
    35     iia
    36     $mi
    37     nak
    38     nik
    39     $co
    40     #se
    41     mat
    42    yuki
    43    ilva
    44    $ma 
    45   aite 
    46   $chen
    47     thi
    48     ass
    49    $mar
    50    chi 
    51     ric
    52    #yuk
    53  michal
    54     #qi
    55    sato
    56     ola
    57   $sun 
    58     mir


90% of the top 500 patterns of the first 2 models are overlapping. Now let's try a completely different model, [`xgboost`](https://github.com/dmlc/xgboost) that ensembles the trees in a boosting manner instead of boostrap samples and aggregate them like in *bagging*. Note that in RF and EE models, we didn't limit the max as we try to build complex trees and average them out so that overfitting can be avoid. In xgboost, we use smaller trees (can even with depth of 2 or 3), but many of them to refine the model in an incremental way.


```python
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators = 200, max_depth=8,nthread =22,objective='multi:softmax')
xgb.fit( data,encoded_label)
sorted_features_idx = np.argsort(xgb.feature_importances_)
top_pattern_size = 500
xgb_sorted_feature_names = [ngram_vectorizer.get_feature_names()[i] for i in sorted_features_idx[-top_pattern_size:]]
# get top patterns sorted by importance
top_features = list(zip(xgb_sorted_feature_names, sorted_features_idx[-top_pattern_size:], np.sort((100*xgb.feature_importances_),axis=0)[-top_pattern_size:]))
# print top patterns after removing noise (1 character pattern)
[(feature_name,idx,importance) for (feature_name,idx,importance) in top_features if len(feature_name.replace('#','').replace('$','').strip())>1][-100:]
```




    [('art', 19035, 0.11031855),
     (' #ro', 2179, 0.110973254),
     (' $la', 4774, 0.110973254),
     ('to ', 46269, 0.11130062),
     ('al ', 17161, 0.111627966),
     (' #ra', 2092, 0.111627966),
     (' $ki', 4609, 0.11261003),
     (' $ka', 4521, 0.11293739),
     ('im ', 30038, 0.11326474),
     (' $me', 5034, 0.11326474),
     (' #mar', 1644, 0.11522888),
     ('kov', 32723, 0.115556225),
     (' #mi', 1708, 0.115556225),
     ('son', 44741, 0.11621093),
     ('da ', 22412, 0.11686564),
     (' $be', 3242, 0.117193),
     ('no ', 37633, 0.11915712),
     (' #jo', 1256, 0.12013919),
     ('ine', 30234, 0.12046655),
     ('eli', 24369, 0.12046655),
     ('car', 21411, 0.1207939),
     ('ova ', 40079, 0.1207939),
     ('ale', 17248, 0.12112126),
     ('ni ', 37231, 0.12112126),
     ('ele', 24317, 0.122430675),
     ('ann', 18167, 0.12275802),
     ('chi', 21795, 0.12308539),
     ('ie ', 29318, 0.1279957),
     (' $ha', 4168, 0.12832306),
     ('ova', 40078, 0.12897778),
     ('ali', 17340, 0.12930512),
     ('un ', 47593, 0.1309419),
     ('ani', 18071, 0.13126925),
     ('lin', 33944, 0.13225132),
     ('kha', 32270, 0.13290603),
     ('ah ', 16688, 0.13323338),
     ('ara', 18523, 0.13552487),
     ('ana ', 17802, 0.13552487),
     ('ed ', 23689, 0.13650693),
     ('am ', 17556, 0.13748899),
     ('sha', 44103, 0.1381437),
     ('ya ', 49279, 0.13945313),
     ('io ', 30456, 0.13945313),
     ('son ', 44742, 0.14043519),
     ('ham', 27650, 0.14600022),
     (' $ba', 3180, 0.147637),
     ('cha', 21647, 0.1482917),
     ('ell', 24445, 0.14861906),
     ('ka ', 31887, 0.14927377),
     (' #ka', 1316, 0.15254731),
     ('ina', 30171, 0.15320201),
     ('ran', 41300, 0.16105853),
     (' #al', 73, 0.16400473),
     (' $sa', 5781, 0.16465944),
     ('le ', 33438, 0.16727827),
     (' #sa', 2249, 0.16924241),
     (' $ch', 3492, 0.16924241),
     (' #ch', 410, 0.17382537),
     ('ad ', 16266, 0.17480743),
     ('li ', 33808, 0.1751348),
     ('do ', 23107, 0.1757895),
     ('man', 34973, 0.18069981),
     ('ang', 18026, 0.18266395),
     ('ou ', 39883, 0.18462807),
     (' $ca', 3423, 0.18561015),
     ('ana', 17801, 0.1865922),
     ('ta ', 45402, 0.18953839),
     ('ro ', 42669, 0.1901931),
     ('ri ', 42085, 0.19215724),
     ('han', 27680, 0.19870433),
     ('ev ', 25870, 0.21736354),
     (' $ma', 4929, 0.21801826),
     ('ar ', 18522, 0.22783889),
     ('ic ', 29101, 0.22882098),
     (' $al', 3014, 0.23373128),
     ('ra ', 41085, 0.25337258),
     ('ari', 18746, 0.25730082),
     ('is ', 30745, 0.2618838),
     ('ian', 28916, 0.26908562),
     ('es ', 25517, 0.27432328),
     ('os ', 39657, 0.27890626),
     ('va ', 48274, 0.29101837),
     (' #an', 133, 0.3162247),
     ('la ', 33004, 0.32539064),
     ('as ', 19147, 0.32866415),
     ('and', 17895, 0.34372246),
     ('mar', 35044, 0.34601396),
     (' #ma', 1594, 0.35681665),
     ('ez ', 26046, 0.37940416),
     ('ov ', 40077, 0.40526515),
     ('ne ', 36853, 0.41213962),
     ('in ', 30169, 0.4258885),
     ('en ', 24662, 0.42818004),
     ('el ', 24248, 0.45109484),
     ('ia ', 28854, 0.514929),
     ('er ', 25116, 0.5201667),
     ('on ', 39021, 0.5362071),
     ('ng ', 37022, 0.57188874),
     ('na ', 36311, 0.795472),
     ('an ', 17793, 1.1467237)]




```python
print('#patterns in xgb and rf intersection',len(list(set(rf_sorted_feature_names)&set(xgb_sorted_feature_names))))
print('#patterns in rf model only',len(list(set(rf_sorted_feature_names) - set(xgb_sorted_feature_names))))
print(pd.DataFrame(list(set(rf_sorted_feature_names) - set(xgb_sorted_feature_names))))
print('#patterns in xgb model only',len(list(set(xgb_sorted_feature_names) - set(rf_sorted_feature_names))))
print(pd.DataFrame(list(set(xgb_sorted_feature_names) - set(rf_sorted_feature_names))))
```

    #patterns in xgb and rf intersection 289
    #patterns in rf model only 211
              0
    0       #hy
    1       #li
    2       enk
    3       $ta
    4       #sa
    5       #ji
    6       uk 
    7      sen 
    8      hiro
    9     ovic 
    10     aite
    11      jun
    12     yev 
    13      #hy
    14    $kim 
    15     ovic
    16      $be
    17      iy 
    18    chen 
    19    #van 
    20      #xi
    21      $sz
    22     ung 
    23      eon
    24      mon
    25      orz
    26    $wang
    27      rt 
    28      ini
    29      $li
    ..      ...
    181     #de
    182     ien
    183     tak
    184    ini 
    185     #jo
    186     #ro
    187     eun
    188     $ki
    189     tsi
    190    owsk
    191     rti
    192     #ji
    193     ke 
    194     wa 
    195    ski 
    196    naka
    197   moto 
    198   $lee 
    199    $wan
    200     #da
    201     $st
    202      #x
    203     $pe
    204   eira 
    205    mura
    206    ien 
    207     $el
    208     $ra
    209     ger
    210     $la
    
    [211 rows x 1 columns]
    #patterns in xgb model only 211
              0
    0       ila
    1     #alex
    2       ata
    3       $ar
    4       #be
    5      $al 
    6       $gu
    7       lon
    8       she
    9       nti
    10      $de
    11      rov
    12      lli
    13      $di
    14      mou
    15      dia
    16      aba
    17      ah 
    18      fer
    19      ir 
    20      #el
    21      aro
    22      $so
    23      ga 
    24     ali 
    25      $to
    26      iko
    27      $am
    28     man 
    29      yu 
    ..      ...
    181     err
    182     #ke
    183     #ya
    184     lad
    185     ent
    186     id 
    187     nes
    188     amb
    189     $re
    190     #je
    191     nik
    192     $jo
    193     $se
    194     bra
    195     $co
    196     $tr
    197     hor
    198   $garc
    199     ous
    200     at 
    201     isa
    202     est
    203     #ti
    204     tho
    205     #ga
    206     $bu
    207     ur 
    208     ate
    209     ish
    210     ora
    
    [211 rows x 1 columns]


As we can see, RF and EE models are very similar in terms of feature importance with around 90% of top 500 features overlapped. RF and XGB only have 60% of the top 500 features overlapped. Here are major observations from the top features.
1. **Most important (at least top-10) features are suffixes of a name** (can be either surname or forename): `an `, `ez ` for Hispanic names, `ov ` for Russian names, `vic ` for Balkan, `sen ` for Dannish names, etc.
2. **Prefixes are much more rare, but seem to be important signals too.** `$al`,`$ch`, `$zh`, are common prefixes for Arabic, German, and Chinese surnames.
3. Important patterns in XGB are in average shorter than ones in Random Forest and ExtraTrees. It looks like XGB looks at more dominant patterns and less randomised than RF and EE.
4. **Whole names themselves in top patterns are `#van` and `$kim`**, all of which are popular names in Dutch, Vietnamese and Korea.

## Comparing Tree-based Models

Now we will try to compare the performance of these tree-based models, RandomForest, ExtraTrees and XGBoost. I will run these models on the datasets with k-fold cross validation. The folds were generated in accordance to the label distribution.



```python
import time
from sklearn.model_selection import *
from sklearn.model_selection import GridSearchCV

# create kfold
kfold = StratifiedKFold(n_splits=5, random_state=42)
# for model in [ RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1,verbose = 1,oob_score=True),ExtraTreesClassifier(n_estimators=100,random_state=42,n_jobs=-1,verbose = 1)]:
for model in [ XGBClassifier(n_estimators = 500, max_depth=5,nthread =20,objective='multi:softmax'), RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1,verbose = 1,oob_score=True),ExtraTreesClassifier(n_estimators=100,random_state=42,n_jobs=-1,verbose = 1)]:
    start = time.time()
    scores_logloss = cross_val_score(model, data,encoded_label, cv=kfold,scoring='neg_log_loss')
    scores_acc = cross_val_score(model, data,encoded_label, cv=kfold,scoring='accuracy')
    scores_f1micro = cross_val_score(model, data,encoded_label, cv=kfold,scoring='f1_micro')
    print(model)
    print("Time:%0.2f, Accuracy: %0.2f (+/- %0.2f)" % ((time.time()-start),scores_acc.mean(), scores_acc.std()))
    print("Logloss: %0.2f (+/- %0.2f)" % (scores_logloss.mean(), scores_logloss.std()))
    print("F1-micro: %0.2f (+/- %0.2f)" % (scores_f1micro.mean(), scores_f1micro.std()))
```
    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=5, min_child_weight=1, missing=None, n_estimators=500,
           n_jobs=1, nthread=20, objective='multi:softmax', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1)
    Time:6442.22, Accuracy: 0.56 (+/- 0.02)
    Logloss: -2.10 (+/- 0.10)
    F1-micro: 0.56 (+/- 0.02)


    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                oob_score=True, random_state=42, verbose=1, warm_start=False)
    Time:95.87, Accuracy: 0.60 (+/- 0.02)
    Logloss: -4.45 (+/- 0.33)
    F1-micro: 0.60 (+/- 0.02)


    ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
               max_depth=None, max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
               oob_score=False, random_state=42, verbose=1, warm_start=False)
    Time:135.81, Accuracy: 0.60 (+/- 0.03)
    Logloss: -4.62 (+/- 0.41)
    F1-micro: 0.60 (+/- 0.03)


Note that if you want to measure multiple scoring with cross-validation in `sklearn`, the easiest way is to use `GridSearchCV`. However, to keep things simple in this blog, I used `cross_val_score` multiple times but this wouldn't be recommended in practice.

The results are quite promising, even without any hyperparameter tuning. All models achieved score above 50% accuracy. If you are working on the name-based nationality classification problem then this 50% would be already good for top-1 accuracy, given that we have over 150 countries in the label set. Best *logloss* score is `xgboost` model, it means that the probability of correct answers in this model is higher than ones in other models. Both bagging-based models (Random Forest and ExtraTrees) yielded similar accuracy and F1-micro score of 60%. 

Time consumed for the models are significantly varied. XGboost took lots of time (about 2000s per `cross_val_score`) while Random Forest and Extra Trees tooks less than 50s per cross_val_score. However, this is a trade-off between performance and memory since with Random Forest and Extra Trees, given the number of features and the size of the dataset, increasing the number of trees (estimators) will lead to out-of-memory easily. So if there was a memory-rich machine, I would go with either RF or ExtraTrees. Also, my experience shows that with higher number of trees, ExtraTrees tends to train faster.


## Summary
In this blog post, we've gone through the steps to find the most important patterns for the name-based nationality prediction task. In particular, we have:
1. loaded the dataset and view some descriptive stats,
2. vectorized the full names into a one-hot vector using `sklearn`'s feature extraction,
3. created 3 tree-based models and look at most important patterns. We've also looked at overlapping between top patterns among the models:
    * **Random Forest**
    * **ExtraTrees** (Extremely Randomised Trees)
    * **XGBoost**
4. run a seariespredictive models to classify names into Nationality with cross-validation and multiple multi-class classification metrics: accurary, logloss, and f1-micro.

In the next blog post, we will try to look at which features are important for a specific country, such as the UK, China or Portugal.

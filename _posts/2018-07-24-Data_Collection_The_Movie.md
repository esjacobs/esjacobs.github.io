---
layout: post
title: Data Collection, The Movie
date: 2017-07-24
published: true
categories: projects
tags:
---
(Go to the READ.ME of this repository for the entire write-up.)

The first step in my capstone project to collect a large database of films was to first try and get a list of every movie I could. I figured that Wikipedia would have as many movies as I would need, and if a movie wasn't on wikipedia, it was also unlikely to be one that provided me with any useful information in regards to Metascore. So, I used a Wikipedia Python package. 


```python
import wikipedia
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import ast
```

After several attempts at trying to search for movies by year on Wikipedia, I found out that Wikipedia just has a page for "list of movies," which was great because it was easy to scrape, though slightly frustrating as I had already put in several hours of trying to scrape the movies by year pages. 


```python
movie_list = []
slug_list = ['numbers', 'A','B','C','D','E','F','G','H','I','J–K','L','M','N-O','P','Q–R','S','T','U-W','X–Z']
for title in slug_list:
    df = wikipedia.page(f'List of films:_{title}')
    movie_list.extend(df.links)
```

The entries weren't just clean titles, and there was some section headers in there as well. Also, the movies had "(film)" appended at the end, sometimes with a year, such as "(2009 film)". Because there are many remakes of movies and movies with the same title, getting the year here was important. So I removed all section headers, removed all "(film)"s, and extracted the year in a tuple. 


```python
for index, n in enumerate(movie_list):
    if 'ist of fil' in n:
        movie_list.remove(n)
    if '(film)' in n:
        movie_list[index] = n[0:-7]
    if ' film)' in n:
        movie_list[index] = (n.split(' (')[0],n.split('(')[1][0:4])

movie_list
```

Here I made sure the list contained no doubles. 


```python
movie_list = list(set(movie_list))
```

It was after this painstaking process that I found a huge list of movies on the site movielens (https://grouplens.org/datasets/movielens/latest/). However, instead of deciding to give up my life as a data scientist and moving to the remote woods, I noticed that this list was only updated as of August 2017, so I knew I had more movies from my scrape. Thus, I download the csv then extracted titles and years from it. 


```python
data = pd.read_csv('movies.csv')
```


```python
data.title[2][0:-7]
data.title[2][-5:-1]
```

Then I appended the titles to my previous list of movies, taking out certain strings that I would later find to be problematic. Also, because every movie in the downloaded database had a year attached, I took that to and made a list of tuples. 


```python
for n in data.title:
    movie_list.append((n[0:-7].split(' (')[0].split(', The')[0].replace('&','and'), n[-5:-1]))
```

Lists of tuples were causing problems for me when loading in the data, so I decided to turn this list of tuples and strings to a DataFrame, putting in NaN values when I didn't have a year.


```python
list_of_dicts = []
for title in (movie_list):
    temp_dict = {}
    if type(title) == str:
        temp_dict['title'] = title
        temp_dict['year'] = np.nan
    else:
        temp_dict['title'] = title[0]
        temp_dict['year'] = title[1]
    list_of_dicts.append(temp_dict)
list_of_dicts
```

Saved it as a csv.


```python
pd.DataFrame(list_of_dicts).to_csv('new_movie_list.csv', index=False)
```

Loaded it back in for Amazon Web Services. 


```python
new_movie_list = pd.read_csv('new_movie_list.csv')
```

Finally, I used a website that collects data from Metacritic, Rotten Tomatoes, IMDB, and a couple others and allows you to search through it. The API provided the following information: 

- Actors
- Awards
- BoxOffice
- Country
- DVD
- Director
- Genre
- Language
- Metascore
- Plot
- Poster
- Production
- Rated
- Ratings
- Released
- Response
- Runtime
- Title
- Type
- Website
- Writer
- Year
- imdbID
- imdbRating
- imdbVotes

I didn't take "Awards," "DVD," "Plot," or "imdbVotes," because all of those attributes are things you will never have access to before a movies comes out. I kept the rating values to use as my target variable. 

I also didn't tape "Country," "Language," "Poster," "Reponse," "Type," or "Website," because none of these things gave any valuable information. Perhaps country or language would be somewhat illuminating, and I may take them at a future date.

My main issues with this API were that it restricted actors to the top four billed, had no other crew (Cinematographer? Hello? Composer?) and also didn't have things like opening weekend box office, budget, months in production, whether the movie is part of a franchise, etc. There are other databases with this type of information and I plan to access those in the future. 

Running through the script below took approximately 11 hours and required my patronage of the API at a rate of 1 dollar a month, which I thought was pretty reasonable. 


```python
new_movie_list = pd.read_csv('new_movie_list.csv')
new_movie_successes = []
new_movie_failures = []
new_all_my_movies = []
index=0
for title, year in zip(new_movie_list.title, new_movie_list.year):
    if index % 1000 == 0:
        pd.DataFrame(new_all_my_movies).to_csv(f'new_all_my_movies_{index}.csv')
    try: 
        temp_dict = {}
        if type(year) == str:
            murl = (f'http://www.omdbapi.com/?apikey=eac947e0&t={title}&y={year}&r=json')
        else:
            murl = (f'http://www.omdbapi.com/?apikey=eac947e0&t={title}&r=json')   
        res = requests.get(murl)
        res_json = res.json()
        temp_dict['Title'] = res_json['Title']
        temp_dict['Rated'] = res_json['Rated']
        temp_dict['Released'] = res_json['Released']
        temp_dict['Year'] = res_json['Year']
        temp_dict['Runtime'] = res_json['Runtime'][0:-4]
        temp_dict['Genre'] = res_json['Genre'].split(',')
        temp_dict['Director'] = res_json['Director'].split(',')
        temp_dict['Writer'] = res_json['Writer'].replace(' (additional dialogue)', '')\
            .replace(' (characters)', '').replace(' (screenplay)', '').replace(' (story)', '').split(',')
        temp_dict['Actors'] = res_json['Actors'].split(',')
        temp_dict['Metascore'] = res_json['Metascore']
        temp_dict['RTRating'] = res_json['Ratings']
        temp_dict['imdbRating'] = res_json['imdbRating']
        temp_dict['imdbID'] = res_json['imdbID']
        temp_dict['BoxOffice'] = res_json['BoxOffice']
        temp_dict['Production'] = res_json['Production']
        new_all_my_movies.append(temp_dict)
        new_movie_successes.append(title)
        pd.DataFrame(new_movie_successes).to_csv('new_movie_successes.csv')
        index += 1
    except:
        new_movie_failures.append(title)
        pd.DataFrame(new_movie_failures).to_csv('new_movie_failures.csv')
        index += 1
        pass
pd.DataFrame(new_all_my_movies).to_csv('new_all_my_movies_final.csv')
```

I came back to scrape the award column, but in the end I thought better of using it and it now gathers dust in my repository. 


```python
additional = pd.read_csv('cleaned_movie_df.csv')
```


```python
additional = pd.read_csv('cleaned_movie_df.csv')
meta_award_add = []
index=0
for imdbid in additional.imdbID:
    if index % 1000 == 0:
        pd.DataFrame(meta_award_add).to_csv(f'meta_award_add{index}.csv')
    try: 
        temp_dict = {}
        murl = (f'http://www.omdbapi.com/?apikey=eac947e0&i={imdbid}&r=json')   
        res = requests.get(murl)
        res_json = res.json()
        temp_dict['imdbID'] = imdbid
        temp_dict['Awards'] = res_json['Awards']
        meta_award_add.append(temp_dict)
        index+=1
    except:
        index+=1
        pass
pd.DataFrame(meta_award_add).to_csv('meta_award_add_final.csv')
```

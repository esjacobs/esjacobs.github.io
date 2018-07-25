---
layout: post
title: Capstone, The Movie
date: 2017-07-24
published: true
categories: projects
tags:
---
# CAPSTONE: THE MOVIE

## By Evan Jacobs

This and the preceeding three blog posts are the capstone project I did for General Assembly. You can find the GitHub [repository](https://github.com/esjacobs/Predicting-Metacritic-Scores) here. Below are two presentations in one. One is technical, the other is non technical. Please note that every headline and slide title, with the exception of one or two, is a quote from a movie.

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQk0fMzdBGgnYFHsuQFAkUNdcB7ws7BYQBX34QOM4P6KbctCd3pkjaoUUWDuU-ENRoFOGEeRNt1F71a/embed?start=true&loop=true&delayms=5000" frameborder="0" width="720" height="434" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

# No Fate But What We Make

There is an ongoing, pernicious problem in the world: people have to wait for a movie to come out to know how good it is. 

This is completely unacceptable. 

Sure, we can wait for reviews to come in, or, better yet, wait for aggregates of reviews to come in on sites such as Metacritic or Rotten Tomatoes, but wouldn't it be great to know that the upcoming superhero movie will disappoint you before you waste time getting excited for the trailers?

Movies are not black boxes (except for The Black Box (2005) and Black Box (2012)). They're produced by people who have produced other movies, written by people who have written other movies, performed in by people who have performed in other movies, and directed by people who have directed other movies. Given that people who work in movies tend to be consistent in their quality of output (see: Michael Bay), we should be able to predict qualities of a movie with only the knowledge of how past movies have performed. 

It's easy enough to say that the new Jared Leto Joker movie will be awful. It will be. Every DC movie has been awful except for one, and Suicide Squad, the only other movie where Jared Leto played the Joker, was truly a crime against humanity. It's easy enough to say that, but can we teach a computer to say it? 

# Some Fellas Collect Stamps

The first step in my capstone project to collect a large database of films was to first try and get a list of every movie I could. I figured that Wikipedia would have as many movies as I would need, and if a movie wasn't on wikipedia, it was also unlikely to be one that provided me with any useful information in regards to Metascore. So, I used a Wikipedia Python package. 

It was after this painstaking process that I found a huge list of movies on the site movielens (https://grouplens.org/datasets/movielens/latest/). However, instead of deciding to give up my life as a data scientist and moving to the woods, I noticed that this list was only updated as of August 2017, so I knew I had more movies from my scrape. Thus, I download the .csv then extracted titles and years from it. 

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

I didn't take "Awards," "DVD," "Plot," or "imdbVotes," because all of those attributes are things you will never have access to before a movie comes out. I kept the rating values to use as my target variable.

I also didn't tape "Country," "Language," "Poster," "Response," "Type," or "Website," because none of these things gave any valuable information. Perhaps country or language would be somewhat illuminating, and I may take them at a future date.

My main issues with this API were that it restricted actors to the top four billed, had no other crew (Cinematographer? Hello? Composer?) and also didn't have things like opening weekend box office, budget, months in production, whether the movie is part of a franchise, etc. There are other databases with this type of information and I plan to access those in the future.

# All Clean. All Clean.

We collected quite a bit of data: 43837 separate movies. The actual cleaning of the data was as tedious and dry as the following paragraphs.

We wrote a bunch of functions to clean the data, even a few we didn't need. This functions did things like turn data to floats or create new columns of data. Because of the size of the database, we didn't spend much time imputing missing values. Because we were targeting Metascore, we went ahead and dropped all movies that didn't have an associated metascore, giving us a new dataframe with 10192 rows. We also got rid of the other columns that weren't going into this project.

After running a few models that performed poorly, we did a little feature engineering. We sought to weight the people involved in a movie by aggregating over their metascores. To do this, we created three dataframes where we isolated the directors, actors, and writers.

We pulled out our list of actors by using a count vectorizer on our features to get lists of columns and aggregate over those lists. We found every director, actor, and writer's mean Metascore.

Finally, I saved a few versions of the dataframe. One with directors weighted, one with directors and actors weighted, and one with directors, actors, and writers weighted. Then for each of those, I saved a version that was count vectorized only if a term appeared twice or if a term appeared three times. In total, six dataframes to see which models the best.

# Please Excuse the Crudity of This Model

For modeling, we took the practice of throwing everything at the wall and seeing what worked. We imported many different models, including linear regression, lasso, SGD regressor, bagging regressor, random forest regressor, SVR, and adaboost regressor, as well as classifiers including logistic regression, random forest classifier, adaboost classifier, k-nearest neighbors classifier, decision tree classifier, and even a neural network.

We then fed the dataframes through the following cell, which gave us three regressor scores, then transformed our y variable for classification (based on median Metacritic score) and fed that through three classifiers. Throughout this process, many models were attempted and thrown out. Dataframes were changed and had to be saved again and reloaded. At the end of the day we decided on the following models:

- Regression
    - Bagging Regressor
    - Random Forest Regressor
    - LASSO
- Classification
    - Logistic Regression
    - Bagging Classifier
    - Random Forest Classifier
    
Except for LASSO and logistic regression, there wasn't much rhyme or reason for modeling choices. These just gave us the best relative scores (of the ones we tried), and also didn't take a huge amount of time. Also, the bagging regressor and classifier, which didn't seem to ever give us scores that were as good as the other models, still worked quickly and served as a veritable canary in a coal mine, warning us if something had gone wrong with the models. 

Our best classifier (logreg) accuracy was 

.6945555555555556 

using C = 10 with an l1 penalty. 

And our best regression R $^2$ score was 

0.21850824761335874

with an alpha = .15

There is no reason we shouldn't be able to achieve better than this given more time in the future. 

# Next Time It'll Be Flawless

Future recommendations are numerous. There are many different ways possible to make this score better, the only constraint being time.

In terms of data collection, there are several other large databases to access, including imdb's itself as well as Metacritic's. It is entirely possible we have all the metacritic scores, but we could always use more. Plus, Metacritic has statistics such as whether the movie is part of a franchise and how well the previous film did. We can, of course, make that data ourselves, but again, time is a factor here.

We would also like access to more of the cast and crew including producers, cinematographers, composers, editors, and more of the cast. After all, the theory underlying this entire endeavor is that people make movies and people are consistent in their product.

Finally, we could impute null values, especially with things like box office revenue, opening weekend box office revenue, Rotten Tomatoes scores, which could all replace Metacritic scores as the target variable. It would then be a simple mapping from one to the other. There could easily be more Rotten Tomatoes scores than Metacritic.

In terms of feature engineering, there are always more columns to make. We could use polynomial features on our numerical data. We could just use directors and writers. We could run more n-grams on the titles. We could change our min_dfs per column. We could sift down out list of actor weights. We could go back and try to get the actors averages like before.

Finally, there are more models for us to use. Several will allow us to tune hyperparameters to eek out better scores. There are models that work better with NLP. We can try a neural network for both classification and regression. With can try a passive aggressive classifier. And we'll do all that and we'll predict movie scores and eventually, they'll make a movie about us.

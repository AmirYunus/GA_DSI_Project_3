
<h1>TECHNICAL REPORT - FAKE NEWS ANALYSIS<span class="tocSkip"></span></h1>

Author: Amir Yunus<br>
GitHub: https://github.com/AmirYunus/GA_DSI_Project_3
***

# Preface

## Problem Statement

Suppose that a government agency wants to tackle the rise of fake news and scam in the community. We have proposed to scrape reddit posts to help build our model. The agency aims the following:

* At least a 95% accuracy for the training set
* At least a 95% accuracy for a separate testing set, unseen by the model
* Test reddit model on news and articles from external websites:
    * What is the accuracy?
    * What are areas of improvement?

## Executive Summary

For this project, we will consider the 2 classes: Fake News and Real News.

There are many definitions to what may constitute a fake news. For the purposes of our analysis, we will consider the following sub-reddits as fake news:
* `r/conspiracy` - A subreddit with posts on conspiracy theories
* `r/Alternativefacts` - A subreddit with posts of 'alternative facts' as coined by U.S. Counselor to the President Kellyanne Conway
* `r/scambaiting` - A subreddit with posts related to scams and scambaiting
* `r/satire` - A subreddit with posts of parody and satire of current events

As we have taken fake news from four sources, we will also consider 4 factual sources to balance our training data.
* `r/worldnews` - A subreddit with news around the world, no later than 1 week old
* `r/politics` - A subreddit with politcal news, but tend towards American politics
* `r/business` - A subreddit with financial, economy and business trends
* `r/technology` - A subreddit with gaming, data and tecnological advancements

As reddit is limiting scrapes to only 1,000 posts per time, we will be continuously scraping and append new results to the old DataFrame. To date, more than 116,000 posts have been scrapped. we will only select 100,000 posts, balanced equally, for our training model.

After scraping, we will look at the most common words for each category - fake and real news. After which, we will run a few models to consider if raw, pre-processed or lemmatization will yield the highest accuracy. Once we can determine the type of data, we will run a grid search on the parameters that will give the best results.

The final model is then fitted with a user-input content. User may input a real or fake news, title or body content of their liking, and run against the model. Finally, the model is tested with a new dataset and we will determine it's accuracy.

## Data Dictionary

| Feature 	| Type 	| Dataset 	| Description 	|
|------------------	|--------	|--------------------------------------------------------	|-------------------------------------	|
| df_fake.content 	| string 	| r/conspiracy <br>r/Alternativefacts <br>r/scambaiting <br>r/satire 	| Fake news scraped from reddit 	|
| df_news.content	| string 	| r/worldnews <br>r/politics <br>r/business <br>r/technology 	| Real news scraped from reddit 	|
| df_onion.content	| string 	| r/nottheonion 	| Real news that appears fake scraped from reddit 	|
| is_fake 	| int 	| (All above) 	| 1: source is fake<br>0: source is real 	|
| pred_fake 	| int 	| model 	| 1: source is predicted to be fake<br>0: source is predicted to be real 	|

# Data Visualisation and Investigation

## Common Words Related to Fake News

![Top Common Words Related to Fake News](/images/fake_wordcloud.png)

## Common Words Related to Real News

![Top Common Words Related to Real News](/images/news_wordcloud.png)

## Common Words Appearing in Both Real and Fake News

![Common Words Appearing in Both Real and Fake News](/images/wordcloud.png)

# Data Analysis

## Modelling Using Raw Data

|  	| Actual Fake 	| Actual Real 	| 0.5<br>prevalence 	| 0.91<br>accuracy 	|
|----------------	|------------------	|--------------------	|---------------------------	|--------------------------	|
| Predicted Fake 	| 11,443 	| 1,141<br>(Type I) 	| 0.90<br>precision 	| 0.10<br>false discovery 	|
| Predicted Real 	| 1,057<br>(Type II) 	| 11,359 	| 0.09<br>false omission 	| 0.91<br>negative predictive 	|
|  	| 0.91<br>sensitivity 	| 0.10<br>fall out rate 	| 10.02<br>positive likelihood 	| 107.77<br>diagnostic odds 	|
|  	| 0.09<br>miss rate 	| 0.90<br>specificity 	| 0.09<br>negative likelihood 	| 0.91<br>F1 score 	|

## Modelling Using Processed Data Without Lemmatization

|  	| Actual Fake 	| Actual Real 	| 0.5<br>prevalence 	| 0.90<br>accuracy 	|
|----------------	|------------------	|--------------------	|---------------------------	|--------------------------	|
| Predicted Fake 	| 11,418 	| 1,184<br>(Type I) 	| 0.90<br>precision 	| 0.10<br>false discovery 	|
| Predicted Real 	| 1,082<br>(Type II) 	| 11,316 	| 0.09<br>false omission 	| 0.91<br>negative predictive 	|
|  	| 0.91<br>sensitivity 	| 0.10<br>fall out rate 	| 9.64<br>positive likelihood 	| 100.85<br>diagnostic odds 	|
|  	| 0.09<br>miss rate 	| 0.90<br>specificity 	| 0.09<br>negative likelihood 	| 0.90<br>F1 score 	|

## Modelling Using Processed Data With Lemmatization

|  	| Actual Fake 	| Actual Real 	| 0.5<br>prevalence 	| 0.90<br>accuracy 	|
|----------------	|------------------	|--------------------	|---------------------------	|--------------------------	|
| Predicted Fake 	| 11,274 	| 1,269<br>(Type I) 	| 0.89<br>precision 	| 0.11<br>false discovery 	|
| Predicted Real 	| 1,226<br>(Type II) 	| 11,231 	| 0.10<br>false omission 	| 0.90<br>negative predictive 	|
|  	| 0.90<br>sensitivity 	| 0.11<br>fall out rate 	| 8.88<br>positive likelihood 	| 81.34<br>diagnostic odds 	|
|  	| 0.10<br>miss rate 	| 0.89<br>specificity 	| 0.10<br>negative likelihood 	| 0.90<br>F1 score 	|

## Final Model - TfidfVectorizer with MultinomialNB

|  	| Actual Fake 	| Actual Real 	| 0.5<br>prevalence 	| 0.99<br>accuracy 	|
|----------------	|------------------	|--------------------	|---------------------------	|--------------------------	|
| Predicted Fake 	| 12,401 	| 156<br>(Type I) 	| 0.98<br>precision 	| 0.02<br>false discovery 	|
| Predicted Real 	| 99<br>(Type II) 	| 12,344 	| 0.01<br>false omission 	| 0.99<br>negative predictive 	|
|  	| 0.99<br>sensitivity 	| 0.02<br>fall out rate 	| 79.49<br>positive likelihood 	| 9911.80<br>diagnostic odds 	|
|  	| 0.01<br>miss rate 	| 0.98<br>specificity 	| 0.01<br>negative likelihood 	| 0.99<br>F1 score 	|

# User Testing

## True Negative

* US troops cross into Iraq as part of withdrawal from Syria
* Petrol bombs and tear gas scar Hong Kong streets as police protesters clash
* Indonesia's Widodo faces test on reform credentials in second term
* Boris Johnson ‘has the numbers’ to win Brexit vote TODAY but ‘poor man’s Cromwell’ Speaker Bercow may block it
* Cristiano Ronaldo’s DNA matched evidence in case of rape accuser Kathryn Mayorga and he told lawyer she said ‘stop’

## False Positive (Type I Error)

* None

## False Negative (Type II Error)

* Following a ban on face masks protesters in Hong Kong use wearable face projectors that trick the facial recognition system used by the government
* Rep. Ilhan Omar protested outside Trump’s Oct. 10 campaign rally in Minneapolis

## True Positive

* Photo shows CIA Director Gina Haspel, a Trump appointee, giving a thumbs-up sign next to the body of a tortured Iraqi man in Abu Ghraib prison
* Photo shows U.S. soldiers on the ground in Syria “crying and visibly shaken saying they could stop this in 10 minutes but Trump won’t let them
* So-called “climate change” is mostly driven by factors unrelated to human activity NASA scientists say

# Testing Against r/nottheonion

`r/nottheonion` is a subreddit that posts news that it is ridiculous that people thought it was fake. It is not surprising if our model will predict more false positive that usual as the wordings will tend towards being fake.

Surprisingly, our model did well by accurately identifying 96.99% of posts as real as compared to the 0.03% false positives (Type I error). This could be due to the large training size we provided into our model.

# Conclusion

We managed to get a model with an accuracy of 98.98%. Even after testing the model with new data from another subreddit, the accuracy is 96.99%.

Further testing using articles outside of reddit generated 80% accuracy. However, we only used 10 randomly picked articles. Given the time, we can collect more data and test our model with external websites.

## Areas for Improvement

* Increase sample size by scraping more topics, more content
* Include user comments when modelling
* Resolve GridSearchCV issue where it crashes due to multi-processing
* Reduce Type II Error for external website testing. Type I error is more favourable.

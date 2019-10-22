import requests
import time
import warnings
import matplotlib
import nltk

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import regex as re
import numpy as np

from IPython.display import display, Markdown
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from csv import reader
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec


class project_3:

    # Colour scheme and style selected
    theme = ['#1F306E', '#553772', '#8F3B76', '#C7417B', '#F5487F']
    colors_palette = sns.palplot(sns.color_palette(theme))
    plt.style.use('seaborn')
    sns.set(style="white", color_codes=True)
    sns.set_palette(colors_palette)

    # Prevent warnings from distracting the reader
    warnings.filterwarnings('ignore')

    # Forces Matplotlib to use high-quality images
    ip = get_ipython()
    ibe = ip.configurables[-1]
    ibe.figure_formats = {'pdf', 'png'}

    matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)
    pd.set_option('display.max_colwidth', -1)
    pd.options.display.max_rows = 999

    nltk.download('wordnet')

    # Funtion to display green box with success
    def success(n):
        return display(
            Markdown(
                f'<div class="alert alert-block alert-success">\
                <b>SUCCESS: </b>{n}</div>'
            )
        )

    # Funtion to display red box with warning
    def warning(n):
        return display(Markdown(f'<div class="alert alert-block alert-danger"><b>WARNING: </b>{n}</div>'))

    # Funtion to display yellow box with check
    def check(n):
        return display(Markdown(f'<div class="alert alert-block alert-warning"><b>CHECK: </b>{n}</div>'))

    # Funtion to display blue box with note
    def note(n):
        return display(Markdown(f'<div class="alert alert-block alert-info"><b>NOTE: </b>{n}</div>'))

    # Function to scrape fake posts
    def fake(topic_1, topic_2, topic_3, topic_4, check):
        # Set reddit topics as a list of topics
        topics = [topic_1, topic_2, topic_3, topic_4]
        posts = []  # Set posts as empty list
        if check == True:  # if True, perform scrape
            df_old = pd.read_csv(f'../data/fake.csv')  # Import old DataFrame
            # Display note old DataFrame loaded
            project_3.note(f'Old DataFrame loaded')
            for i in topics:
                after = None  # Set after = None, to scrape first page
                url = "https://www.reddit.com/r/"+i + \
                    "/hot.json?limit=100"  # Set url based on reddit topic
                # Display note to inform reader that scraping is in process
                project_3.note(f'Scraping {url}')
                # assign list of posts from scrape_data into scrape variable
                scrape = project_3.scrape_data(url, after)
                for j in range(len(scrape)):  # Loop based on length of scrape variable
                    # Append only the data of each post in scrape variable into posts
                    posts.append(scrape[j]['data'])

                after = None  # Set after = None, to scrape first page
                url = "https://www.reddit.com/r/"+i + \
                    "/new.json?limit=100"  # Set url based on new reddit topic
                # Display note to inform reader that scraping is in process
                project_3.note(f'Scraping {url}')
                # assign list of posts from scrape_data into scrape variable
                scrape = project_3.scrape_data(url, after)
                for j in range(len(scrape)):  # Loop based on length of scrape variable
                    # Append only the data of each post in scrape variable into posts
                    posts.append(scrape[j]['data'])

                after = None  # Set after = None, to scrape first page
                url = "https://www.reddit.com/r/"+i + \
                    "/controversial.json?limit=100&t=all"  # Set url based on controversial reddit topic
                # Display note to inform reader that scraping is in process
                project_3.note(f'Scraping {url}')
                # assign list of posts from scrape_data into scrape variable
                scrape = project_3.scrape_data(url, after)
                for j in range(len(scrape)):  # Loop based on length of scrape variable
                    # Append only the data of each post in scrape variable into posts
                    posts.append(scrape[j]['data'])

                after = None  # Set after = None, to scrape first page
                url = "https://www.reddit.com/r/"+i + \
                    "/top.json?limit=100&t=all"  # Set url based on top reddit topic
                # Display note to inform reader that scraping is in process
                project_3.note(f'Scraping {url}')
                # assign list of posts from scrape_data into scrape variable
                scrape = project_3.scrape_data(url, after)
                for j in range(len(scrape)):  # Loop based on length of scrape variable
                    # Append only the data of each post in scrape variable into posts
                    posts.append(scrape[j]['data'])

                after = None  # Set after = None, to scrape first page
                url = "https://www.reddit.com/r/"+i + \
                    "/rising.json?limit=100"  # Set url based on rising reddit topic
                # Display note to inform reader that scraping is in process
                project_3.note(f'Scraping {url}')
                # assign list of posts from scrape_data into scrape variable
                scrape = project_3.scrape_data(url, after)
                for j in range(len(scrape)):  # Loop based on length of scrape variable
                    # Append only the data of each post in scrape variable into posts
                    posts.append(scrape[j]['data'])
            df = pd.DataFrame(posts)  # Transform posts as DataFrame
            # Merge titles and selftext as content
            df['content'] = df.title + " " + df.selftext
            # Drop duplicate posts with the same content
            df.drop_duplicates(subset='content', inplace=True)
            # Save DataFrame into csv to import without scraping in the future
            df = df_old.append(df)  # Append new DataFrame to old DataFrame
            # Display note when appended
            project_3.note(f'New DataFrame appended')
            df.to_csv(f'../data/fake.csv')
            # Display success at end of function
            project_3.success(f'All topics scraped')
        return

    # Function to scrape news posts
    def news(topic_1, topic_2, topic_3, topic_4, check):
        # Set reddit topics as a list of topics
        topics = [topic_1, topic_2, topic_3, topic_4]
        posts = []  # Set posts as empty list
        if check == True:  # if True, perform scrape
            df_old = pd.read_csv(f'../data/news.csv')  # Import old DataFrame
            # Display note old DataFrame loaded
            project_3.note(f'Old DataFrame loaded')
            for i in topics:
                after = None  # Set after = None, to scrape first page
                url = "https://www.reddit.com/r/"+i + \
                    "/hot.json?limit=100"  # Set url based on reddit topic
                # Display note to inform reader that scraping is in process
                project_3.note(f'Scraping {url}')
                # assign list of posts from scrape_data into scrape variable
                scrape = project_3.scrape_data(url, after)
                for j in range(len(scrape)):  # Loop based on length of scrape variable
                    # Append only the data of each post in scrape variable into posts
                    posts.append(scrape[j]['data'])

                after = None  # Set after = None, to scrape first page
                url = "https://www.reddit.com/r/"+i + \
                    "/new.json?limit=100"  # Set url based on new reddit topic
                # Display note to inform reader that scraping is in process
                project_3.note(f'Scraping {url}')
                # assign list of posts from scrape_data into scrape variable
                scrape = project_3.scrape_data(url, after)
                for j in range(len(scrape)):  # Loop based on length of scrape variable
                    # Append only the data of each post in scrape variable into posts
                    posts.append(scrape[j]['data'])

                after = None  # Set after = None, to scrape first page
                url = "https://www.reddit.com/r/"+i + \
                    "/controversial.json?limit=100&t=all"  # Set url based on controversial reddit topic
                # Display note to inform reader that scraping is in process
                project_3.note(f'Scraping {url}')
                # assign list of posts from scrape_data into scrape variable
                scrape = project_3.scrape_data(url, after)
                for j in range(len(scrape)):  # Loop based on length of scrape variable
                    # Append only the data of each post in scrape variable into posts
                    posts.append(scrape[j]['data'])

                after = None  # Set after = None, to scrape first page
                url = "https://www.reddit.com/r/"+i + \
                    "/top.json?limit=100&t=all"  # Set url based on top reddit topic
                # Display note to inform reader that scraping is in process
                project_3.note(f'Scraping {url}')
                # assign list of posts from scrape_data into scrape variable
                scrape = project_3.scrape_data(url, after)
                for j in range(len(scrape)):  # Loop based on length of scrape variable
                    # Append only the data of each post in scrape variable into posts
                    posts.append(scrape[j]['data'])

                after = None  # Set after = None, to scrape first page
                url = "https://www.reddit.com/r/"+i + \
                    "/rising.json?limit=100"  # Set url based on rising reddit topic
                # Display note to inform reader that scraping is in process
                project_3.note(f'Scraping {url}')
                # assign list of posts from scrape_data into scrape variable
                scrape = project_3.scrape_data(url, after)
                for j in range(len(scrape)):  # Loop based on length of scrape variable
                    # Append only the data of each post in scrape variable into posts
                    posts.append(scrape[j]['data'])

            df = pd.DataFrame(posts)  # Transform posts as DataFrame
            # Merge titles and selftext as content
            df['content'] = df.title + " " + df.selftext
            # Drop duplicate posts with the same content
            df.drop_duplicates(subset='content', inplace=True)
            # Save DataFrame into csv to import without scraping in the future
            df = df_old.append(df)  # Append new DataFrame to old DataFrame
            # Display note when appended
            project_3.note(f'New DataFrame appended')
            df.to_csv(f'../data/news.csv')
            # Display success at end of function
            project_3.success(f'All topics scraped')
        return

    # Function to scrape news posts
    def onion(onion, check):
        posts = []  # Set posts as empty list
        if check == True:  # if True, perform scrape
            df_old = pd.read_csv(f'../data/news.csv')  # Import old DataFrame
            # Display note old DataFrame loaded
            project_3.note(f'Old DataFrame loaded')
            after = None  # Set after = None, to scrape first page
            url = "https://www.reddit.com/r/"+onion + \
                "/hot.json?limit=100"  # Set url based on reddit topic
            # Display note to inform reader that scraping is in process
            project_3.note(f'Scraping {url}')
            # assign list of posts from scrape_data into scrape variable
            scrape = project_3.scrape_data(url, after)
            for j in range(len(scrape)):  # Loop based on length of scrape variable
                # Append only the data of each post in scrape variable into posts
                posts.append(scrape[j]['data'])

            after = None  # Set after = None, to scrape first page
            url = "https://www.reddit.com/r/"+onion + \
                "/new.json?limit=100"  # Set url based on new reddit topic
            # Display note to inform reader that scraping is in process
            project_3.note(f'Scraping {url}')
            # assign list of posts from scrape_data into scrape variable
            scrape = project_3.scrape_data(url, after)
            for j in range(len(scrape)):  # Loop based on length of scrape variable
                # Append only the data of each post in scrape variable into posts
                posts.append(scrape[j]['data'])

            after = None  # Set after = None, to scrape first page
            # Set url based on controversial reddit topic
            url = "https://www.reddit.com/r/"+onion+"/controversial.json?limit=100&t=all"
            # Display note to inform reader that scraping is in process
            project_3.note(f'Scraping {url}')
            # assign list of posts from scrape_data into scrape variable
            scrape = project_3.scrape_data(url, after)
            for j in range(len(scrape)):  # Loop based on length of scrape variable
                # Append only the data of each post in scrape variable into posts
                posts.append(scrape[j]['data'])

            after = None  # Set after = None, to scrape first page
            url = "https://www.reddit.com/r/"+onion + \
                "/top.json?limit=100&t=all"  # Set url based on top reddit topic
            # Display note to inform reader that scraping is in process
            project_3.note(f'Scraping {url}')
            # assign list of posts from scrape_data into scrape variable
            scrape = project_3.scrape_data(url, after)
            for j in range(len(scrape)):  # Loop based on length of scrape variable
                # Append only the data of each post in scrape variable into posts
                posts.append(scrape[j]['data'])

            after = None  # Set after = None, to scrape first page
            # Set url based on rising reddit topic
            url = "https://www.reddit.com/r/"+onion+"/rising.json?limit=100"
            # Display note to inform reader that scraping is in process
            project_3.note(f'Scraping {url}')
            # assign list of posts from scrape_data into scrape variable
            scrape = project_3.scrape_data(url, after)
            for j in range(len(scrape)):  # Loop based on length of scrape variable
                # Append only the data of each post in scrape variable into posts
                posts.append(scrape[j]['data'])

            df = pd.DataFrame(posts)  # Transform posts as DataFrame
            # Merge titles and selftext as content
            df['content'] = df.title + " " + df.selftext
            # Drop duplicate posts with the same content
            df.drop_duplicates(subset='content', inplace=True)
            # Save DataFrame into csv to import without scraping in the future
            df = df_old.append(df)  # Append new DataFrame to old DataFrame
            # Display note when appended
            project_3.note(f'New DataFrame appended')
            df.to_csv(f'../data/onion.csv')
            # Display success at end of function
            project_3.success(f'All topics scraped')
        return

    # Function to scrape the data
    def scrape_data(url, after):

        # Set headers to prevent status code error
        headers = headers = {'User-agent': 'Gauss Markov'}

        posts = []  # Set posts as an empty list
        i = 1
        # Each time, it will only scrape 2100 posts. To get 1,000 posts, need to perform 10 times.
        for _ in range(10):
            if after == None:  # If after == None, scrape first page
                params = {}
            else:  # If there are more posts to scrape, set params = after
                params = {'after': after}
            res = requests.get(url, params=params, headers=headers)
            if res.status_code == 200:  # Check if status code == 200
                data = res.json()
                posts.extend(data['data']['children'])
                after = data['data']['after']

            else:  # If status code is not 200, display warning and break
                warning(f'Status Code {res.status_code}')
                break

            display(Markdown(f'<p style="font-family:courier;"><b>{i}/10:</b> Scraped {len(data["data"]["children"])}</p>'))
            i += 1
            # Set timeout to 3 to prevent being blocked by reddit
            time.sleep(3)
        # Display success when scraping is complete
        project_3.success(f'Scraped {len(posts)} posts from {url}')
        return posts

    # Function to read csv
    def read(n):
        df = pd.read_csv(f'../data/{n}.csv')
        project_3.check(f'There are {df.shape[0]} posts with {df.shape[1]} columns')
        return df

    # Function to merge DataFrames together
    def merge(df_news, df_fake):
        # Create the target
        df_news['is_fake'] = 0  # If news, set is_fake = 0
        df_fake['is_fake'] = 1  # If fake, set is_fake = 1

        # Merge the DataFrames
        df = df_news.append(df_fake, sort=True)  # Append df_fake to df_news
        # Only keep content and is_fake columns
        df = df[['is_fake', 'content']]
        project_3.check(f'There are {df.shape[0]} posts with {df.shape[1]} columns')
        return df

    # Function to stop transition, conjunction, preposition and selected words
    def stop():
        # Selected words that are not English or are able to identify if a content is fake or real
        stop_selected = ['fake', 'real', 'deepfake', 'deepfakes', 'amp', 'let', 'https',
                         'de', 'que', 'en', 'el', 'don', 'un', 're', 'la', 'castaña', 'lo', 'del']

        # Transitions used in English grammar
        stop_transition_similar = ['in', 'the', 'first', 'place', 'not', 'only', 'but', 'also', 'as', 'a', 'matter', 'of', 'fact', 'in', 'like', 'manner', 'in', 'addition', 'coupled', 'with', 'in', 'the', 'same', 'fashion', 'in', 'the', 'same', 'way', 'first', 'second', 'third', 'in', 'the', 'light', 'of', 'not', 'to', 'mention', 'to',
                                   'say', 'nothing', 'of', 'equally', 'important', 'by', 'the', 'same', 'token', 'again', 'to', 'and', 'also', 'then', 'equally', 'identically', 'uniquely', 'like', 'as', 'too', 'moreover', 'as', 'well', 'as', 'together', 'with', 'of', 'course', 'likewise', 'comparatively', 'correspondingly', 'similarly', 'furthermore', 'additionally']
        stop_transition_contradict = ['although', 'this', 'may', 'be', 'true', 'in', 'contrast', 'different', 'from', 'of', 'course', 'but', 'on', 'the', 'other', 'hand', 'on', 'the', 'contrary', 'at', 'the', 'same', 'time', 'in', 'spite', 'of', 'even', 'so', 'even', 'though', 'be', 'that', 'as', 'it', 'may', 'then', 'again',
                                      'above', 'all', 'in', 'reality', 'after', 'all', 'but', 'still', 'unlike', 'or', 'yet', 'while', 'albeit', 'besides', 'as', 'much', 'as', 'even', 'though', 'although', 'instead', 'whereas', 'despite', 'conversely', 'otherwise', 'however', 'rather', 'nevertheless', 'nonetheless', 'regardless', 'notwithstanding']
        stop_transition_condition = ['in', 'the', 'event', 'that', 'granted', 'that', 'so', 'long', 'as', 'as', 'long', 'as' 'on', 'condition', 'on', 'the', 'condition', 'that', 'for', 'the', 'purpose', 'of', 'with', 'this', 'intention', 'with', 'this', 'in', 'mind', 'in', 'the', 'hope', 'that', 'to', 'the', 'end', 'that', 'for', 'fear', 'that',
                                     'in', 'order', 'to', 'seeing', 'that', 'being', 'that', 'in', 'view', 'of', 'if', 'then', 'unless', 'when', 'whenever', 'while', 'because', 'of', 'as', 'since', 'while', 'lest', 'in', 'case', 'provided', 'that', 'given', 'that', 'only', 'if', 'even', 'if', 'so', 'that', 'so', 'as', 'to', 'owing', 'to', 'inasmuch', 'as', 'due', 'to']
        stop_transition_emphasis = ['in', 'other', 'words', 'to', 'put', 'it', 'differently', 'for', 'one', 'thing', 'as', 'an', 'illustration', 'in', 'this', 'case', 'for', 'this', 'reason', 'to', 'put', 'it', 'another', 'way', 'that', 'is', 'to', 'say', 'with', 'attention', 'to', 'by', 'all', 'means', 'important', 'to', 'realize', 'another', 'key', 'point', 'first', 'thing', 'to', 'remember', 'most', 'compelling', 'evidence', 'must', 'be', 'remembered', 'point', 'often', 'overlooked', 'to', 'point', 'out', 'on', 'the', 'positive', 'side',
                                    'on', 'the', 'negative', 'side', 'with', 'this', 'in', 'mind', 'notably', 'including', 'like', 'to', 'be', 'sure', 'namely', 'chiefly', 'truly', 'indeed', 'certainly', 'surely', 'markedly', 'such', 'as', 'especially', 'explicitly', 'specifically', 'expressly', 'surprisingly', 'frequently', 'significantly', 'particularly', 'in', 'fact', 'in', 'general', 'in', 'particular', 'in', 'detail', 'for', 'example', 'for', 'instance', 'to', 'demonstrate', 'to', 'emphasize', 'to', 'repeat', 'to', 'clarify', 'to', 'explain', 'to', 'enumerate']
        stop_transition_effect = ['as', 'a', 'result', 'under', 'those', 'circumstances', 'in', 'that', 'case', 'for', 'this', 'reason', 'in', 'effect',
                                  'for', 'thus', 'because', 'the', 'then', 'hence', 'consequently', 'therefore', 'thereupon', 'forthwith', 'accordingly', 'henceforth']
        stop_trasition_conclusion = ['as', 'can', 'be', 'seen', 'generally', 'speaking', 'in', 'the', 'final', 'analysis', 'all', 'things', 'considered', 'as', 'shown', 'above', 'in', 'the', 'long', 'run', 'given', 'these', 'points', 'as', 'has', 'been', 'noted', 'in', 'a', 'word', 'for', 'the', 'most', 'part', 'after', 'all', 'in', 'fact', 'in',
                                     'summary', 'in', 'conclusion', 'in', 'short', 'in', 'brief', 'in', 'essence', 'to', 'summarize', 'on', 'balance', 'altogether', 'overall', 'ordinarily', 'usually', 'by', 'and', 'large', 'to', 'sum', 'up', 'on', 'the', 'whole', 'in', 'any', 'event', 'in', 'either', 'case', 'all', 'in', 'all', 'obviously', 'ultimately', 'definitely']
        stop_transition_time = ['at', 'the', 'present', 'time', 'from', 'time', 'to', 'time', 'sooner', 'or', 'later', 'at', 'the', 'same', 'time', 'up', 'to', 'the', 'present', 'time', 'to', 'begin', 'with', 'in', 'due', 'time', 'as', 'soon', 'as', 'as', 'long', 'as', 'in', 'the', 'meantime', 'in', 'a', 'moment', 'without', 'delay', 'in', 'the', 'first', 'place', 'all', 'of', 'a', 'sudden', 'at', 'this', 'instant', 'first', 'second',
                                'immediately', 'quickly', 'finally', 'after', 'later', 'last', 'until', 'till', 'since', 'then', 'before', 'hence', 'since', 'when', 'once', 'about', 'next', 'now', 'formerly', 'suddenly', 'shortly', 'henceforth', 'whenever', 'eventually', 'meanwhile', 'further', 'during', 'in', 'time', 'prior', 'to', 'forthwith', 'straightaway', 'by', 'the', 'time', 'whenever', 'until', 'now', 'now', 'that', 'instantly', 'presently', 'occasionally']
        stop_transition_place = ['in', 'the', 'middle', 'to', 'the', 'left', 'to', 'the', 'right', 'in', 'front', 'of', 'on', 'this', 'side', 'in', 'the', 'distance', 'here', 'and', 'there', 'in', 'the', 'foreground', 'in', 'the', 'background', 'in', 'the', 'center', 'of', 'adjacent',
                                 'to', 'opposite', 'to', 'here', 'there', 'next', 'where', 'from', 'over', 'near', 'above', 'below', 'down', 'up', 'under', 'further', 'beyond', 'nearby', 'wherever', 'around', 'between', 'before', 'alongside', 'amid', 'among', 'beneath', 'beside', 'behind', 'across']
        stop_transition = stop_transition_similar + stop_transition_contradict + stop_transition_condition + \
            stop_transition_emphasis + stop_transition_effect + \
            stop_trasition_conclusion + stop_transition_time + stop_transition_place

        # Conjunctions used in English grammar
        stop_conjunction = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'though', 'although', 'even', 'though', 'while', 'if', 'only', 'if', 'unless', 'until', 'provided', 'that', 'assuming', 'that', 'even', 'if', 'in', 'case', 'that', 'in', 'case', 'lest', 'than', 'rather', 'than', 'whether', 'as', 'much', 'as', 'whereas', 'after', 'as', 'long', 'as', 'as', 'soon', 'as', 'before', 'by', 'the', 'time', 'now', 'that', 'once', 'since', 'till', 'until', 'when', 'whenever', 'while', 'because', 'since', 'so', 'that', 'in', 'order', 'that', 'in', 'order', 'why', 'that', 'what', 'whatever',
                            'which', 'whichever', 'who', 'whoever', 'whom', 'whomever', 'whose', 'how', 'as', 'though', 'as', 'if', 'where', 'wherever', 'as', 'just as', 'both', 'hardly', 'scarcely', 'so', 'when', 'and', 'either', 'or', 'neither', 'nor', 'if', 'then', 'not', 'but', 'what', 'with', 'whether', 'or', 'not', 'only', 'also', 'no', 'sooner', 'rather', 'than', 'also', 'besides', 'furthermore', 'likewise', 'moreover', 'however', 'nevertheless', 'nonetheless', 'still', 'conversely', 'instead', 'otherwise', 'rather', 'accordingly', 'consequently', 'hence', 'meanwhile', 'then', 'therefore', 'thus']

        # Prepositions used in English grammar
        stop_prepositions = ['on', 'in', 'at', 'since', 'until', 'till', 'for', 'ago', 'during', 'before', 'after', 'to', 'past', 'from', 'by', 'in', 'at', 'on', 'off', 'by',
                             'beside', 'under', 'over', 'below', 'above', 'up', 'down', 'across', 'through', 'to', 'into', 'out', 'of', 'onto', 'towards', 'from', 'of', 'by', 'about', 'for', 'with']

        # Make a list of all stop words
        stopwords = stop_selected + stop_transition + \
            stop_conjunction + stop_prepositions + list(STOPWORDS)

        return stopwords

    # Function to display wordcloud
    def wordcloud(data, content):
        news_colour = ['#FFFFFF', '#A3CDFF', '#D1E6FF',
                       '#162F4D', '#1F306E']  # Set colour scheme for news
        fake_colour = ['#FFFFFF', '#FFA6A6', '#FFD2D2',
                       '#A13030', '#E74645']  # Set colour scheme for fake
        default_colour = ['#EFEEB4', '#DAD873', '#58B368',
                          '#309975', '#454D66']  # Set colour scheme for fake
        if content == 'news':  # If news, use news_colour
            theme = news_colour
        elif content == 'fake':  # If fake, use fake_colour
            theme = fake_colour
        else:  # If neither, use default_colour
            theme = default_colour
        wc = WordCloud(  # Initiate WordCloud
            background_color='white',  # Set background as white
            stopwords=project_3.stop(),  # Set stopwords as defined in function
            max_words=1000,  # Only show top 1,000 words
            max_font_size=100,  # Emphasise on common words
            scale=3,  # Determine the scale factor
            random_state=42,  # Set random state as 42
            colormap=LinearSegmentedColormap.from_list(
                "mycmap", theme),  # Define colours
        ).generate(str(data))  # Generate the string output
        fig = plt.figure(1, figsize=(16, 9))  # Set figure size as 16:9 ratio
        plt.axis('off')  # Do not display axis
        plt.imshow(wc)  # Plot wordcloud
        plt.show()  # Display plot

    # Function to pre-process content before analysing
    def clean(df, lemma):
        i = 0
        df_tmp = df
        while i < df_tmp.shape[0]:
            df_tmp.content[i] = BeautifulSoup(
                df_tmp.content[i]).get_text()  # Remove HTMLs
            df_tmp.content[i] = re.sub(r'^https:\/\/\S+(\/\S+)*(\/)?$', '',
                                       df_tmp.content[i], flags=re.MULTILINE)  # Remove URLs
            df_tmp.content[i] = re.sub(
                "[^a-zA-Z]", " ", df_tmp.content[i])  # Remove non-letters
            # Convert to lower case, split into individual words
            df_tmp.content[i] = df_tmp.content[i].lower()
            if lemma == True:
                tokenizer = RegexpTokenizer(r'\w+')
                lemmatizer = WordNetLemmatizer()
                df_tmp.content[i] = tokenizer.tokenize(df_tmp.content[i])
                df_tmp.content[i] = [lemmatizer.lemmatize(
                    j, pos="v") for j in df_tmp.content[i]]
                # Join the words back into one string separated by space, and return the result
                df_tmp.content[i] = (" ".join(df_tmp.content[i]))
            i += 1
            if i % 10000 == 0:
                display(Markdown(f'Cleaned {i} rows out of {df_tmp.shape[0]}'))
        project_3.success(f'{df_tmp.shape[0]} rows cleaned')
        return df_tmp

    # Function to find the features in the DataFrame
    def features(df, y):
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, random_state=42, stratify=y)  # Train test split
        # Initialise CountVectorizer
        cv = CountVectorizer(stop_words=project_3.stop())
        # Fit and transform content
        X_train = cv.fit_transform(X_train.content)
        # Create a DataFrame of content
        cv_train = pd.DataFrame(
            X_train.todense(), columns=cv.get_feature_names())
        words = list(cv_train.sum().sort_values(
            ascending=False).index)  # Create a list of words
        print(len(words))
        cv_train['is_fake'] = y_train.values
        cv_train = cv_train.groupby('is_fake').sum()[words]
        project_3.note(f'Top 10 words')
        display(cv_train.T.head(10))
        return words

    # Function to search the best parameters
    def search(vects, models, df, y, use_params):
        model_solns = {}  # Create an empty dictionary
        idx = 0  # Set model index as 0
        for v in vects:  # For each vectorizer in vects and for each model in models, perform grid search
            for m in models:
                X_train, X_test, y_train, y_test = train_test_split(
                    df, y, random_state=42, stratify=y)  # Train test split and stratify by y
                idx += 1  # Inccrease the index number by 1
                # Declare vecorizer in loop as pipe_items list
                pipe_items = [v]
                pipe_items.append(m)  # Append the model to pipe_items
                [train_score, test_score, y_test, y_test_hat, best_params, f1_score] = project_3.pipeline(
                    pipe_items, use_params, X_train.content, X_test.content, y_train, y_test)  # Call pipeline to perform grid search for best parameters
                model_solns[idx] = {'vectorizer': v, 'model': m, 'train_score': train_score, 'test_score': test_score,
                                    'best_params': best_params, 'f1_score': f1_score}  # Add the values of grid search to model_solns dictionary
        # Display success when completed
        project_3.success(f'<b>COMPLETED</b>')
        # Create a DataFrame from dictionary
        df_solns = pd.DataFrame(model_solns)
        # Save DataFrame as csv so we do not need to run this the next time
        df_solns.to_csv(f'../data/gridsearch.csv')
        return df_solns, X_train, X_test, y_train, train_score, test_score, y_test, y_test_hat, best_params, f1_score

    # Function to search best parameters using grid search cv via pipeline
    def pipeline(items, use_params, X_train, X_test, y_train, y_test):
        pipe_items = {  # Initialise vectorizers and models for our pipeline
            # Initialise CountVectorizer
            'cv': CountVectorizer(),
            # Initialise TfidfVectorizer
            'tv': TfidfVectorizer(),
            # Initialise HashingVectorizer
            'hv': HashingVectorizer(alternate_sign=True),
            'wv': Word2Vec(),
            'lr': LogisticRegression(),  # Initialise LogisticRegression
            'bnb': BernoulliNB(),  # Initialise BernoulliNB
            'mnb': MultinomialNB(),  # Initialise MultinomialNB
            'gnb': GaussianNB(),  # Initialise GaussianNB
            'knn': KNeighborsClassifier()
        }
        param_items = {  # Declare parameters for grid search
            'cv': {  # CountVectorizer parameters
                'cv__stop_words': [None, project_3.stop()],
                'cv__ngram_range': [(1, 1), (1, 2)],  # ngram range from 1 to 3
                'cv__max_df': [0.95, 1.0],  # max df of 95% or 100%
                'cv__min_df': [1, 2],  # min df of 1 or 2 documents
                # max features from 1,000 to maximum
                'cv__max_features': [10000, 20000, 3000, None]
            },
            'tv': {  # TfidfVectorizer parameters
                'tv__stop_words': [None, project_3.stop()],
                'tv__ngram_range': [(1, 1), (1, 2)],  # ngram range from 1 to 3
                'tv__max_df': [0.95, 1.0],  # max df of 95% or 100%
                'tv__min_df': [1, 2],  # min df of 1 or 2 documents
                # max features from 1,000 to maximum
                'tv__max_features': [10000, 20000, 3000, None]
            },
            'hv': {  # HashingVectorizer parameters
                # ngram range from 1 to 3
                'hv__ngram_range': [(1, 1), (1, 2), (1, 3)]
            },
            'wv': {

            },
            'lr': {  # LogisticRegression parameters
                'lr__C': [1, .05],  # Inverse alpha of 1 and 0.05
                'lr__penalty': ['l1', 'l2']  # L1 or L2 regularization
            },
            'bnb': {  # BernoulliNB parameters
                # alpha range from 0.0001 to 1.0
                'bnb__alpha': [0.0001, 0.01, 1.0]
            },
            'mnb': {  # MultinomialNB parameters
                # alpha range from 0.0001 to 1.0
                'mnb__alpha': [0.0001, 0.01, 1.0]
            },
            'gnb': {  # GaussianNB parameters
            },
            'knn': {
                'knn__n_neighbors': [35, 200]
            }
        }
        params = dict()  # Create the parameters for GridSearch
        if use_params:
            for i in items:  # for each vectorizer,
                for p in param_items[i]:  # for each parameter
                    # set parameters for vectorizer in the dictionary
                    params[p] = param_items[i][p]
        pipe_list = [(i, pipe_items[i]) for i in items]
        method = list()  # create an empty list
        for p in pipe_list:  # for each vectorizer in pipeline
            # append values to method list
            method.append(str(p[1]).split('(')[0])
        # Display vectorizer and model used for current grid search
        display(Markdown(f'<b>Using {method[0]} with {method[1]}</b>'))
        pipe = Pipeline(pipe_list)  # Create the pipeline
        # Perform grid search for the vectorizer and parameters in pipeline. Verbose set to 1 to display progress
        gs = GridSearchCV(pipe, param_grid=params,
                          verbose=1, pre_dispatch=None, cv=2)
        # fit values of grid search onto X_train and y_train
        gs.fit(X_train, y_train)
        train_params = gs.best_params_  # Declare the best parameters for model
        train_score = gs.best_score_  # Declare training score for model
        y_test_hat = gs.predict(X_test)  # Predict target from model
        test_score = gs.score(X_test, y_test)  # Accuracy of model
        for k in train_params:  # For each parameter in grid search,
            print(f"{k}: {train_params[k]}")  # Print parameter used
        print(f'\nTrain score: {train_score}')  # Print training score
        print(f'Test score: {test_score} (Accuracy)\n')  # Print accuracy
        tn, fp, fn, tp = confusion_matrix(
            y_test, y_test_hat).ravel()  # Get the confusion matrix
        f1_score = 2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))
        print(f"True Negatives: {tn}")  # Predicted real news correctly
        # Predicted fake news incorrectly
        print(f"False Positives: {fp} (Type I Error)")
        # Predicted real news incorrectly
        print(f"False Negatives: {fn} (Type II Error)")
        print(f"True Positives: {tp}\n")  # Predicted fake news correctly
        # Display completed banner
        display(Markdown(f'<b>Sensitivity:</b> {tp/(tp+fn)}'))
        display(Markdown(f'<b>Miss Rate:</b> {fn/(tp+fn)}'))
        display(Markdown(f'<b>Specificity:</b> {tn/(fp+tn)}'))
        display(Markdown(f'<b>Fall Out Rate:</b> {fp/(fp+tn)}'))
        print()
        display(Markdown(f'<b>Precision:</b> {tp/(tp+fp)}'))
        display(Markdown(f'<b>False Discovery Rate:</b> {fp/(tp+fp)}'))
        display(Markdown(f'<b>Negative Predictive Value:</b> {tn/(tn+fn)}'))
        display(Markdown(f'<b>False Omission Rate:</b> {fn/(tn+fn)}'))
        print()
        display(Markdown(f'<b>Prevalence:</b> {(tp+fn)/(tp+fp+tn+fn)}'))
        display(Markdown(f'<b>Accuracy:</b> {(tp+tn)/(tp+fp+tn+fn)}'))
        print()
        display(Markdown(f'<b>Positive Likelihood Ratio:</b> {(tp/(tp+fn))/(fp/(fp+tn))}'))
        display(Markdown(f'<b>Negative Likelihood Ratio:</b> {(fn/(tp+fn))/(tn/(fp+tn))}'))
        display(Markdown(f'<b>Diagnostic Odds Ratio:</b> {((tp/(tp+fn))/(fp/(fp+tn)))/((fn/(tp+fn))/(tn/(fp+tn)))}'))
        display(Markdown(f'<b>F1 Score:</b> {2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))}'))
        print()
        project_3.check(f'Completed {method[0]} with {method[1]}')
        return train_score, test_score, y_test, y_test_hat, train_params, f1_score

    # Function to get user content
    def get_input():
        # Prompts user to key in content
        user_input = [str('"'+input("Enter content to test:\n")+'"')]
        df_user = pd.DataFrame(list(reader(user_input)))  # Create a DataFrame
        # Rename feature as content
        df_user = df_user.rename(columns={0: 'content'})
        k = 0
        while k == 0:
            user_is_fake = str(input("Is the content fake (Y/N)?\n"))
            if user_is_fake.lower() == 'y':
                df_user['is_fake'] = 1
                k += 1
            elif user_is_fake.lower() == 'n':
                df_user['is_fake'] = 0
                k += 1
            else:
                project_3.warning(f'Invalid entry. Please try again.')
        return df_user

    # Function to check if user content is fake or not
    def check_user(best_vect, best_max_df, best_min_df, best_ngram_range, best_max_features, best_model, best_alpha, df, df_user, y, default_penalty='l2'):
        if best_vect == 'cv':  # If cv, use CountVectorizer with best parameters
            vect = CountVectorizer(stop_words=project_3.stop(
            ), max_df=best_max_df, min_df=best_min_df, ngram_range=best_ngram_range, max_features=best_max_features)
        elif best_vect == 'tv':  # If tv, use TfidfVectorizer with best parameters
            vect = TfidfVectorizer(stop_words=project_3.stop(
            ), max_df=best_max_df, min_df=best_min_df, ngram_range=best_ngram_range, max_features=best_max_features)
        elif best_vect == 'hv':  # If hv, use HashingVectorizer with best parameters
            vect = HashingVectorizer(stop_words=project_3.stop(
            ), alternate_sign=True, max_df=best_max_df, min_df=best_min_df, ngram_range=best_ngram_range)
        else:  # If none of above, display error
            project_3.warning(f'Error')
            return
        if best_model == 'lr':  # If lr, use LogisticRegression with best parameters
            model = LogisticRegression(C=best_alpha, penalty=default_penalty)
        elif best_model == 'bnb':  # If bnb, use BernoulliNB with best parameters
            model = BernoulliNB(alpha=best_alpha)
        elif best_model == 'mnb':  # If mnb, use MultinomialNB with best parameters
            model = MultinomialNB(alpha=best_alpha)
        elif best_model == 'gnb':  # If gnb, use GaussianNB
            model = GaussianNB()
        else:  # If none of above, display error
            project_3.warning(f'Error')
            return
        X = df.content  # Set X as training content
        X_user = df_user.content  # Set X_user as user input content
        # Use selected vectorizer to fit and transform X
        X = vect.fit_transform(X)
        # Use same vectorizer to transform X_user
        X_user = vect.transform(X_user)
        model.fit(X, y)  # Use selected model to fit X and y
        # Use same model to predict fake or not using user content
        df_user['pred_fake'] = model.predict(X_user)
        display(df_user)
        return df_user

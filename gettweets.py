## This script will collect tweets from politician's twitter handles

import sys
import operator
import requests
import twitter
import csv

twitter_consumer_key = 'xQVTgPR6yd5BieRzgz8ykQVqF'
twitter_consumer_secret = 'gowijejSpDyBHt5hYoUz0ByPf7haJ2USOlXiqj05zKCR7P2UsI'
twitter_access_token = '52266958-DCsi4ANXJ8cb5RGfNqLejWMwj85r11V1gKDh8ihAt'
twitter_access_secret = '12fddueZRrGTexbEzbEfI6JE01wNbsWBu4X9Rj8hIXG1g'

twitter_api = twitter.Api(consumer_key=twitter_consumer_key,
                          consumer_secret=twitter_consumer_secret,
                          access_token_key=twitter_access_token,
                          access_token_secret=twitter_access_secret)

handles = ['@RealDonaldTrump',
            '@SenJohnMcCain',
            '@SenateMajLdr',
            '@SpeakerRyan',
            '@marcorubio',
            '@tedcruz',
            '@mike_pence',
            '@LindseyGrahamSC',
            '@GOPLeader',
            '@HillaryClinton',
            '@chuckschumer',
            '@NancyPelosi',
            '@BernieSanders',
            '@CoryBooker',
            '@SenatorDurbin',
            '@PattyMurray',
            '@SenWarren',
            '@TulsiGabbard',
            '@GovGaryJohnson',
            '@DrJillStein']

if __name__ == '__main__':
    # Collect Tweets
    # with csv.writer('output.csv') as f:
    f = open('output.csv', mode='w')
    f = csv.writer(f)
    # For csv header
    fields = ['handle', 'text', 'fav', 'location']
    f.writerow(fields)
    
    for handle in handles:
        statuses = twitter_api.GetUserTimeline(screen_name=handle, 
                                               count=200, 
                                               include_rts=True,
                                               exclude_replies=True)
        for status in statuses:
            # Grab tweet text, favorite count, location
            print(str(type(status.text)))
            f.writerow([str(handle),
                        status.text.encode('unicode_escape'),
                        # str(status.text.encode,'utf-8'),
                        #status.text.encode('ascii'),
                        str(status.favorite_count),
                        str(status.geo),
                        '\n'])

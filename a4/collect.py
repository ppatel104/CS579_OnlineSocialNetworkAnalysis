"""
collect.py
"""
from collections import Counter,defaultdict
from TwitterAPI import TwitterAPI
import sys
import time
import os
import pandas as pd
from textblob import TextBlob

consumer_key = 'ZOOWwEthSfHkfMPqeOEE4wFZB'
consumer_secret = '50PRBUll1cOjQadxumw00OotYjmJxOfMsCcLiQwnueAxmL6cBb'
access_token = '719604138674413569-U8kncovTj7NHDhI7aLN4HVggcjZD10o'
access_token_secret = 'HblWXoW5iEYjbHRCEfLxt7tEfwcPQDlFuT3op2LbumQr8'

def get_twitter():
    """ Create an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_tweets(twitter):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids
    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.
    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of responses, and a list of user_ids.
    """
    main_response = []
    main_ids = []
    max_tweets = 10000
    m_id = False
    max_id = 0
    while len(main_ids)<max_tweets:
        try:
            if(not m_id):
                request = robust_request(twitter,'search/tweets',{'q':'@united','count':100,'lang':'en'})
                response = [r for r in request]
                ids = [x['id'] for x in response]
                main_response += response
                main_ids += ids
                if(len(main_ids)<max_tweets):
                    max_id = response[len(response)-1]['id']
                    m_id = True
            else:
                request = robust_request(twitter,'search/tweets',{'q':'@united','count':100,'lang':'en','max_id':max_id})
                response = [r for r in request]
                ids = [x['id'] for x in response]
                main_response += response
                main_ids += ids
                if(len(main_ids)<max_tweets):
                    max_id = response[len(response)-1]['id']
        except:
            break
    return main_response, main_ids


def collect_tweets():
    twitter = get_twitter()
    responses, ids = get_tweets(twitter)
    t_id = []
    seen = set()
    for i in range(0, len(responses)):
        if (responses[i]['text'] not in seen):
            t_id.append([responses[i]['id'], responses[i]['text']])
            seen.add(responses[i]['text'])
    for i in t_id:
        blob = TextBlob(i[1])
        if (blob.sentiment.polarity > 0):
            i.append(1)
        if (blob.sentiment.polarity < 0):
            i.append(0)
    t_i = [x for x in t_id if len(x) > 2]
    tweet_data = pd.DataFrame(t_i,columns=["tweet_user_id","tweet","sentiment"])
    return tweet_data

def get_friends(twitter,screen_name,count):
    request = twitter.request('followers/ids',{'screen_name':screen_name,'count':count})
    response = [r for r in request]
    return response

def get_users(twitter, screen_names):
    u_ids = []
    for i in screen_names:
        request = twitter.request('users/lookup',{'screen_name':i})
        response = [r['id'] for r in request]
        u_ids.append((i,response[0]))
    return u_ids

def main():
    twitter = get_twitter()
    users = ['BillGates','tim_cook']
    u_id = get_users(twitter, users)
    l = []
    f = open("friends.txt", "w")
    for i in u_id:
        for j in get_friends(twitter, i[0], 100):
            l.append([i[0], i[1], j])
            f.write("%s" % i[1] + "\t" + "%s\n" % j)
    f.close()
    tweet_data = collect_tweets()
    print ("Number of users collected: %d"%len(l))
    print ("Number of messsages collected: %d"%len(tweet_data))
    tweet_data.to_csv("united_tweets.csv",index=False)
    f = open("collect.txt", "w")
    f.write("%d\n"%len(l))
    f.write("%d"%len(tweet_data))
    f.close()

if __name__ == '__main__':
    print ("-------------Collect.py----------------")
    main()
    print ("-----------Collect.py------------------")
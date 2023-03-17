import datetime as dt
import textwrap
import praw
from pmaw import PushshiftAPI
import pandas
import sys
import numpy as np
import requests
import time
from dateutil.relativedelta import relativedelta 
import matplotlib.pyplot as plt

sys.path.append('../NaiveBayes')

import bayes

def requestJSON(url):
    while True:
        try:
            r = requests.get(url)
            if r.status_code != 200:
                print('error code', r.status_code)
                time.sleep(5)
                continue
            else:
                break
        except Exception as e:
            print(e)
            time.sleep(5)
            continue
    return r.json()

def show_replies(comment, depth=0):
    print(textwrap.indent("-{}".format(comment.body), depth * '  '))
    for reply in comment.replies:
        show_replies(reply, depth+1)

def submission_to_dict(submission):
    return {'title':submission.title, 'comments':submission.num_commments, 'score':submission.score, 'ratio':submission.upvote_ratio, 'edited':submission.edited, 'locked':submission.locked ,'author':submission.author}


agent = "r/ucsc data scrapper"
reddit = praw.Reddit(client_id="edhlsn3xRnsuK_U1n_qOfQ", client_secret="DyZHk39I6aJmKweONoS3IFU2kmyHIQ", user_agent = agent)

api = PushshiftAPI()

model = bayes.NaiveBayes()
model.load('../NaiveBayes/trained_models/attempt1.csv')
model.load_add('../NaiveBayes/trained_models/attempt3.csv')

urls = []

start_date=dt.datetime(2023, 3, 10)
end_date=dt.datetime(2023, 3, 17)

for i in range(25):
    url = 'https://api.pushshift.io/reddit/search/'\
                  + "submission"+ '/?subreddit='\
                  + "UCSC"\
                  + '&size=' + "1000"\
                  + '&after=' + str(int(start_date.timestamp()))\
                  + '&before=' + str(int(end_date.timestamp()))
    print(url)
    json = requestJSON(url)
    print(len(json['data']))
    urls += json['data']
    start_date -= relativedelta(weeks=1)
    end_date -= relativedelta(weeks=1)

print(len(urls))

count = 0

weeks = [*range(1,54)]
counts = np.zeros(len(weeks))

classes = {'politics':np.zeros(len(weeks)), 'relationships':np.zeros(len(weeks)), 'classwork':np.zeros(len(weeks)), 'jokes':np.zeros(len(weeks)), 'complaints':np.zeros(len(weeks))}
classes_upvotes = {'politics':np.zeros(len(weeks)), 'relationships':np.zeros(len(weeks)), 'classwork':np.zeros(len(weeks)), 'jokes':np.zeros(len(weeks)), 'complaints':np.zeros(len(weeks))}

for post in urls:
    try:
        submission = reddit.submission(url=post["url"])
        classification = model.predict(submission)
        count += 1
        time = dt.datetime.fromtimestamp(submission.created)
        if count % 10 == 0:
            print(count)
        week = int((time.timetuple().tm_yday)//7)
        counts[week] += 1
        for i in classification:
            #print("here")
            classes[i][week] += 1
            #print(submission.upvote_ratio)
            classes_upvotes[i][week] += submission.upvote_ratio
    except:
        pass

def unzero(arr):
    replaced = arr.copy()
    replaced[replaced == 0] = 1
    return replaced

plt.plot(weeks, np.array(classes_upvotes['politics'])/unzero(np.array(classes['politics'])), label="political")
plt.plot(weeks, np.array(classes_upvotes['relationships'])/unzero(np.array(classes['relationships'])), label="relationship")
plt.plot(weeks, np.array(classes_upvotes['classwork'])/unzero(np.array(classes['classwork'])), label="classwork")
plt.plot(weeks, np.array(classes_upvotes['jokes'])/unzero(np.array(classes['jokes'])), label="jokes")
plt.plot(weeks, np.array(classes_upvotes['complaints'])/unzero(np.array(classes['complaints'])), label="complaints")
plt.xlabel('Week')
plt.ylabel('Weekly Average Upvotes')
plt.legend()
plt.show()
plt.savefig('Reddit_posts.png')

print("politics:",np.array(classes_upvotes['politics']).sum()/np.array(classes['politics']).sum())
print("relationships:",np.array(classes_upvotes['relationships']).sum()/np.array(classes['relationships']).sum())
print("classwork:",np.array(classes_upvotes['classwork']).sum()/np.array(classes['classwork']).sum())
print("jokes:",np.array(classes_upvotes['jokes']).sum()/np.array(classes['jokes']).sum())
print("complaints:",np.array(classes_upvotes['complaints']).sum()/np.array(classes['complaints']).sum())
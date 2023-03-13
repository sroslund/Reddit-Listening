import datetime as dt
import textwrap
import praw
from pmaw import PushshiftAPI
import pandas

def show_replies(comment, depth=0):
    print(textwrap.indent("-{}".format(comment.body), depth * '  '))
    for reply in comment.replies:
        show_replies(reply, depth+1)

def submission_to_dict(submission):
    return {'title':submission.title, 'comments':submission.num_commments, 'score':submission.score, 'ratio':submission.upvote_ratio, 'edited':submission.edited, 'locked':submission.locked ,'author':submission.author}


agent = "r/ucsc data scrapper"
reddit = praw.Reddit(client_id="edhlsn3xRnsuK_U1n_qOfQ", client_secret="DyZHk39I6aJmKweONoS3IFU2kmyHIQ", user_agent = agent)

api = PushshiftAPI()

start_date=int(dt.datetime(2022, 9, 23).timestamp())
#end_date=int(dt.datetime(2022, 2, 20).timestamp())

pmaw_api = api.search_submissions(subreddit='UCSC', after=start_date, limit=1)

print(len(pmaw_api))

urls = [x['url'] for x in pmaw_api if x['url'][-3:] != 'jpg']

print(urls)

for url in urls:
    try:
        submission = reddit.submission(url=url)
        print('#####################################')
        submission.comments.replace_more(limit=None)
        print(submission.selftext)
        for comment in submission.comments:
            show_replies(comment)
    except:
        pass
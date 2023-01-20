import numpy as np
import praw 
import textwrap
from pmaw import PushshiftAPI

# put words into a set so we don't double count

agent = "r/ucsc data scrapper"
reddit = praw.Reddit(client_id="edhlsn3xRnsuK_U1n_qOfQ", client_secret="DyZHk39I6aJmKweONoS3IFU2kmyHIQ", user_agent = agent)

def show_replies(comment, depth=0):
    print(textwrap.indent("-{}".format(comment.body), depth * '  '))
    for reply in comment.replies:
        show_replies(reply, depth+1)

for submission in reddit.subreddit('ucsc').top(limit=2):
    print('#####################################')
    submission.comments.replace_more(limit=None)
    print(submission.title)
    for comment in submission.comments:
        show_replies(comment)
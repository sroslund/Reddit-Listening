import datetime as dt
import textwrap
import praw
from pmaw import PushshiftAPI

def show_replies(comment, depth=0):
    print(textwrap.indent("-{}".format(comment.body), depth * '  '))
    for reply in comment.replies:
        show_replies(reply, depth+1)


agent = "r/ucsc data scrapper"
reddit = praw.Reddit(client_id="edhlsn3xRnsuK_U1n_qOfQ", client_secret="DyZHk39I6aJmKweONoS3IFU2kmyHIQ", user_agent = agent)

api = PushshiftAPI()

start_epoch=int(dt.datetime(2022, 1, 1).timestamp())
end_epoch=int(dt.datetime(2022, 2, 20).timestamp())

pmaw_api = api.search_submissions(subreddit='UCSC' ,limit=50)

print(len(pmaw_api))

urls = [x['url'] for x in pmaw_api if x['url'][-3:] != 'jpg']

print(urls)

for url in urls:
    try:
        submission = reddit.submission(url=url)
    except:
        pass
    print('#####################################')
    #submission.comments.replace_more(limit=None)
    print(submission.title)
    for comment in submission.comments:
        show_replies(comment)
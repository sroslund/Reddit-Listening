from collections import defaultdict
import datetime as dt
import re
import pandas as pd

# p = political, r = relationships, c = classwork, j = joke, k = complaint, o = other

def get_replies(comment, depth=0):
    comment_data = set(re.split('[ ,;.!?/\n-"()*]',comment.body.lower()))
    for reply in comment.replies:
        comment_data |= get_replies(reply, depth+1)
    return comment_data

class NaiveBayes:
    def __init__(self):
        self.classes = ['politics', 'relationships', 'classwork', 'jokes', 'complaints']

        # this defines the initial probability of each class
        self.counts = [defaultdict(lambda: 4),defaultdict(lambda: 4),defaultdict(lambda: 4),defaultdict(lambda: 4),defaultdict(lambda: 4)]
        
        self.training_subreddits = [["PoliticalDiscussion"],["relationships"],["HomeworkHelp"],["funny"],["depression"]]

    def add_word(self, word, classification):
        self.counts[classification][word] += 1

    def train(self, pushapi, reddit):

        start_date=int(dt.datetime(2022, 1, 1).timestamp()) #modify this line to change the time we are searching

        for i in range(len(self.training_subreddits)):
            pmaw_api = pushapi.search_submissions(subreddit=self.training_subreddits[i][0], after=start_date, limit=1000)
            urls = [x['url'] for x in pmaw_api if x['url'][-3:] != 'jpg']
            for url in urls:
                try:
                    submission = reddit.submission(url=url)
                    submission.comments.replace_more(limit=None)

                    

                    words = set(re.split('[ ,;.!?/\n-"()*]',submission.title.lower()))
                    words |= set(re.split('[ ,;.!?/\n-"()*]',submission.selftext.lower()))
                    for comment in submission.comments:
                        words |= get_replies(comment)
                    for word in words:
                        self.add_word(word,i)
                except:
                    pass
            print("Finished Training {}".format(self.classes[i]))

    def save(self, path):
        df = pd.DataFrame.from_records(self.counts,self.classes)
        df.to_csv(path)

    def load(self, path):
        pass

    def __repr__(self):
        out = ""
        for i in range(len(self.counts)):
            out += self.training_subreddits[i][0]
            out += '\n'
            out += str(self.counts[i])
            out += '\n\n'
        return out

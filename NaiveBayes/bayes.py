from collections import defaultdict
import datetime as dt
import re
import pandas as pd
import numpy as np

use_list = True

# p = political, r = relationships, c = classwork, j = joke, k = complaint, o = other

def get_replies_list(comment, depth=0):
    #comment_data = set(re.split('[ ,;.!?/\n-"()*]',comment.body.lower()))
    comment_data = list(re.split('[ ,;.!?/\n-"()*]',comment.body.lower()))
    for reply in comment.replies:
        #comment_data |= get_replies(reply, depth+1)
        comment_data += get_replies_list(reply, depth+1)
    return comment_data

def get_replies(comment, depth=0):
    comment_data = set(re.split('[ ,;.!?/\n-"()*]',comment.body.lower()))
    #comment_data = list(re.split('[ ,;.!?/\n-"()*]',comment.body.lower()))
    for reply in comment.replies:
        comment_data |= get_replies(reply, depth+1)
        #comment_data += get_replies(reply, depth+1)
    return comment_data

class NaiveBayes:
    def __init__(self):
        self.classes = ['politics', 'relationships', 'classwork', 'jokes', 'complaints']

        # this defines the initial probability of each class
        self.counts = [defaultdict(lambda: 4),defaultdict(lambda: 4),defaultdict(lambda: 4),defaultdict(lambda: 4),defaultdict(lambda: 4)]
        
        #self.training_subreddits = [["PoliticalDiscussion"],["relationships"],["HomeworkHelp"],["funny"],["depression"]]
        self.training_subreddits = [["uspolitics"],["socialskills"],["GetStudying"],["copypasta"],["Vent"]]

    def add_word(self, word, classification):
        self.counts[classification][word] += 1

    def train(self, pushapi, reddit):

        start_date=int(dt.datetime(2022, 1, 1).timestamp()) #modify this line to change the time we are searching
        #end_date = int(dt.datetime(2022, 1, 1).timestamp())

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

    def predict(self, submission):
        #normalize_2 = np.array([.70,.85,1.50,3.5,1.05])
        normalize = np.array([1.1,.75,1.22,1.52,1.00])

        probabilities = np.array([.2,.2,.2,.2,.2])
        submission.comments.replace_more(limit=None)
        #words = set(re.split('[ ,;.!?/\n-"()*]',submission.title.lower()))
        #words |= set(re.split('[ ,;.!?/\n-"()*]',submission.selftext.lower()))
        words = list(re.split('[ ,;.!?/\n-"()*]',submission.title.lower()))
        words += list(re.split('[ ,;.!?/\n-"()*]',submission.selftext.lower()))
        for comment in submission.comments:
            #words |= get_replies(comment)
            words += get_replies(comment)
        for word in words:
            p_word_catagory = np.array([i[word] for i in self.counts])
            p_word = p_word_catagory.sum() # getting appearences of this word
            bayes_factor = p_word_catagory*5/p_word
            #print(bayes_factor)
            #print(normalize)
            probabilities *= bayes_factor*normalize
        output = []
        for i in np.argwhere(probabilities>1):
            output.append(self.classes[i[0]])
        return output

    def save(self, path):
        df = pd.DataFrame.from_records(self.counts,self.classes)
        df.to_csv(path)

    def load(self, path):
        df = pd.read_csv(path)
        data = df.to_dict(orient='records')
        for index, i in enumerate(data):
            i = {key:val for key, val in i.items() if not pd.isna(val)}
            self.counts[index] = defaultdict(lambda: 1, i)
        
    def load_add(self, path):
        df = pd.read_csv(path)
        data = df.to_dict(orient='records')
        #print(data)
        for index, i in enumerate(data):
            for key, val in i.items():
                if not pd.isna(val) and (type(val) == int or type(val) == float):
                    #print(val)
                    self.counts[index][key] += val-4

    def __repr__(self):
        out = ""
        for i in range(len(self.counts)):
            out += self.training_subreddits[i][0]
            out += '\n'
            out += str(self.counts[i])
            out += '\n\n'
        return out

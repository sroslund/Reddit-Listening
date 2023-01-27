from collections import defaultdict
import datetime as dt

# p = political, r = relationships, c = classwork, j = joke, k = complaint

class NaiveBayes:
    def __init__(self):
        self.classes = ['p', 'r', 'c', 'j', 'k']

        # this defines the initial probability of each class
        self.counts = [defaultdict(4),defaultdict(4),defaultdict(4),defaultdict(4),defaultdict(4)]
        
        self.training_subreddits = [["politics"],["relationships"],["HomeworkHelp"],["memes"],["Negativity"]]

    def train(self, pushapi, reddit):

        start_date=int(dt.datetime(2022, 9, 23).timestamp()) #modify this line to change the time we are searching

        

    def save(self):
        pass


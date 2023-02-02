import bayes
import praw
from pmaw import PushshiftAPI

agent = "r/ucsc data scrapper"
reddit = praw.Reddit(client_id="edhlsn3xRnsuK_U1n_qOfQ", client_secret="DyZHk39I6aJmKweONoS3IFU2kmyHIQ", user_agent = agent)

api = PushshiftAPI()

model = bayes.NaiveBayes()
model.train(api, reddit)
model.save('./trained_models/attempt1.csv')
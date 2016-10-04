import pickle
from fitLDAModel2Corpus import TopicModel
import sentenceSelectionModel
import displaySummary 


def getFederalDockets():
    dockets = ['APHIS-2006-0044','CPSC-2012-0050', 
               'APHIS-2006-0085', 'APHIS-2009-0017']
    return dockets

def getComments():
    regulations = dict()
    comments = list()
    dockets = getFederalDockets()
    for docket in dockets:
        file_name = 'example_data/' + docket + '.pickle'
        cmts = pickle.load(open(file_name, 'rb'))
        regulations[docket] = cmts
        comments.extend(cmts)
    return regulations, comments


def main(num_topics=15):
    
    regulations, comments = getComments()
    
    topicModel = TopicModel(num_topics)
    topicModel.fit(comments)
    
    
    summary_data = sentenceSelectionModel.main(regulations, topicModel)
    
    displaySummary.showSummaries(summary_data)



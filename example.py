import pickle
from topicModel import TopicModel
from documentSummaries import DocumentSummaries

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

    for docket_id, document in regulations.iteritems():
        docSummaries = DocumentSummaries(topicModel, num_dominant_topics=3, number_of_sentences=4)
        docSummaries.summarize(document)
        print docket_id
        docSummaries.display()


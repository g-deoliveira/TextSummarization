import pickle
import fitLDAModel2Corpus as fit
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
    
    model = fit.fitModel(comments, num_topics, 
        min_word_count=10, top_most_common_words=10)

    summary_data = sentenceSelectionModel.main(regulations, 
        model.lda, model.dictionary, fit.tokenizer, model.bigramizer)

    displaySummary.showSummaries(summary_data)

    return summary_data, model


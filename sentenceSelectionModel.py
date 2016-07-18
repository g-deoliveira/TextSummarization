import fitLDAModel2Corpus as fit
import nltk.data
import numpy as np

class topicSummary(object):

    def __init__(self, topic_id, terms, weights, sentences):
        self.topic_id = topic_id
        self.terms = terms
        self.weights = weights
        self.sentences = sentences

    def __str__(self):
        if self.sentences is None or len(self.sentences) == 0:
            return 'topic does not have any sentences'
        text = str()
        
        for t in self.terms:
            text += '{:s},'.format(t)
        text += '\n'
        
        for w in self.weights:
            text += '{:5.4f},'.format(w)
        text += '\n'
        for sentence in self.sentences:
            text += sentence[2] + ' '
        return text


def splitIntoSentences(documents, MIN_SENTENCE_LENGTH = 8, MAX_SENTENCE_LENGTH = 25):
    # splits a document into sentences. Discards sentences that are too short or too long.
    # input: a list of documents
    # output: a list of lists of tuples (sentence #, sentence)
    #
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    #
    # discard sentences that have fewer than 10 tokens
    # according to
    # https://strainindex.wordpress.com/2008/07/28/the-average-sentence-length/
    # discard sentences that are longer than appx 20 tokens
    #
    sentence_groups = list()
    for document in documents:
        sentences = sentence_detector.tokenize(document)
        sentence_group = list()
        for k, sentence in enumerate(sentences):
            length = len(sentence.split())
            if (length > MIN_SENTENCE_LENGTH and length < MAX_SENTENCE_LENGTH):
                sentence_group.append((k, sentence))
        sentence_groups.append(sentence_group)
    return sentence_groups


def sentenceDistributions(sentence_groups, dictionary, lda):
    # computes topic distributions for each sentence
    # output: list of lists
    # each list corresponds to a document and stores a tuple per sentence
    # the 1st element is the sentence number in the group
    # the 2nd element is a tuple of (topic_id, weight)
    distributions = list()
    get_bow = dictionary.doc2bow
    get_document_topics = lda.get_document_topics
    tokenize = fit.tokenizer
    for sentences in sentence_groups:
        sentence_distributions = list()
        for k, sentence in sentences:
            tkns = tokenize(sentence)
            if tkns is None:
                continue
            bow = get_bow(tkns)
            dist = get_document_topics(bow)
            # this is to get list of dominant indices in decreasing order
            #dist.sort(key=lambda x: x[1], reverse=True)
            #dist = [d[0] for d in dist]
            #
            # this is to get the dominant index only (not a list)
            try:
                dist = max(dist, key=lambda x: x[1])
            except ValueError, ve:
                continue
            sentence_distributions.append((k, dist))
        distributions.append(sentence_distributions)
    return distributions

def filterSentencesByTopic(distributions, topic_id):
    # get only the document/sentence numbers in distributions
    # that match the given topic_id
    #
    # the output is a list of triplets:
    # (document number, sentence number, weight)
    filtered_by_topic_id = list()
    for k, distribution in enumerate(distributions):
        filtered = [d for d in distribution if d[1][0] == topic_id]
        for item in filtered:
            filtered_by_topic_id.append((k, item[0], item[1][1]))
    return filtered_by_topic_id


def innerProduct(bow1, bow2):
    keys1 = set(bow1)
    keys2 = set(bow2)
    keys = keys1.intersection(keys2)
    if not keys:
        return 0.0
    inner_product = 0.0
    for key in keys:
        inner_product += bow1[key] * bow2[key]
    sum1 = 0.0
    sum2 = 0.0
    for v in bow1.itervalues():
        sum1 += v*v
    for v in bow2.itervalues():
        sum2 += v*v
    inner_product /= np.sqrt(sum1 * sum2)
    return inner_product



def cosineSimilarity(sentence_bow, list_of_sentence_bow):
    # check similarity between sentence_bow and items in list_of_sentence_bow
    for bow in list_of_sentence_bow:
        inner_product = innerProduct(sentence_bow, bow)
        if inner_product >= 0.66:
            return True
    return False




def sentencesPerTopic(dominant_topic_id, sentence_groups, 
                      distributions, dictionary, number_of_sentences):
    
    # get only the document/sentence numbers that are dominated
    # by the dominant topic
    filtered_by_topic_id = filterSentencesByTopic(distributions, dominant_topic_id)
    
    # if the filtered list finds no sentences, move on
    # this event is highly unlikely so this is bad!!
    if len(filtered_by_topic_id) == 0:
        return
    
    # loop until you have collected number_of_sentences sentences
    # sometimes there may be no match between sentence_no and dominant_topic
    sn = 0
    
    similarity_list = list()
    top_sentences = list()
    
    while (len(top_sentences) < number_of_sentences and sn < len(distributions)):
        
        filtered_by_sn = [f for f in filtered_by_topic_id if f[1] == sn]
        sorted_by_weight = sorted(filtered_by_sn, key=lambda x: x[2], reverse=True)
        
        if len(sorted_by_weight) == 0:
            if sn == len(distributions) - 1:
                print 'No results in filtered set for sentence:', sn
            sn += 1
            continue
        
        document_id = sorted_by_weight[0][0]
        passage = sentence_groups[document_id]
        sentence = [p[1] for p in passage if p[0] == sn]
        assert len(sentence) == 1
        sentence = sentence[0]
        sentence_bow = dictionary.doc2bow(tkns for tkns in sentence.lower().split())
        sentence_bow = dict(sentence_bow)

        if cosineSimilarity(sentence_bow, similarity_list):
            sn += 1
            continue
        similarity_list.append(sentence_bow)
        top_sentences.append((document_id, sn, sentence))
        sn += 1
    return top_sentences


def dominantTopics(corpus, lda, num_dominant_topics):
    
    # get topic weight matrix using lda.inference
    # the matrix has dimensions (num documents) x (num topics)
    inference = lda.inference(corpus)
    inference = inference[0] # the inference is a tuple, need the first term
    num_topics = lda.num_topics
    
    # find dominant topics across documents (vertical sum)
    column_sum_of_weights = np.sum(inference, axis=0)
    sorted_weight_indices = np.argsort(column_sum_of_weights)
    idx = np.arange(num_topics - num_dominant_topics, num_topics)
    dominant_topic_ids = sorted_weight_indices[idx]
    # the dominant_topic_ids store the ids in descending order of dominance
    dominant_topic_ids = dominant_topic_ids[::-1]
    
    return dominant_topic_ids.tolist()


def sentenceSelection(lda, dominant_topic_ids, sentence_groups, 
                      distributions, dictionary, num_dominant_topics=5, 
                      number_of_sentences=5, verbose=False):
    
    results_per_docket = dict()
    results_per_docket['number_of_documents'] = len(sentence_groups)
    results_per_docket['dominant_topic_ids'] = dominant_topic_ids
    
    for dtid in dominant_topic_ids:
        results_per_topic = dict()
        
        top_sentences = sentencesPerTopic(dtid, sentence_groups, distributions, 
                                          dictionary, number_of_sentences)
        
        topic_terms = lda.show_topic(dtid)
        terms = [t[0] for t in topic_terms]
        weights = [w[1] for w in topic_terms]
        
        ts = topicSummary(topic_id = dtid, terms=terms, 
                          weights=weights, sentences=top_sentences)
        
        if verbose:
            displaySummary(top_sentences, topic_terms)
        
        results_per_docket[dtid] = ts
    
    return results_per_docket


def main(regulations, model):
    # the tokenizer should be the same method used to fit the LDA
    # the bigramizer should be the same method used to fit the LDA
    num_dominant_topics=5
    number_of_sentences=5
    summaries = {}
    
    lda = model.lda
    dictionary = model.dictionary
    tokenizer = fit.tokenizer
    bigramizer = model.bigramizer
    
    for docket_id in regulations:
        
        documents = regulations[docket_id]
        
        tokens = [tokenizer(document) for document in documents]
        tokens = [bigramizer[tkn] for tkn in tokens]
        corpus = [dictionary.doc2bow(tkn) for tkn in tokens]
        
        dominant_topic_ids = dominantTopics(corpus, lda, num_dominant_topics)
        
        sentence_groups = splitIntoSentences(documents)
        
        distributions = sentenceDistributions(sentence_groups, dictionary, lda)
        
        summary_data = sentenceSelection(lda, dominant_topic_ids, sentence_groups, 
            distributions, dictionary, num_dominant_topics, number_of_sentences, False)
        
        summaries[docket_id] = summary_data
    return summaries


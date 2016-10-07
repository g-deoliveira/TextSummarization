from topicModel import tokenizer
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



class DocumentSummaries(object):
    '''
    Generates summaries for a set of documents given a topic model.
    
    Parameters
    ----------
    model: TopicModel
        a TopicModel object trained on a corpus of documents
      
    num_dominant_topics: int, default: 5
        The number of dominant topics - corresponds to the
        number of summaries that are generated.

    number_of_sentences: int, default: 5
        The number of sentences per summary. 
        
    Attributes
    ----------
    summary_data: dictionary

    
    '''
    
    def __init__(self, model, num_dominant_topics=5, number_of_sentences=5):
        # the bigramizer should be the same object that was trained in TopicModel
        self.num_dominant_topics = num_dominant_topics
        self.number_of_sentences= number_of_sentences
        self.lda = model.lda
        self.dictionary = model.dictionary
        self.bigramizer = model.bigramizer
    
    
    def summarize(self, documents):
        
        tokens = [tokenizer(document) for document in documents]
        tokens = [self.bigramizer[tkn] for tkn in tokens]
        corpus = [self.dictionary.doc2bow(tkn) for tkn in tokens]
            
        self.dominant_topic_ids = self.getDominantTopics(corpus)
            
        self.sentence_groups = self.splitIntoSentences(documents)
            
        self.distributions = self.getSentenceDistributions()
            
        self.summary_data = self.sentenceSelection(verbose=False)
            
    
    def getDominantTopics(self, corpus):
    
        # get topic weight matrix using lda.inference
        # the matrix has dimensions (num documents) x (num topics)
        inference = self.lda.inference(corpus)
        inference = inference[0] # the inference is a tuple, need the first term
        num_topics = self.lda.num_topics
        
        # find dominant topics across documents (vertical sum)
        column_sum_of_weights = np.sum(inference, axis=0)
        sorted_weight_indices = np.argsort(column_sum_of_weights)
        idx = np.arange(num_topics - self.num_dominant_topics, num_topics)
        dominant_topic_ids = sorted_weight_indices[idx]
        # the dominant_topic_ids store the ids in descending order of dominance
        dominant_topic_ids = dominant_topic_ids[::-1]
        
        return dominant_topic_ids.tolist()

    
    def splitIntoSentences(self, documents, MIN_SENTENCE_LENGTH = 8, MAX_SENTENCE_LENGTH = 25):
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
    
    
    def getSentenceDistributions(self):
        # computes topic distributions for each sentence
        # output: list of lists
        # each list corresponds to a document and stores a tuple per sentence
        # the 1st element is the sentence number in the group
        # the 2nd element is a tuple of (topic_id, weight)
        distributions = list()
        get_bow = self.dictionary.doc2bow
        get_document_topics = self.lda.get_document_topics
        for sentences in self.sentence_groups:
            sentence_distributions = list()
            for k, sentence in sentences:
                tkns = tokenizer(sentence)
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
    
    
    def sentenceSelection(self, verbose=False):
        
        results_per_docket = dict()
        results_per_docket['number_of_documents'] = len(self.sentence_groups)
        results_per_docket['dominant_topic_ids'] = self.dominant_topic_ids
        
        for dtid in self.dominant_topic_ids:
            results_per_topic = dict()
            
            top_sentences = self.sentencesPerTopic(dtid)
            
            topic_terms = self.lda.show_topic(dtid)
            terms = [t[0] for t in topic_terms]
            weights = [w[1] for w in topic_terms]
            
            ts = topicSummary(topic_id = dtid, terms=terms, 
                              weights=weights, sentences=top_sentences)
            
            if verbose:
                displaySummary(top_sentences, topic_terms)
            
            results_per_docket[dtid] = ts
        
        return results_per_docket


    def sentencesPerTopic(self, dominant_topic_id):
        
        # get only the document/sentence numbers that are dominated
        # by the dominant topic
        filtered_by_topic_id = self.filterSentencesByTopic(dominant_topic_id)
        
        # if the filtered list finds no sentences, move on
        # this event is highly unlikely so this is bad!!
        if len(filtered_by_topic_id) == 0:
            return
        
        # loop until you have collected number_of_sentences sentences
        # sometimes there may be no match between sentence_no and dominant_topic
        sn = 0
        
        similarity_list = list()
        top_sentences = list()
        
        while (len(top_sentences) < self.number_of_sentences and sn < len(self.distributions)):
            
            filtered_by_sn = [f for f in filtered_by_topic_id if f[1] == sn]
            sorted_by_weight = sorted(filtered_by_sn, key=lambda x: x[2], reverse=True)
            
            if len(sorted_by_weight) == 0:
                if sn == len(self.distributions) - 1:
                    print 'No results in filtered set for sentence:', sn
                sn += 1
                continue
            
            document_id = sorted_by_weight[0][0]
            passage = self.sentence_groups[document_id]
            sentence = [p[1] for p in passage if p[0] == sn]
            assert len(sentence) == 1
            sentence = sentence[0]
            sentence_bow = self.dictionary.doc2bow(tkns for tkns in sentence.lower().split())
            sentence_bow = dict(sentence_bow)

            if cosineSimilarity(sentence_bow, similarity_list):
                sn += 1
                continue
            similarity_list.append(sentence_bow)
            top_sentences.append((document_id, sn, sentence))
            sn += 1
        return top_sentences
    
    def filterSentencesByTopic(self, topic_id):
        # get only the document/sentence numbers in distributions
        # that match the given topic_id
        #
        # the output is a list of triplets:
        # (document number, sentence number, weight)
        filtered_by_topic_id = list()
        for k, distribution in enumerate(self.distributions):
            filtered = [d for d in distribution if d[1][0] == topic_id]
            for item in filtered:
                filtered_by_topic_id.append((k, item[0], item[1][1]))
        return filtered_by_topic_id
    
    
    def display(self):
        '''
        '''
        print 'The dominant topics in descending order are:'
        for dtid in self.dominant_topic_ids:
            print dtid, 
        print ''
        
        for k in range(self.num_dominant_topics):
            dtid = self.dominant_topic_ids[k]
            topicSummary = self.summary_data[dtid]
            terms = topicSummary.terms
            weights = topicSummary.weights
            num_terms = len(terms)
            sentences = topicSummary.sentences
            
            print '\nTopic {:d}'.format(dtid)
            print 'The top {:d} terms and corresponding weights are:'.format(num_terms)
            for term, weight in zip(terms, weights):
                print ' * {:s} ({:5.4f})'.format(term, weight)
            
            print '\n\nThe selected sentences are:',
            n_sentences = len(sentences)
            for j in range(n_sentences):
                item = sentences[j]
                print '{:d},'.format(item[0]),
            print ' '
            for j in range(n_sentences):
                item = sentences[j]
                sentence = item[2]
                print sentence
            print


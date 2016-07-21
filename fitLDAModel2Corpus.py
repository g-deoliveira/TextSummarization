import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from gensim import models, corpora
from gensim.models import Phrases
import numpy as np


wnl = WordNetLemmatizer()
lemmatizer = wnl.lemmatize

def tokenizer(document):
    """
    input: a string
    output: a list of strings
    converts a string into tokens and performs the following steps:
    1. elimaintes non alphabetical characters
    2. converts to lower case
    3. lemmatizes using the nltk.stem.WordNetLemmatizer
    4. splits into tokens
    """
    text = re.sub('[^a-zA-Z]', ' ', document)
    tokens = text.lower().split()
    tokens = [lemmatizer(tkn) for tkn in tokens]
    return tokens


class fitModel(object):

    def __init__(self, documents, num_topics=100, min_word_count=20, 
                 top_most_common_words=10, min_row_length=40, 
                 max_row_length=1000, random_state=None):
        '''
        input:
          corpus of documents: list of strings
          num_topics: input parameter to LDA
          min_word_count: if a token has fewer than min_word_count occurences in the entire corpus, 
                          then it will be pruned from the processed corpus
          top_most_common_words: if a token is within the top_most_common_words, 
                                 then it will be pruned from the processed corpus
          min_row_length: if the number of tokens within a processed document 
                          is fewer than min_row_length, then the document is excluded
          max_row_length: if the number of tokens within a processed document 
                          is greater than max_row_length, then the document is excluded
          random_state: used for setting the seed of the LDA calculation
        
        This module preprocesses a corpus of documents and runs
        Latent Dirichlet Allocation (LDA) on a corpus of documents.
        
        output:
          the trained Gensim bigramizer
          the tokens (list of list of strings)
          the dictionary (id to token mapping)
          the corpus (bag of words vectorization of the tokens)
          the Gensim LDA object
          the dominant topic ids (list, decreasing order of dominance)
        '''
        self.num_topics = num_topics
        self.min_word_count = min_word_count
        self.top_most_common_words = top_most_common_words
        
        assert max_row_length > min_row_length, \
               "max_row_length must be greater than min_row_length"
        self.min_row_length = min_row_length
        self.max_row_length = max_row_length
        
        # natural language processing
        self.stop_words = self.getEnglishStopWords()
        self.bigramizer = Phrases()#min_count = min_word_count)
        
        # tokens, dictionary, corpus for LDA
        self.tokens = self.preProcessCorpus(documents)
        self.dictionary = corpora.Dictionary(self.tokens)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokens]
        
        self.lda = self.getLDA(dictionary=self.dictionary, 
                               corpus=self.corpus, 
                               num_topics=self.num_topics, 
                               random_state=random_state)
        
        self.num_dominant_topics=min(10, num_topics)
        self.dominant_topic_ids = self.getDominantTopics(self.corpus, 
                                                         self.lda, 
                                                         self.num_dominant_topics)


    def __str__(self):
        description = ("topic model: token length={0:,d}, dictionary length={1:,d}, "
                       "num_topics={2:,d}, min_word_count={3:,d}, "
                       "top_most_common_words={4:,d}, min_row_length={5:,d}, "
                       "max_row_length={6:,d}")
        return description.format(len(self.tokens), 
                                  len(self.dictionary),
                                  self.num_topics, 
                                  self.min_word_count, 
                                  self.top_most_common_words, 
                                  self.min_row_length, 
                                  self.max_row_length)

    @staticmethod
    def getEnglishStopWords():
        '''
        returns a set of stop words for NLP pre-processing
        from nltk.corpus.stopwords()
        Also, some words and letters are added to the set,
        such as "please", "sincerely", "u", etc...
        '''
        stop_words = set(stopwords.words("english"))
        
        stop_words.add('please')
        stop_words.add('would')
        stop_words.add('use')
        stop_words.add('also')
        stop_words.add('thank')
        stop_words.add('sincerely')
        stop_words.add('regards')
        stop_words.add('hi')
        stop_words.add('hello')
        stop_words.add('greetings')
        stop_words.add('hey')
        stop_words.add('attachment')
        stop_words.add('attached')
        stop_words.add('attached_file')
        stop_words.add('see')
        stop_words.add('file')
        stop_words.add('comment')
        for item in 'abcdefghijklmnopqrstuvwxyz':
            stop_words.add(item)
        return stop_words
    
    
    @staticmethod
    def getFrequencies(tokens):
        """
        input: tokens, a list of list of tokens
        output: a collections.Counter() object that contains token counts
        """
        frequencies = Counter()
        for row in tokens:
            frequencies.update(row)
        return frequencies
    
    @staticmethod
    def getLowFreqWords(frequencies, countCutOff):
        """
        input: 
          frequencies: a collections.Counter() object
          countCutOff: the minimum frequency below which tokens are added to the set
                       of low frequency tokens
        """
        lowFreqTokens = set()
        for token, freq in frequencies.iteritems():
            if freq <= countCutOff:
                lowFreqTokens.add(token)
        return lowFreqTokens


    def preProcessCorpus(self, documents, min_word_count=None, 
                         top_most_common_words=None, min_row_length=None, 
                         max_row_length=None):
        '''
        this function pre-processes the documents and converts them into a list of list of tokens
        
        input: 
          documents: a list of strings (each string represents a document)
          min_word_count: if a token has fewer than min_word_count occurences in the entire corpus, 
                          then it will be pruned from the processed corpus
          top_most_common_words: if a token is within the top_most_common_words, 
                                 then it will be pruned from the processed corpus
          min_row_length: if the number of tokens within a processed document 
                          is fewer than min_row_length, then the document is excluded
          max_row_length: if the number of tokens within a processed document 
                          is greater than max_row_length, then the document is excluded
        output:
          a list of list of tokens
          the trained Gensim bigram model which concatenates bigrams, eg "New_York"
        '''
        if min_word_count is None:
            min_word_count = self.min_word_count
        if top_most_common_words is None:
            top_most_common_words = self.top_most_common_words
        if min_row_length is None:
            min_row_length = self.min_row_length
        if max_row_length is None:
            max_row_length = self.max_row_length
        
        tokens = [tokenizer(document) for document in documents]
        
        # exclude comments that are longer than max_row_length
        tokens = [tkn for tkn in tokens if len(tkn) < max_row_length]
        
        # train Gensim Phrases model for bigrams
        self.bigramizer.add_vocab(tokens)
        
        # apply Gensim Phrases model to generate bigrams
        tokens = [self.bigramizer[tkn] for tkn in tokens]
        
        # exclude stop words
        tokens = [[t for t in tkn if t not in self.stop_words] for tkn in tokens]
        
        # exclude tokens that are shorter than min_row_length
        tokens = [tkn for tkn in tokens if len(tkn) > min_row_length]
        
        # calculate token frequencies to exclude low and high frequency tokens
        freqs = self.getFrequencies(tokens)
        low_freq_tokens = set(x[0] for x in freqs.iteritems() if x[1] < min_word_count)
        high_freq_tokens = [word[0] for word in freqs.most_common(top_most_common_words)]
        
        tokens =  [[t for t in tkn if t not in low_freq_tokens] for tkn in tokens]
        tokens =  [[t for t in tkn if t not in high_freq_tokens] for tkn in tokens]
        
        print '\nnumber of low frequency tokens pruned = {:d}'\
              .format(len(low_freq_tokens))
        print 'min_word_count = {:d}, top_most_common_words = {:d}'\
              .format(min_word_count, top_most_common_words)
        print 'number of high frequency tokens pruned = {:d}'\
              .format(len(high_freq_tokens))
        print 'tokens = {:d} rows'.format(len(tokens))
        return tokens


    def getLDA(self, dictionary=None, corpus=None, num_topics=None, 
               random_state=None):
        # get LDA for dictionary_all and corpus_all
        
        if dictionary is None:
            dictionary = self.dictionary
        if corpus is None:
            corpus = self.corpus
        if num_topics is None:
            num_topics = self.num_topics
        
        lda = models.ldamodel.LdaModel(corpus=corpus, 
                                       alpha='auto', 
                                       id2word=dictionary, 
                                       num_topics=num_topics)
        return lda


    def getDominantTopics(self, corpus, lda, num_dominant_topics=None):
        
        if corpus is None:
            corpus = self.corpus
        if lda is None:
            lda = self.lda
        if num_dominant_topics is None:
            num_dominant_topics = self.num_dominant_topics
        
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
        
        # convert from numpy array to list and return
        return dominant_topic_ids.tolist()

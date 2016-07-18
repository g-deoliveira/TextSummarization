



def showSummary(summary, num_topics=5, num_sentences=3):
    '''
    summary is a dictionary
    the keys-value pairs are:
        'dominant_topic_ids': list of dominant topic ids, [id1, ..., idn]
        'number_of_documents': integer
        id1: topicSummary object containing terms, weights and sentences
        ...
        idn: topicSummary object containing terms, weights and sentences
    '''
    dominant_topic_ids = summary['dominant_topic_ids']
    n_topics = min(len(dominant_topic_ids), num_topics)
    
    print 'The dominant topics in descending order are:'
    for dtid in dominant_topic_ids:
        print dtid, 
    print ''
    
    for k in range(n_topics):
        dtid = dominant_topic_ids[k]
        topicSummary = summary[dtid]
        terms = topicSummary.terms
        weights = topicSummary.weights
        num_terms = len(terms)
        sentences = topicSummary.sentences
        
        print '\nTopic {:d}'.format(dtid)
        print 'The top {:d} terms and corresponding weights are:'.format(num_terms)
        for term, weight in zip(terms, weights):
            print ' * {:s} ({:5.4f})'.format(term, weight)
        
        print '\n\nThe selected sentences are:',
        n_sentences = min(len(sentences), num_sentences)
        for j in range(n_sentences):
            item = sentences[j]
            print '{:d},'.format(item[0]),
        print ' '
        for j in range(n_sentences):
            item = sentences[j]
            sentence = item[2]
            print sentence
        print

def showSummaries(summaries, n_topics=5, n_sentences=4):
    for doc_id in summaries:
        print 'document:', doc_id, '-'*40
        summary = summaries[doc_id]
        showSummary(summary, n_topics, n_sentences)



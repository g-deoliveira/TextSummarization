# Extractive Text Summarization Using Topic Modelling

This library generates short, human-readable summaries of documents using topic modelling. It is geared towards creating summaries of corpuses of relatively short documents, such as comments on proposed government regulations, or online product reviews. The algorithm uses Latent Dirichlet Allocation to identify the dominant topics, then identifies sentences that reflect those topics and stitches them together.

The library consists of two main scripts: `topicModel.py` and `documentSummaries.py`. I have provided a set of sample documents in the `example_data` folder and the associated commands are provided in the `example.py` script. Briefly, the following reads the sample data and outputs a dictionary `regulations` and a list `comments`. The dictionary stores document identifiers and the corresponding list of documents, and `comments` stores the corresponding corpus of documents, ie a list that containing the union of all documents.

```python
regulations, comments = getComments()
```

Using scikit-learn style syntax, you initialize the topic model and fit it to the corpus of comments to compute the dominant topics:

```python
topicModel = TopicModel(num_topics=3)
topicModel.fit(comments)
```

The `TopicModel` object reads, preprocesses and vectorizes the list of documents, performs the LDA computation and identifies the dominant topics. Specifically, the following pre-processing steps are carried out on the text:
* stripping out punctuation and non-alphabetical characters
* tokenization
* lemmatization using the NLTK WordNetLemmatizer, [here](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.wordnet) 
* bi-grammization via Gensim, [here](https://radimrehurek.com/gensim/models/phrases.html)
* removal of stopwords using an augmented version of the NLTK English stopwords corpus, [here](http://www.nltk.org/nltk_data/)
* removal of low and high frequency tokens
* removal of documents that are too short and too long (this step is specifically geared towards public feedback or product reviews)

Once the dominant topics have been identified, summaries are computed for provided documents as follows:

```python
# generate and display the computed summary for each regulation
for docket_id, document in regulations.iteritems():
    docSummaries = DocumentSummaries(topicModel, num_dominant_topics=3, number_of_sentences=4)
    docSummaries.summarize(document)
    print docket_id
    docSummaries.display()
```
The DocumentSummaries.summarize method performs the following steps to extract the sumaries for a given topic id :
<ol>
<li> Pass the individual comments in the document to the LDA object to determine the distribution of topics for each comment.
<li> Filter out the topics whose dominant topic is not equal to the given topic id. What is left is a subset of topics that reflect the given topic.
<li> For each comment within this subset:
<ol> Split the comment up into sentences, using the NLTK sentence tokenizer, [here](http://www.nltk.org/nltk_data)
<li> Feed the sentences to the LDA object to determine the topic distribution of each sentence.
<li> Filter out the sentences whose dominant topic is not equal to the given topic id, as well as sentences that are too short or sentences that are too long. What is left is a subset of sentences that reflect the given topic.
</ol>
</ol>


I worked on this project while attending the [Insight Data Science Fellowship Program](http://insightdatascience.com/) and used it to create summaries of public feedback on government regulations. Thus the pickled sample files in the *example_data* folder contain public comments on federal regulations that I downloaded from the API at [regulations.gov](https://www.regulations.gov/).

Here are some sample results, for the CPSC-2012-0050 docket. The five dominant topics identified by the algorithm are:
```
13 0 7 10 5
```
The top ten terms and associated weights for topic 13 are:
```
 magnet (0.0209)
 product (0.0145)
 cpsc (0.0121)
 ban (0.0096)
 should_be (0.0072)
 food_supply (0.0059)
 many (0.0046)
 danger (0.0040)
 time (0.0040)
 like (0.0039)
```
The generated summary is:
> There are far more dangerous and non-educational products marketed towards children that pose a higher hazard risk than Magnet sets.
> The average household contains many dangerous chemicals, small  objects, and electric appliances which pose greater danger to humans of all ages  than do magnets.
> A child swallowing ball bearings or nails, or many other small metal items can be deadly, yet nobody would imagine banning these items.
> Link to a Time magazine article: http://healthland.time.com/2012/09/06/household-hazard-kids-swallowing-laundry-detergent-capsules/  Banning the sale of small magnets is impractical in the long run.

Note the sentences come from these comments in the original data set: 30, 20, 1586, 971, respectively. 

As you can tell, the docket CPSC-2012-0050 concerns magnets. In case you are curious, more information on this regulation can be found [here](https://www.regulations.gov/document?D=CPSC-2012-0050-0001).

While I applied this library to public feedback for federal regulations, this code can handle a more general corpus. The number of topics is a parameter that should be chosen carefully. I will add more information about this soon.

## Additional Information

I will provide a more in depth description of the following:
* the sentence selection algorithm
* determination of the number of topics for LDA

In the mean time, more information can be found on a deck, [here](http://commentstldr.com/presentation).

## Dependencies

* [Gensim:](https://github.com/RaRe-Technologies/gensim) `pip install -U gensim`
* [nltk:](http://www.nltk.org/) `run sudo pip install -U nltk`
  * `sudo python -m nltk.downloader -d /usr/share/nltk_data punkt`
  * `sudo python -m nltk.downloader -d /usr/share/nltk_data stopwords`

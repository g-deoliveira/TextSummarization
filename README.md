# Extractive Text Summarization Using Topic Modelling

This library generates short, human-readable summaries of documents using topic modelling. The algorithm uses Latent Dirichlet Allocation to identify latent topics throughout a corpus of documents. Then for each individual document, the sentences that reflect the dominant topics are selected and stitched together.

I have provided a set of sample documents in the *example_data* folder and the associated commands are provided in the example.py script. Briefly:

```python
    regulations, comments = getComments()
    
    model = fit.fitModel(comments, num_topics)
    
    summary_data = sentenceSelectionModel.main(regulations, model)
    
    displaySummary.showSummaries(summary_data)
```

The document preprocessing, bag-of-words vectorization and LDA computation is done in the `fitLDAModel2Corpus` module. The sentence selection is carried out in `sentenceSelectionModel`. The input variable to the `fitLDAModel2Corpus` module is `comments`, which stores a list of strings that each represent a document, and the number of topics. 

The document preprocessing involves the usual Natural Language Processing steps, including:
* lemmatization using the NLTK WordNetLemmatizer [here](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.wordnet) 
* bi-gramming via Gensim [here](https://radimrehurek.com/gensim/models/phrases.html)
* removal of stopwords using an augmented version of the NLTK English stopwords corpus [here](http://www.nltk.org/nltk_data/)
* removal of low and high frequency tokens

I worked on this project while attending the [Insight Data Science Fellowship Program](http://insightdatascience.com/) and used it to create summaries of public feedback on government regulations. Thus the pickled sample files in the *example_data* folder contain public comments on federal regulations that I downloaded from the API at [regulations.gov](https://www.regulations.gov/).

Here are some sample results, for the CPSC-2012-0050 docket. The dominant topics identified by the algorithm are:
```
13 0 7 10 5
```
The top terms and associated weights for topic 13 are:
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

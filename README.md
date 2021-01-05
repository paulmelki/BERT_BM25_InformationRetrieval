# ***Homemade* BERT-based Search Engine**
#### Anh-Dung LE & Paul MELKI ([Toulouse School of Economics](https://www.tse-fr.eu/ ))

This project aims at implementing a BERT-based working search engine. :mag_right: :satellite:

The project is inspired by the recent work by **R. Noguiera and K. Cho (2019), [*Passage Re-ranking with BERT*](https://arxiv.org/pdf/1901.04085.pdf)**, which shows that language models are particularly useful for information retrieval, the main task implemented by a search engine. It also aims at comparing different models and different architecture (*END-TO-END* vs. *BM25 + BERT*). 

It is implemented as part of the course *"Mathematics of Deep Learning Algorithms, Part II*, at Toulouse School of Economics.

---

# **Part I: Building a Corpus**

We start our project by building a textual corpus which will form our articles' 
"database". That is, it will be the database from which the answers to our queries will be retrieved. Due to network and computational constraints, we work on a rather small corpus: a part (only 233 MB, compressed) of a very recent English Wikipedia dump (uploaded online on 20/12/2020), formed of articles whose titles start with the letter "A", which can be retrieved from [**here**](https://dumps.wikimedia.org/enwiki/20201220/). 

The data downloaded is in XML format, and contains many XML tags and links that need to be cleaned out in order to obtain the raw text of each article. Thankfully, the `gensim` library offers such a tool in its `WikiCorpus` class implemented in the `gensim.corpora` module. 

For this task, we implement a function `make_corpus()` that iteratively reads every article in our downloaded XML dump and saves its text into a separate `.txt` file in a local directory. This task needs to be only implemented once. Any subsequent runs, need only to use the function `read_corpus()` that reads all the text files from the local directory and saves each Wikipedia article as an element in a list. 

The code and further details related to implementation can be found in the project's Jupyter Notebook. 

---

# **Part II: Experiments**

Now that we have a valid corpus, we begin experimenting with different IR methods and noting the results on queries whose correct results we know (either from perosnal knowledge or from common sense). In particular, we compare the famous *Okapi BM25* method to a simple *BERT* implementation.

## **Information Retrieval using BM25**

**BM25** is a TF-IDF method, that retrieves the article that has the highest score based on the query given and the following formula: 

Given, a document <img src="https://render.githubusercontent.com/render/math?math=D"> and a <img src="https://render.githubusercontent.com/render/math?math=Q"> that contains keywords <img src="https://render.githubusercontent.com/render/math?math=q_1,..., q_n">, we define the BM25 score of the document <img src="https://render.githubusercontent.com/render/math?math=D"> as:


<img src="https://render.githubusercontent.com/render/math?math=score(D, Q) = \sum_{i = 1}^n IDF(q_i) \cdot \frac{TF(q_i, D) \cdot (k_1 + 1)}{TF(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{avgdl} \right)}">


where: 
- <img src="https://render.githubusercontent.com/render/math?math=TF(q_i, D)"> is the *text frequency* of keyword <img src="https://render.githubusercontent.com/render/math?math=q_i"> in document <img src="https://render.githubusercontent.com/render/math?math=D">,
- <img src="https://render.githubusercontent.com/render/math?math=IDF(q_i)"> is the *inverse document frequency* of keyword <img src="https://render.githubusercontent.com/render/math?math=q_i">, using the well-known definition,
- <img src="https://render.githubusercontent.com/render/math?math=|D|"> is the length of the document <img src="https://render.githubusercontent.com/render/math?math=D"> in words.
- <img src="https://render.githubusercontent.com/render/math?math=avgdl"> is the average document length in words in the whole corpus.
- <img src="https://render.githubusercontent.com/render/math?math=k_1"> and <img src="https://render.githubusercontent.com/render/math?math=b"> are free parameters that are chosen rather than estimated, and which are usually chosen as <img src="https://render.githubusercontent.com/render/math?math=k_1 \in [1.2, 2.0]"> and <img src="https://render.githubusercontent.com/render/math?math=b = 0.75">. These may also be chosen based on some advanced optimization.

After computing the BM25 score of each document, which gives the relevance of each document to the given query, we sort the documents in descending order from most relevant to least relevant.

On the implementation side, we use `Rank-BM25` library developed by Dorian Brown (https://github.com/dorianbrown/rank_bm25), and which implements different variants of the BM25 algorithm. It can be easily installed using `pip install rank-bm25`. 

## **Information Retrieval using BERT**
Following Nogueira and Cho's (2019) method, we try to implement BERT as a document re-ranker that will rank the relevance of the documents in the corpus with respect to a given query. 

As we know, BERT for classification tasks takes two sentences as input. Given a document $D$ and a query $Q$ that have been tokenized using a BERT tokenizer, we concatenate the query (Sentence 1) and the document (Sentence 2) together, separating them with a `[CLS]` classification token, and feed them to the original pre-trained BERT model implement as a binary classifier where the two classes are: 


<img src="https://render.githubusercontent.com/render/math?math=\begin{cases} 0 = \text{not relevant}, \\ 1 = \text{relevant} \end{cases}">


As such, BERT will return the probability of document <img src="https://render.githubusercontent.com/render/math?math=D"> being relevant to the query <img src="https://render.githubusercontent.com/render/math?math=Q">. Given a certain query <img src="https://render.githubusercontent.com/render/math?math=Q">, we apply this method on all documents <img src="https://render.githubusercontent.com/render/math?math=D_1, D_2, ..., D_n"> in the corpus and get a *relevance score* for each of them. The documents are then ranked by their obtained scores from most relevant to least relevant (similarly to BM25) and this will be the result of our information retrieval task.

As we know, the corpus on which BERT has been trained contains the **full English Wikipedia** (2,500M words) along with the BooksCorpus (800M words).

For this reason, we thought that we do not need to re-train and finetune BERT for our scoring task, since it has already "seen" the articles found in our corpus. Being trained on document-level corpus and not word-based ones, BERT would be able to idenitfy the connections between our queries and the articles available in the small corpus that we have.

Furthermore, finetuning BERT would require training again on query-answers data sets such as [**MSMARCO**](https://microsoft.github.io/msmarco/) or [**TREC-CAR**](https://trec.nist.gov/pubs/trec26/papers/Overview-CAR.pdf), which were used by Nogueira and Cho (2019) in their implementation. However, due to network constraints (downloading the huge data sets proved not possible) and computational constraints, as well as time constraints (according to Nogueira and Cho, finetuning BERT required more than 30 hours of training), we were unable to finetune it to our specific task. We assumed that it may provide good results 'out-of-the-box'. Unfortunately, experimental results have shown otherwise:

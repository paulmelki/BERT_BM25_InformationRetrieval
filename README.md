# ***Homemade* BERT-based Search Engine**
#### Anh-Dung LE & Paul MELKI ([Toulouse School of Economics](https://www.tse-fr.eu/ ))

This project aims at comparing two IR methods: BM25 and a BERT-based search engine. :mag_right: :satellite:

The project is inspired by the recent work by **R. Noguiera and K. Cho (2019), [*Passage Re-ranking with BERT*](https://arxiv.org/pdf/1901.04085.pdf)**, which shows that language models are particularly useful for information retrieval, the main task implemented by a search engine.

It is implemented as part of the course *"Mathematics of Deep Learning Algorithms, Part II*, at Toulouse School of Economics.

**Structure of this repository:**
- Folders:
    - `Assets`: folder containing screenshots of some obtained results, used in the following report.
    - `Read Corpus`: folder containing the corpus in textual format after it has been read from the raw corpus as described in the report. Each article is in its own separate `.txt` file.
    - `BM25_BERT_DungMelki.ipynb`: Jupyter Notebook containing all the implementation and commentaries on the results.

---

# **Part I: Building a Corpus**

We start our project by building a textual corpus which will form our articles' 
"database". That is, it will be the database from which the answers to our queries will be retrieved. Due to network and computational constraints, we work on a rather small corpus: a part (only 233 MB, compressed) of a very recent English Wikipedia dump (uploaded online on 20/12/2020), formed of articles whose titles start with the letter "A", which can be retrieved from [**here**](https://dumps.wikimedia.org/enwiki/20201220/). 

The data downloaded is in XML format, and contains many XML tags and links that need to be cleaned out in order to obtain the raw text of each article. Thankfully, the `gensim` library offers such a tool in its `WikiCorpus` class implemented in the `gensim.corpora` module. 

For this task, we implement a function `make_corpus()` that iteratively reads every article in our downloaded XML dump and saves its text into a separate `.txt` file in a local directory. This task needs to be only implemented once. Any subsequent runs, need only to use the function `read_corpus()` that reads all the text files from the local directory and saves each Wikipedia article as an element in a list. 

The code and further details related to implementation can be found in the project's Jupyter Notebook. 

In this repository we provide: 
- A link to the original raw Wikipedia dump file (in `xml.bz2` format) which can be used to create the corpus from scratch using the function `make_corpus()`. We made the file available on our Google Drive, [**here**](https://drive.google.com/file/d/19UstaNTXard1UHjG0DWY1A2f2HhdwbBM/view?usp=sharing).
- Alternatively, the created textual corpus in the subfolder `Read Corpus` which contains 1000 articles, each article in its own separate `.txt` file.

---

# **Part II: Experiments**

Now that we have a valid corpus, we begin experimenting with different IR methods and noting the results on queries whose correct results we know (either from perosnal knowledge or from common sense). In particular, we compare the famous *Okapi BM25* method to a simple *BERT* implementation.

## **Information Retrieval using BM25**

**BM25** is a TF-IDF method, that retrieves the article that has the highest score based on the query given and the following formula: 

Given, a document <img src="https://render.githubusercontent.com/render/math?math=D"> and a <img src="https://render.githubusercontent.com/render/math?math=Q"> that contains keywords <img src="https://render.githubusercontent.com/render/math?math=q_1,..., q_n">, we define the BM25 score of the document <img src="https://render.githubusercontent.com/render/math?math=D"> as:


<img src="https://render.githubusercontent.com/render/math?math=score(D, Q) = \sum_{i = 1}^n IDF(q_i) \cdot \frac{TF(q_i, D) \cdot (k_1 + 1)}{TF(q_i, D) + k_1 \cdot \left( 1 - b + b \cdot \frac{|D|}{avgdl} \right)}">


where: 
- <img src="https://render.githubusercontent.com/render/math?math=TF(q_i, D)"> is the *text frequency* of keyword <img src="https://render.githubusercontent.com/render/math?math=q_i"> in document <img src="https://render.githubusercontent.com/render/math?math=D">,
- <img src="https://render.githubusercontent.com/render/math?math=IDF(q_i)"> is the *inverse document frequency* of keyword <img src="https://render.githubusercontent.com/render/math?math=q_i">, using the well-known definition,
- <img src="https://render.githubusercontent.com/render/math?math=|D|"> is the length of the document <img src="https://render.githubusercontent.com/render/math?math=D"> in words.
- <img src="https://render.githubusercontent.com/render/math?math=avgdl"> is the average document length in words in the whole corpus.
- <img src="https://render.githubusercontent.com/render/math?math=k_1"> and <img src="https://render.githubusercontent.com/render/math?math=b"> are free parameters that are chosen rather than estimated, and which are usually chosen as <img src="https://render.githubusercontent.com/render/math?math=k_1 \in [1.2, 2.0]"> and <img src="https://render.githubusercontent.com/render/math?math=b = 0.75">. These may also be chosen based on some advanced optimization.

After computing the BM25 score of each document, which gives the relevance of each document to the given query, we sort the documents in descending order from most relevant to least relevant.

On the implementation side, we use `Rank-BM25` library developed by Dorian Brown (https://github.com/dorianbrown/rank_bm25), and which implements different variants of the BM25 algorithm. It can be easily installed using `pip install rank-bm25`. 

#### **Some Results**
As we know the topics of some of the articles included, we implement some queries about these topics and see whether their relevant articles are returned. Some of these topics included:
- Autism
- Anarchism 
- ATM 

We first try to implement some simple queries that include only the title of the article, and see if the relevant article is returned. The results are shown in the below screenshots:

![BM25 Autism Results](https://github.com/paulmelki/bert-search-engine/blob/main/Assets/autism_bm25.PNG?raw=true)

![BM25 Anarchism Results](https://github.com/paulmelki/bert-search-engine/blob/main/Assets/anarchism_bm25.PNG?raw=true)

In the above two queries, we see that the results obtained are relevant. For the query about "autism", only the top result seems to be relevant. However, this could be simply due to the unavailability of more relevant articles in the small corpus we have, and not due to a problem in the method.

In the second query related to "anarchism", we see that the top three results are relevant indeed: the first one being an article exactly related to the topic, the second one being a related one and the third being about an author (Ayn Rand) who wrote many pieces and books about anarchism.

So far, BM25 looks like a useful method. However, we will see how it fails when the queries become more complicated, such as when they contain a question, a whole sentence, or an abbreviation. For example, the results of a query about an abbrevation and one that contains a full interrogative sentence are the following:

![BM25 ATM Results](https://github.com/paulmelki/bert-search-engine/blob/main/Assets/ATM_bm25.PNG?raw=true)

![BM25 WhatIsAnarchism Results](https://github.com/paulmelki/bert-search-engine/blob/main/Assets/whatisanarchism_bm25.PNG?raw=true)

In the above results, we can see clearly how BM25 fails as the queries become more complicated. This could mainly due to the fact that it is a pure TF-IDF method that does not prioritize keywords in the query over other words. 

This problem could be solved by combining BM25 with more advanced text processing techniques. Indeed, as we can see, we are not applying any advanced processing techniques such as lemmatization or keyword extraction. Further experiments will work on implementing these.


## **Information Retrieval using BERT**
Following Nogueira and Cho's (2019) method, we try to implement **BERT** as a document re-ranker that will rank the relevance of the documents in the corpus with respect to a given query. 

As we know, BERT for classification tasks takes two sentences as input. Given a document $D$ and a query $Q$ that have been tokenized using a BERT tokenizer, we concatenate the query (Sentence 1) and the document (Sentence 2) together, separating them with a `[CLS]` classification token, and feed them to the original pre-trained BERT model implement as a binary classifier where the two classes are: 


<img src="https://render.githubusercontent.com/render/math?math=\begin{cases} 0 = \text{not relevant}, \\ 1 = \text{relevant} \end{cases}">


As such, BERT will return the probability of document <img src="https://render.githubusercontent.com/render/math?math=D"> being relevant to the query <img src="https://render.githubusercontent.com/render/math?math=Q">. Given a certain query <img src="https://render.githubusercontent.com/render/math?math=Q">, we apply this method on all documents <img src="https://render.githubusercontent.com/render/math?math=D_1, D_2, ..., D_n"> in the corpus and get a *relevance score* for each of them. The documents are then ranked by their obtained scores from most relevant to least relevant (similarly to BM25) and this will be the result of our information retrieval task.

### **BERT without Finetuning**
As we know, the corpus on which BERT has been trained contains the **full English Wikipedia** (2,500M words) along with the BooksCorpus (800M words).

For this reason, we thought that we do not need to re-train and finetune BERT for our scoring task, since it has already "seen" the articles found in our corpus. Being trained on document-level corpus and not word-based ones, BERT would be able to idenitfy the connections between our queries and the articles available in the small corpus that we have.

Furthermore, finetuning BERT would require training again on query-answers data sets such as [**MSMARCO**](https://microsoft.github.io/msmarco/) or [**TREC-CAR**](https://trec.nist.gov/pubs/trec26/papers/Overview-CAR.pdf), which were used by Nogueira and Cho (2019) in their implementation. However, due to network constraints (downloading the huge data sets proved not possible) and computational constraints, as well as time constraints (according to Nogueira and Cho, finetuning BERT required more than 30 hours of training), we were unable to finetune it to our specific task. We assumed that it may provide good results 'out-of-the-box'. Unfortunately, experimental results have shown otherwise:

![BM25 BERT Results](https://github.com/paulmelki/bert-search-engine/blob/main/Assets/bertResults.PNG?raw=true)

Looking at the above results, we can clearly see that the none of the top 5 returned
documents is related to the simple query we searched for.

### **BERT Finetuned on MS-MARCO**
Another experiment we implement is using BERT model implemented by Nogueira and Cho (2019) which has been trained on the full MS-MARCO data set. This trained model is made available on their project's [GitHub page](https://github.com/nyu-dl/dl4marco-bert) and can be easily downloaded and imported. 

After importing the model, we also experiment on some queries with it, without obtaining any improvement in the results. For some reason (that we have not yet figured out), the retrieved articles are not relevant to the queries. This is in contradiction with the actual results obtained by the researchers, who have achieved state-of-the-art results. We will conduct further investigation into our implementation in order to find out the reason for these uncomforming results.

**Note**: Indeed, the main problem with experimenting with BERT is the long time it takes to obtain scores for the documents we have. While BM25 returns results in a couple of seconds, scoring using BERT takes around 15-20 minutes each time we try a certain query. This can be quite problematic for heavy testing and also for providing a valid information retrieval solution for users, for example.

# **Part III: Final Discussion**

In this short project, we conducted some small scale experiments comparing information retrieval results on simple queries using **BM25**  and **BERT-based** scoring. The results obtained have proven to be quite unexpected and we may refer the reasons to the following: 

- Due to network constraints, big corpuses and datasets such as a bigger subset (or the full) English Wikipedia or MS-MARCO could not be downloaded and used for implementing more reliable information retrieval solutions. Having to work with a considerably small corpus is liable to lead to weird results.
- Due to computational constraints, BERT could not be finetuned on larger scale datasets and this may be the main reason for the unexpected results we obtain. 
- An additional layer may need to be added to the BERT binary classifier in order to compute a better score for each document, thus leading to more accurate results.

### **What's next?**
Even if the deadline for the project has arrived, we do not consider this project as over. We consider that what has been implemented is just a part of a *work-in-progress* that will hopefully be continued beyond the limits of this course. The upcoming steps will consist mainly of figuring out the reasons for the unexpected results obtained and fixing them, as well as creating a faster and more efficient implementation of IR using BERT, and comparing with other IR methods that have not been discussed in this short report, such as End-to-End methods.


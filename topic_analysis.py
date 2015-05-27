
# coding: utf-8

# In[9]:

#get_ipython().magic('matplotlib inline') ipython notebooks inline plots
import psycopg2
import datetime
import re
import string
import os
from nltk.stem.snowball import EnglishStemmer
from gensim import corpora, models, similarities, logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import unContract
from nltk.tag.stanford import POSTagger


conn = psycopg2.connect(database="database", user="user", password="password")
cursor = conn.cursor()

# committee query

def billCommitteeQuery():
    cursor.execute("SELECT s.urlcache,s.content_en,s.time,c.party_id FROM hansards_statement s JOIN core_electedmember c ON s.member_id = c.id WHERE (urlcache LIKE '%committee%bill-c-%') AND (urlcache NOT LIKE '%the-chair%');")

def billHouseQuery():
    
    # Yes, this is a gross looking hack.  But it avoids concatenating strings.
    
    cursor.execute("SELECT hs.urlcache,hs.content_en,hs.time,c.party_id     FROM hansards_statement hs JOIN hansards_statement_bills sb ON sb.statement_id = hs.id     JOIN bills_billinsession bbs ON sb.bill_id = bbs.bill_id JOIN bills_bill bb ON bb.id=sb.bill_id     JOIN core_electedmember c ON hs.member_id = c.id WHERE     ((number LIKE 'C-2') AND (session_id LIKE '39-1') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-27') AND (session_id LIKE '39-1') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-35') AND (session_id LIKE '39-1') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-30') AND (session_id LIKE '39-1') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-2') AND (session_id LIKE '39-2') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-20') AND (session_id LIKE '39-2') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-2') AND (session_id LIKE '39-2') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-28') AND (session_id LIKE '39-2') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-31') AND (session_id LIKE '40-2') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-49') AND (session_id LIKE '40-3') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-32') AND (session_id LIKE '40-3') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-11') AND (session_id LIKE '41-1') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%')) OR     ((number LIKE 'C-18') AND (session_id LIKE '41-1') AND (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%'));")

def committeeQuery():
    cursor.execute("SELECT s.urlcache,s.content_en,s.time,c.party_id FROM hansards_statement s JOIN core_electedmember c ON s.member_id = c.id WHERE (urlcache LIKE '%committee%') AND (urlcache NOT LIKE '%the-chair%');")
    
def houseQuery():
    cursor.execute("SELECT s.urlcache,s.content_en,s.time,c.party_id FROM hansards_statement s JOIN core_electedmember c ON s.member_id = c.id WHERE (urlcache LIKE '%debates%') AND (urlcache NOT LIKE '%speaker%');")

def doStop(linelist):
    stop = stopwords.words('english')
    newline = []
    for i in linelist:
        if i not in stop:
            newline.append(i)
    return newline

def doProcedural(linelist):
    proc = ["chair","mr","madam","sir","ms","bill","motion","committee","monsieur","hon","order","chairman","minutes","move","time"]
    newline = []
    for i in linelist:
        if i not in proc:
            newline.append(i)
    return newline

class queryIterator(object):
    def __init__(self, sentence = False):
        self.paragraphInProgress = False
        self.workingParagraph = []
        self.sentence = sentence
        
    def __iter__(self):
        return self
    
    def sentenceHandler(self,res):
        result = list(res)
        result[1] = re.sub('<[^<]+?>', '', result[1])  # strip html
        x=unContract.unContract(result[1]) # unpack contractions
        
        for c in ["—", "-"]:
            x = x.replace(c," ")
        for c in string.punctuation:
            x = x.replace(c,"")
        for c in ["”", "“"]: # strip special unicode inverted commas, dash, etc.
            x = x.replace(c,"")
        x = x.lower() # lowercase
        return(x)
        
        
    def __next__(self):
        if self.sentence == False: # we will treat one document = one hansard statement
            res = cursor.fetchone()
            if res == None:
                raise StopIteration
            else:
                x = self.sentenceHandler(res)
                x=word_tokenize(x) # tokenize

                # optional stemmer
            
                #wnl = EnglishStemmer()
                #lemmed = []
                #for word in x:
                #    newword=wnl.stem(word)
                #    lemmed.append(newword)
        
                y=doStop(x) # remove stopwords and procedural words
                x=doProcedural(y)
                return (x)
        else: # we will treat one document = one sentence
            if self.paragraphInProgress==False:
                # this is a new paragraph, so fetch it
                res = cursor.fetchone()
                if res == None:
                    raise StopIteration
                else: # new paragraph fetched successfully
                    self.paragraphInProgress==True
                    x = self.sentenceHandler(res)
                    self.workingParagraph = sent_tokenize(x)

                    doc = self.workingParagraph.pop(0)
                    
                    doc=word_tokenize(doc) # tokenize
                    y=doStop(doc) # remove stopwords and procedural words
                    x=doProcedural(y)
                    
                    # before we end, check whether this was a one-sentence paragraph
                    
                    if len(self.workingParagraph)==0:
                        self.paragraphInProgress==False
                        
                    return (x)

            elif self.paragraphInProgress==True:
                # we have already started a paragraph with list of sentences, so pop the first one and yield it as tokens
                # if length becomes 0 at the end, reset the paragraphInProgress flag
                
                    doc = self.workingParagraph.pop(0)
                    
                    doc=word_tokenize(doc) # tokenize
                    y=doStop(doc) # remove stopwords and procedural words
                    x=doProcedural(y)
                    
                    # before we end, check whether this was a one-sentence paragraph
                    
                    if len(self.workingParagraph)==0:
                        self.paragraphInProgress==False
                        
                    return (x)
                
# POSTAGGER IF NEEDED
#postag = POSTagger('/home/kwlchanc/nltk_data/taggers/stanford-postagger/models/english-bidirectional-distsim.tagger','/home/kwlchanc/nltk_data/taggers/stanford-postagger/stanford-postagger.jar')
#print(postag.tag(x))

class writeCorpusIter(object):
    def __init__(self, corpusType, sentence=False):
        if corpusType =="billcom":
            billCommitteeQuery()
        elif corpusType =="billhoc":
            billHouseQuery()
        elif corpusType =="com":
            committeeQuery()
        elif corpusType =="hoc":
            houseQuery()
        else:
            raise IOError
            
        self.q = queryIterator(sentence)

    def __iter__(self):
        return self
        
    def __next__(self):
        return next(self.q)
    
# Code for creating initial integer dictionaries/vector corpus

def writeCorpusDict(corpusType, sentence=False):
    test = writeCorpusIter(corpusType, sentence)
    dictionary = corpora.Dictionary(x for x in test)

    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
    dictionary.filter_tokens(once_ids) # remove stop words and words that appear only once
    dictionary.compactify() # remove gaps in id sequence after words that were removed
    print(dictionary)
    
    if sentence == True:
        dictionary.save_as_text(('/home/kwlchanc/nltk_data/'+corpusType+'sentences.txt'))
    else:

        dictionary.save_as_text(('/home/kwlchanc/nltk_data/'+corpusType+'.txt'))  # save as text is used here for debugging

class HansardCorpus(object):
    def __init__(self,corpusType,sentence=False):
        self.texts = writeCorpusIter(corpusType,sentence)
        if sentence==True:
            self.dictionary = corpora.Dictionary.load_from_text(('/home/kwlchanc/nltk_data/'+corpusType+'sentences.txt'))
        else:
            self.dictionary = corpora.Dictionary.load_from_text(('/home/kwlchanc/nltk_data/'+corpusType+'.txt'))
    def __iter__(self):
        return self      
    def __next__(self):
        return self.dictionary.doc2bow(next(self.texts))
    
def writeVectorCorpus(corpusType,sentence=False):
    thecorpus = HansardCorpus(corpusType, sentence)
    if sentence == True:
        corpora.SvmLightCorpus.serialize(('/home/kwlchanc/nltk_data/'+corpusType+'sentences.svmlight'), thecorpus)
    else:
        corpora.SvmLightCorpus.serialize(('/home/kwlchanc/nltk_data/'+corpusType+'.svmlight'), thecorpus)

def close():
    cursor.close()
    conn.close()
    

# comparative policy agendas ?


# In[ ]:

def writeAllDictionaries():
    for c in ["billhoc","billcom","hoc","com"]:
        writeCorpusDict(c)  # use to write initial dictionaries

def writeAllCorpora():
    for c in ["billhoc","billcom","hoc","com"]:
        writeVectorCorpus(c)
        
def writeAllCorporaSentences():
    for c in ["billhoc","billcom","hoc","com"]:
        writeVectorCorpus(c, True)
        
def writeAllDictionariesSentences():
    for c in ["billhoc","billcom","hoc","com"]:
        writeCorpusDict(c, True)
        


#writeAllDictionariesSentences()
#writeAllCorporaSentences()
#writeAllDictionaries()
#writeAllCorpora()


# In[19]:

# analysis setup

for dev in ["hoc","com","hocsentences","comsentences","billhocsentences","billcomsentences"]:
# main corpus
    dictionary = corpora.Dictionary.load_from_text('/home/kwlchanc/nltk_data/'+dev+'.txt')
    corpus = corpora.SvmLightCorpus('/home/kwlchanc/nltk_data/'+dev+'.svmlight')
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # optional comparison corpus

    #dictionary2 = corpora.Dictionary.load_from_text('/home/kwlchanc/nltk_data/billhocsentences.txt')
    #corpus2 = corpora.SvmLightCorpus('/home/kwlchanc/nltk_data/billhocsentences.svmlight')
    #tfidf2 = models.TfidfModel(corpus2)
    #corpus_tfidf2 = tfidf2[corpus2]

    print("corpora loaded")

    #lsi2 = models.LsiModel(corpus_tfidf2, id2word=dictionary2, num_topics=2)
    #corpus_lsi2 = lsi2[corpus_tfidf2]
    #lsi2.print_topics(10)

    print("lsi2")

    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=4)
    corpus_lda = lda[corpus_tfidf]
    with open(os.path.join("/home/kwlchanc/nltk_data/", "4"+dev+"topicsLDA.txt"), 'wt') as file:
        for topic in lda.print_topics(4):
            file.write(topic + '\n')



#lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=10)
#corpus_lda = lda[corpus_tfidf]
#with open(os.path.join("/home/kwlchanc/nltk_data/", "10comsentencestopicsLDA.txt"), 'wt') as file:
#    for topic in lda.print_topics(10):
#        file.write(topic + '\n\n')
    

# similarity analysis

#index = similarities.MatrixSimilarity(corpus_lsi, num_features=2)
#simsa = index[corpus_lsi2]
#sims = list(enumerate(simsa))
#simsb = sorted(sims, key=lambda item:item[1].mean())

#print("simsdone")

#disSims = []
#for n in simsb:
#    if n[1].mean()<=0:
#        disSims.append(n[0])


#disText = []
#for n in disSims:
#    workingText=[]
#    workingDoc=corpus.docbyoffset(n)
#    for d in workingDoc:
#        workingText.append(dictionary.get(d))
#    disText.append(workingText)

#print(disText[:5])

#with open(os.path.join("/home/kwlchanc/nltk_data/vectors", "dis.txt"), 'wb') as file:
#    for dis in disText:
#        file.write(dis + '\n\n')

        

        


#fcoords = open(os.path.join("/home/kwlchanc/nltk_data/vectors", "comcoordstfidf.csv"), 'wb')
#for vector in corpus:
#    fcoords.write(bytes("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]), 'UTF-8'))
#fcoords.close()





# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

MODELS_DIR = "/home/kwlchanc/nltk_data/vectors"
MAX_K = 12
NUM_TOPICS = 10

def kMeansAnalysis(coordsfile):
    X = np.loadtxt(os.path.join(MODELS_DIR, (coordsfile+".csv")), delimiter="\t")

    kmeans = KMeans(NUM_TOPICS).fit(X)
    y = kmeans.labels_

    colors = ["Blue", "Green", "Red", "Magenta", "Cyan", "Yellow","Black","Indigo","Chocolate","Teal"]
    for i in list(range(X.shape[0])):
        plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)    
    #plt.show()
    plt.savefig(os.path.join(MODELS_DIR, (coordsfile+'.png')), bbox_inches='tight')
    plt.close()

def kMeansPretest(coordsfile):
    
    X = np.loadtxt(os.path.join(MODELS_DIR, (coordsfile+".csv")), delimiter="\t")
    ks = range(1, MAX_K + 1)

    inertias = np.zeros(MAX_K)
    diff = np.zeros(MAX_K)
    diff2 = np.zeros(MAX_K)
    diff3 = np.zeros(MAX_K)
    for k in ks:
        kmeans = KMeans(k).fit(X)
        inertias[k - 1] = kmeans.inertia_
        # first difference    
        if k > 1:
            diff[k - 1] = inertias[k - 1] - inertias[k - 2]
        # second difference
        if k > 2:
            diff2[k - 1] = diff[k - 1] - diff[k - 2]
        # third difference
        if k > 3:
            diff3[k - 1] = diff2[k - 1] - diff2[k - 2]

    elbow = np.argmin(diff3[3:]) + 3

    plt.plot(ks, inertias, "b*-")
    plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
             markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
    plt.ylabel("Inertia")
    plt.xlabel("K")
    #plt.show()
    plt.savefig(os.path.join(MODELS_DIR, (coordsfile+'pretest.png')), bbox_inches='tight')
    #plt.close()

for n in ["comcoordstfidf","hoccoordstfidf"]:
    kMeansPretest(n)


# In[7]:

def generateIndices(sentences=False):
    
    typelist = []
    
    if sentences==False:
        typelist = ["billhoc","billcom","hoc","com"]
    elif sentences==True:
        typelist = ["billhocsentences","billcomsentences","hocsentences","comsentences"]
        
    for corpusType in typelist:
        dictionary = corpora.Dictionary.load_from_text(('/home/kwlchanc/nltk_data/'+corpusType+'.txt'))
        corpus = corpora.SvmLightCorpus(('/home/kwlchanc/nltk_data/'+corpusType+'.txt'))

        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)
        corpus_lsi = lsi[corpus_tfidf]
        
        index = similarities.MatrixSimilarity(corpus_lsi)
        index.save(('/home/kwlchanc/nltk_data/'+corpusType+'.index'))

#generateIndices(sentences=False)
#generateIndices(sentences=True)


# In[ ]:

close()


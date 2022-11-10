import numpy as np
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# query is line 5 in the file
index = 5


# read the documents and query
def read_file():
    df = open("corona-vaccine-fake-news.txt", 'r', encoding='UTF8')
    # list for documents
    doc = list()
    while True:
        line = df.readline()
        if not line:
            break

        line = line.lower()

        line = re.sub(r'[^a-zA-Z0-9\s]', ' ', line)
        doc.append(line)

    df.close()
    # make the list for query
    query = list()
    query.append(doc[index])
    # pop the query
    doc.pop()

    return doc, query


# cosine similarity
def cosine_similarity(table, query):
    q = query.ravel()
    query_len = calculate_length(q)
    cos_sim = list()

    i = 1
    for k in table:
        k = k.ravel()
        doc_len = calculate_length(k)
        cos_sim.append([i, np.dot(k, q) / doc_len * query_len])
        i += 1

    return cos_sim


# calculate length of documents and query
def calculate_length(k):
    sum = 0
    for i in range(len(k)):
        sum += math.pow(k[i], 2)

    return math.sqrt(sum)


#  documents - query ranking
def order_rank(cos_sim):
    # sort the rank by cosine similarity
    rank = sorted(cos_sim, key=lambda sim: sim[1], reverse=True)
    print('Ranked list:')
    i = 1
    for r in rank:
        print('No. ', i, ': Doc', r[0], ', Similarity score: ', r[1])
        i += 1


# Read doc and query
doc, query = read_file()
vectorizer = TfidfVectorizer(stop_words='english').fit(doc)

# documents
TF_IDF_table = vectorizer.transform(doc).toarray()

# query
TF_IDF_query = vectorizer.transform(query).toarray()
# cosine similarity
cos_sim = cosine_similarity(TF_IDF_table, TF_IDF_query)

# ranking
order_rank(cos_sim)

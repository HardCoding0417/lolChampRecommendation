import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

df_comments = pd.read_csv('data/cleaned_one_comments.csv')
# df_comments.info()

comments = list(df_comments['comments'])
# print(comments[0])

tokens = []
for comment in comments:
    token = comment.split()
    tokens.append(token)

# print(tokens[0])

embedding_model = Word2Vec(tokens, vector_size=100,window=4, min_count=20, workers = 12, epochs=100, sg=1)
embedding_model.save('./models/word2vec_lol_one_comments.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))

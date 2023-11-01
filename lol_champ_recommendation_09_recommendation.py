import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
import re
from gensim.models import Word2Vec

# 코사인 유사도를 입력받아
# 가장 유사한 문서를 리턴하는 함수
def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore,key = lambda x: x[1],reverse=True)
    simScore = simScore[:11]
    movieIdx = [i[0] for i in simScore]
    recMovieList = df_comments.iloc[movieIdx, 0]
    return recMovieList

df_comments = pd.read_csv('./data/cleaned_one_comments.csv')
Tfidf_matrix = mmread('./models/Tfidf_lol_one_comments.mtx').tocsr()
with open('./models/tfidf_one.pickle','rb') as f:
    Tfidf = pickle.load(f)

# 코사인 유사도0
# print(df_comments.iloc[104,0]) # 해당하는 챔프의 이름을 출력
cosine_sim = linear_kernel(Tfidf_matrix[104],Tfidf_matrix) # 104에 대한 문서와 다른 문서간의 코사인 유사도를 측정
# print(cosine_sim[0]) # 요소가 1개인 2차원 행렬

recommendation = getRecommendation(cosine_sim)
# print(recommendation)

try:
    embedding_model = Word2Vec.load('./models/word2vec_lol_one_comments.model')
    keyword = '고인'
    sim_word = embedding_model.wv.most_similar(keyword, topn=10)
    # print(sim_word)
    words = [keyword]
    for word, _ in sim_word:
        words.append(word)
    # print(words)

    # 가장 많이 나온 단어는 10개 그 다음은 9개 그 다음은 8개 ...
    sentence = []
    count = 10
    print(words[:20])
    for word in words:
        sentence = sentence + [word] * count
        count-=1
    sentence = ' '.join(sentence)
    # print(sentence)
    sentence_vec = Tfidf.transform([sentence])
    cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
    recommendation = getRecommendation(cosine_sim)
    print(recommendation)
except:
    print('다른 키워드를 이용하세요.')

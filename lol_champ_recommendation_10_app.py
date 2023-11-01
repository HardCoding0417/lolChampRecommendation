import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5.QtCore import QStringListModel

form_window = uic.loadUiType('./ui/wechamp.ui')[0] #Qt디자인에서 만든ui 불러오는 코드

class Exam(QMainWindow, form_window):
## 클래스 작성
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Tfidf_matrix = mmread('./models/Tfidf_lol_one_comments.mtx').tocsr()
        with open('./models/tfidf_one.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_lol_one_comments.model')
        self.df_comments = pd.read_csv('./data/cleaned_one_comments.csv')
        self.comments = list(self.df_comments['comments'])
        self.comments.sort()
        # for comment in self.comments:
        #     self.comboBox.addItem(comment)

        # # 자동완성용 코드
        # model = QStringListModel()
        # model.setStringList(self.comments)
        # completer = QCompleter()
        # completer.setModel(model)
        # self.le_keyword.setCompleter(completer)

        # self.comboBox.currentIndexChanged.connect(self.combobox_slot)

        self.btn_recommendation.clicked.connect(self.btn_slot)


    def btn_slot(self):
        keyword = self.le_keyword.text()
        self.le_keyword.setText('')
        print(keyword)
        if keyword:
            if keyword in self.comments:
                recommendation = self.recommendation_by_champ_name(keyword)
                # self.lbl_recommendation.setText(recommendation) # 이전 코드
            else:
                recommendation = self.recommendation_by_keyword(keyword)
                # self.lbl_recommendation.setText(recommendation) # 이전 코드

            # 리스트 위젯에 추천 항목 추가
            self.listWidget_recommendations.clear()  # 이전 항목 삭제
            for champ in recommendation.split('\n'):
                self.listWidget_recommendations.addItem(champ)

    def recommendation_by_keyword(self, keyword):
        try:
            sim_word = self.embedding_model.wv.most_similar(keyword, topn=10)
            print(sim_word)
            words = [keyword]
            for word, _ in sim_word:
                words.append(word)
            print(words)

            sentence = []
            count = 10
            for word in words:
                sentence = sentence + [word] * count
                count -= 1
            sentence = ' '.join(sentence)
            print(sentence)
            sentence_vec = self.Tfidf.transform([sentence])
            cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
            recommendation = self.getRecommendation(cosine_sim)
            return recommendation
        except:
            return '다른 키워드를 입력해주세요.'

    # def combobox_slot(self):
    #     title = self.comboBox.currentText()
    #     recommendation = self.recommendation_by_movie_title(title)
    #     self.lbl_recommendation.setText(recommendation)

    # def recommendation_by_champ_name(self, title):
    #     champ_idx = self.df_comments[self.df_comments['champions']==champion].index[0]
    #     cosine_sim = linear_kernel(self.Tfidf_matrix[champ_idx], self.Tfidf_matrix)
    #     recommendation = self.getRecommendation(cosine_sim)
    #     return recommendation


    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:11]
        champIdx = [i[0] for i in simScore]
        recChampList = self.df_comments.iloc[champIdx, 0]
        recChampList = '\n'.join(recChampList[1:])  ## 총 11개 중에 첫번째(같은거)빼고 줄바꿈으로 출력
        print(recChampList)
        return recChampList

if __name__=='__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())


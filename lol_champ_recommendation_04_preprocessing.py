import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('data/lol_champ_repl_new.CSV', encoding='UTF-8', delimiter=',', header=None, names=['champions', 'comments'])
df.info()
# print(df.comments)

okt = Okt()

df_stopwords = pd.read_csv('data/stopwords.csv')
stopwords = list(df_stopwords['stopword'])
cleaned_comments = []

count = 0
for comment in df.comments:
#작업상태 시각화
    count +=1
    if count % 1 == 0:
        print('.', end='')
    if count % 10 == 0:
        print()
    if count % 100 == 0:
        print(count / 1000, end='')

# 토크나이징
    comment = re.sub('[^가-힣]', ' ', comment)
    tokened_comment = okt.pos(comment, stem=True)

    # 쓰고 싶은 품사만 남김
    df_token = pd.DataFrame(tokened_comment, columns=['word', 'class'])
    df_token = df_token[(df_token['class'] == 'Noun') |
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')]
    # 불용어 제거
    words = []
    for word in df_token.word:
        if 1 < len(word):
            if word not in stopwords:
                words.append(word)
    cleaned_comment = ' '.join(words)
    cleaned_comments.append(cleaned_comment)

# csv로 저장
df['cleaned_comments'] = cleaned_comments
df = df[['champions', 'cleaned_comments']]
print(df.head(10))
df.to_csv('data/cleaned_comments.csv', index=False)


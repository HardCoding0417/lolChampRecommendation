# dropna 빼고는 딱히 필요 없는 것 같음. 애초에 잘 붙여놨기 때문에...

import pandas as pd

df = pd.read_csv('data/cleaned_comments.csv')
df.dropna(inplace=True)
df.info()

one_comments = []
for champion in df['champions'].unique():
    temp = df[df['champions'] == champion]
    one_comment = ' '.join(temp['cleaned_comments'])  # comments 컬럼을 모두 합친다
    one_comments.append(one_comment)

df_one = pd.DataFrame({'champions':df['champions'].unique(), 'comments':one_comments})
print(df_one.head())
df_one.info()
df_one.to_csv('data/cleaned_one_comments.csv', index=False)


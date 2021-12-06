#必要なモジュールをインポート
import pandas as pd
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#pandasライブラリで読み込み。自動的にカラム名設定。
df = pd.read_csv("./ml-latest-small/ratings.csv", sep=",")
#最初の0~2列目までを利用（ユーザーID、映画ID、評価の3列を利用）
df = df.iloc[:,0:3]
#疎行列(行列の中身が0ばかりの行列)に変換。行名が映画のID、列名がユーザーID、行列の中身が評価
df_piv = df.pivot(index= "movieId",columns="userId",values="rating").fillna(0)
#scikit-learnでの処理が速くなるデータ形式に変換
df_sp = csr_matrix(df_piv.values)
#KNNのインスタンス化。評価方法をコサイン類似度に設定。bruteは総当り方式。
rec = NearestNeighbors(n_neighbors=10,algorithm= "brute", metric= "cosine")
# KNNで訓練
rec_model = rec.fit(df_sp)

# 映画ID「100」に対するおすすめの映画10個を類似度(下記のdistances)が近い順番に表示。
#下記のindicesは各点から近い順にインデックスが入っており、行iには点iに近い順にインデックスが入っている。
# 下記のn_neighbors=11は自身を含めて11個のデータを取り出す。
Movie = 100#任意の好きな映画ID
distance, indice = rec_model.kneighbors(df_piv.iloc[df_piv.index== Movie].values.reshape(1,-1),n_neighbors=11)
for i in range(11):#i=0は映画ID100のことであり、i=1~10までは評価が似ている映画のこと。
    if  i == 0:#i=0はID100のこと。
        print("映画ID{}を見た人におすすめの映画IDは以下です。".format(Movie))
    else:
        print("{0}: {1}".format(i,df_piv.index[indice.flatten()[i]]))#似ている順に映画IDを表示。
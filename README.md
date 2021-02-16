# ML100Days

### 機器學習概論
  * Day1 資料分析與評估資料 選擇一組dataset並說明WHAT, WHO, WHICH
  * Day2 機器學習概論 ML應用前景，可能的發展
  * Day3 機器學習流程與步驟
    * 資料蒐集、前置處理
      * 政府公開資料、Kaggle 資料
        * 結構化資料 : Excel 檔、CSV 檔
        * 非結構化資料 : 圖片、影音、文字
      * 使用 Python 套件
        * 開啟圖片 : PIL、skimage、open-cv
        * 開啟文件 : pandas
      * 資料前置處理
        * 缺失值填補
        * 離群值處理
        * 標準化
    * 定義目標與評估
      * 回歸問題？分類問題？
      * 將資料分為：訓練集(training set)、驗證集(validation set)、測試集(test set)
      * 評估指標
        * 回歸問題 (預測值為實數)
          * RMSE : Root Mean Squeare Error
          * MAE : Mean Absolute Error
          * R-Square
        * 分類問題 (預測值為類別)
          * Accuracy
          * F1-score
          * AUC，Area Under Curve
    * 建立模型與調整參數
      * Regression，回歸模型
      * Tree-base model，樹模型
      * Neural network，神經網路
      * Hyperparameter，根據對模型了解和訓練情形進行調整
  * Day4 HTTP Server-Client 架構說明與 利用 Python 存取 API

### 資料清理數據前處理
  * D5：如何新建一個 dataframe? 如何讀取其他資料? (非 csv 的資料)
    * 用pd.DataFrame建立
    * CSV
     ```
     import pandas as pd
     df = pd.read_csv('example.csv') # sep=','
     df = pd.read_table('example.csv') # sep='\t'
     ```
    * text
     ```
     with open('example.txt','r') as f:
     data = f.readlines()
     print(data)
     ```
    * Json
     ```
     import json
     with open('example.json','r') as f:
     data = json.load(f)
     print(data)
     ```
    * 矩陣檔 (mat)
     ```
     import scipy.io as sio
     data = sio.load('example.mat')
     ```
    * PNG/JPG
     ```
     import cv2
     image = cv2.imread('example.jpg') # Cv2 會以 GBR 讀入
     image = cv2.cvtcolor(image,cv2.COLOR_BGR2RGB)
     ```
  * D6：EDA: 欄位的資料類型介紹及處理
  * D7：特徵類型
  * D8：EDA資料分佈

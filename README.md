# ML100Days

### 機器學習概論
  * D1 資料分析與評估資料 選擇一組dataset並說明WHAT, WHO, WHICH
  * D2 機器學習概論 ML應用前景，可能的發展
  * D3 機器學習流程與步驟
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
  * D4 HTTP Server-Client 架構說明與 利用 Python 存取 API

### 資料清理數據前處理
  * D5：如何新建一個 dataframe? 如何讀取其他資料? (非 csv 的資料)
    * 用pd.DataFrame建立
    * CSV
     ```python
     import pandas as pd
     df = pd.read_csv('example.csv') # sep=','
     df = pd.read_table('example.csv') # sep='\t'
     ```
    * text
     ```python
     with open('example.txt','r') as f:
     data = f.readlines()
     print(data)
     ```
    * Json
     ```python
     import json
     with open('example.json','r') as f:
     data = json.load(f)
     print(data)
     ```
    * 矩陣檔 (mat)
     ```python
     import scipy.io as sio
     data = sio.load('example.mat')
     ```
    * PNG/JPG
     ```python
     import cv2
     image = cv2.imread('example.jpg') # Cv2 會以 GBR 讀入
     image = cv2.cvtcolor(image,cv2.COLOR_BGR2RGB)
     ```
  * D6：EDA: 欄位的資料類型介紹及處理
    > LabelEncoder(.fit/.transform)+OneHotEncoder(get_dummies)
  * D7：特徵類型
    > pd.concat合併表,drop,groupby,aggregate + int/float/object類型認識
    * 交叉驗證(Cross Validation)
      * The Validation Set Approach
      * LOOCV (Leave-one-out cross-validation)
      * K-fold Cross Validation
  * D8：EDA資料分佈
    > 處理異常值
    * 平均數：mean()、中位數：median()、眾數：mode()、最小值：min()、最大值：max()、四分位差：quantile()、變異數：var()、標準差：std()
    * [matplotlib](https://matplotlib.org/stable/index.html)
     ```python
     import matplotlib.pyplot as plt
     # using the variable ax for single a Axes
     fig, ax = plt.subplots()

     # using the variable axs for multiple Axes
     fig, axs = plt.subplots(2, 2)

     # using tuple unpacking for multiple Axes
     fig, (ax1, ax2) = plt.subplot(1, 2)
     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplot(2, 2)
     ```
  * D9:EDA 離群值(Outliner)及其處理
    * Outliers 的處理方法
      * 新增欄位用以紀錄異常與否(Y/N)
      * 整欄不用
      * 填補：中位數, Min, Max 或平均數
    * 檢查異常值的方法
      * 統計值：如平均數、標準差、中位數、分位數
      * 畫圖：如直方圖、盒圖、次數累積分布等
      * [Ways to Detect and Remove the Outliers](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)
        * 視覺方法--[boxplot,](https://cloud.tencent.com/developer/article/1429994) scatter plot
         ```python
         import seaborn as sns
         sns.boxplot(x=boston_df['DIS'])
         ```
        * 統計方法--zscore, IQR
         ```python
         Q1 = boston_df_o1.quantile（0.25）
         Q3 = boston_df_o1.quantile（0.75）
         IQR = Q3-Q1 
         print（IQR）  #計算四分衛間距(中間50%)
         print(boston_df_o1 < (Q1 - 1.5 * IQR)) |(boston_df_o1 > (Q3 + 1.5 * IQR))   #列出所有outliers
         
         #刪除outlier
         boston_df_out = boston_df_o1 [〜（（boston_df_o1 <（Q1- 1.5 * IQR））|（boston_df_o1>（Q3 + 1.5 * IQR）））。any（axis = 1）]
         boston_df_out.shape
         ```
      * 標準差與容忍範圍
        * 1 個標準差: 涵蓋 68% 數據
        * 2 個標準差: 涵蓋 95% 數據
        * 3 個標準差: 涵蓋 99.7% 數據
        * 如果一個數字超過平均值 + 3 個標準差 !!!有問題
  

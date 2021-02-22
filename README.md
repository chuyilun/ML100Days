# ML100Days

### 機器學習概論
  * __D1：資料分析與評估資料 選擇一組dataset並說明WHAT, WHO, WHICH__
  * __D2：機器學習概論 ML應用前景，可能的發展__
  * __D3：機器學習流程與步驟__
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
  * __D4：HTTP Server-Client 架構說明與 利用 Python 存取 API__

### 資料清理數據前處理
  * __D5：如何新建一個 dataframe? 如何讀取其他資料? (非 csv 的資料)__
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
     > PIL, Skimage, CV2 (CV2 的速度較快，但須注意讀入的格式為 BGR，其他兩個君為RGB) [三者比較](https://zhuanlan.zhihu.com/p/52344534)
  * __D6：EDA: 欄位的資料類型介紹及處理__
    > LabelEncoder(.fit/.transform)+OneHotEncoder(get_dummies)
  * __D7：特徵類型__
    > pd.concat合併表,drop,groupby,aggregate + int/float/object類型認識
    * 交叉驗證(Cross Validation)
      * The Validation Set Approach
      * LOOCV (Leave-one-out cross-validation)
      * K-fold Cross Validation
  * __D8：EDA資料分佈__
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
  * __D9:EDA 離群值(Outliner)及其處理__
    > [loc, iloc用法](https://blog.csdn.net/W_weiying/article/details/81411257)
    * Outliers 的處理方法
      * 新增欄位用以紀錄異常與否(Y/N)
      * 整欄不用
      * 填補：中位數, Min, Max 或平均數
    * 檢查異常值的方法
      * 統計值：如平均數、標準差、中位數、分位數
       ```python
       df.describe()
       df[col].value_counts()
       ```
      * 畫圖：如直方圖、盒圖、次數累積分布等
       ```python
       df.plot.hist()  # 直方圖
       df.boxplot()    # 盒圖
       cdf = app_train['AMT_INCOME_TOTAL'].value_counts().sort_index().cumsum()  #畫ECDF
       plt.plot(list(df.index), df/df.max())   # 次數累積圖
       ```
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
  * __D10：數值型特徵 - 去除離群值__
    * 1 way:捨棄outlier
     ```python
     keep_indexs = (df['1stFlrSF']> 450) & (df['1stFlrSF']< 2500)  #設立條件後，將新的數值們存到新變數中
     df = df[keep_indexs]
     ```
    * 2 way:調整outlier
     ```python
     #dataframe.clip()用於在數據單元格中任何單元格可以具有的值設置下限和上限
     df['1stFlrSF'] = df['1stFlrSF'].clip(0, 2500)
     ```
  * __D11：常用的數值取代：中位數與分位數連續數值標準化__
    > [np.percentile用法](https://www.796t.com/article.php?id=22735)
    * 常用以替補的統計值
    
    | 常用以替補的統計值  |     方法                          |
    | ----------------- |:---------------------------------:|
    | 中位數(median)     | np.median(value_array)            |
    | 分位數(quantiles)  | np.quantile(value_array, q = ... )|
    | 眾數(mode)         | dictionary method :較快的方法      |
    | 平均數(mean)       | np.mean(value_array)              |
  * __D12：數值型特徵-補缺失值與標準化__
  
    |     標準化模型     |              例子                          |
    | ----------------- |:------------------------------------------:|
    |     非樹狀模型     |  線性回歸、羅吉斯回歸、類神經，對預測會有影響  |
    |      樹狀模型      |決策樹、隨機森林樹、梯度提升樹，對預測不會有影響|
  * __D13：DataFrame operationData frame merge/常用的 DataFrame 操作__
    * 轉換
     ```python
     pd.melt(df)    #將column轉成row
     pd.pivot(colums='var', values='val')    #將row轉成column
     ```
    
    * 合併
   
     ```python
     pd.concat([df1,df2])    #沿row合併兩個dataframe，中括號間放表名
     pd.concat([df1,df2], axis = 1)    #沿column合併兩個dataframe，中括號間放表名
     ```
     ```python
     pd.merge(df1,df2,on = 'id', how = 'outer')    #將df1, df2 以"id"欄做全部合併(缺失值以na補)
     pd.merge(df1,df2,on = 'id', how = 'inner')    #將df1, df2 以"id"欄做部分合併
     ```
    * subset
     ```python
     sub_df = df[df.age > 20]        #邏輯操作(>,<,=,&,|,~,^)
         df.column.isin(value)
         pd.isnull(obj)  # df.isnull()   #為Nan
         pd.notnull(obj) # df.notnull()  #非Nan
     df = df.drop_duplicates()       #移除重複
     sub_df = df.head(n = 10)        #找前n筆資料
     sub_df = df.tail(n = 10)        #找後n筆資料
     sub_df = df.sample(frac = 0.5)  #抽50%
     sub_df = df.sample(n = 10)      #抽10筆
     sub_df = df.iloc[n:m]           #第n筆到m筆
     new_df = df['col1']或df.col1    #單一欄位
     new_df = df[['col1','col2','col3']]    #複數欄位
     new_df = df.filter(regex = ...) #Regex篩選
     ```
    * Group
     ```python
     sub_df_object = df.groupby(['col1'])

     sub_df_object.size()          #計算各組數量
     sub_df_object.describe()      #得到各組的基本統計值
     sub_df_object['col2'].mean()  #根據col1分組後，計算col2統計值(此例是取平均值)
     sub_df_object['col2'].apply() #對依col1分組後的col2引用操作
     sub_df_object['col2'].hist()  #對依col1分組後的col2繪圖
     ```
  * __D14：程式實作 EDA: correlation/相關係數簡介__
    * Correlation Coefficient：
     了解兩個變數之間的線性關係，數值介於-1~1，數值越大相關性越強

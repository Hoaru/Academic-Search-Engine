from snownlp import SnowNLP
import pandas as pd
import numpy as np
import threading
import jieba
import math
import time
import sys


from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets

from gensim import corpora

import search
import result


#信号类
class MySignal(QWidget):
    # 定义信号,定义参数为无类型
    update_date = pyqtSignal()


class HMM:
    def __init__(self, trainpath):
        self.param = self.train(trainpath)


    def train(self, fileName):
        # HMM模型由三要素决定 lambda=（A，B，pi）
        # A为状态转移矩阵
        # B为观测概率矩阵
        # pi为初始状态概率向量

        # 在该函数中，我们需要通过给定的训练数据（包含S个长度相同的观测序列【每一句话】和对应的状态序列【每一句话中每个词的词性】

        # 在中文分词中，包含一下集中状态（词性）
        # B：词语的开头（单词的头一个字）
        # M：中间词（即在一个词语的开头和结尾之中）
        # E：单词的结尾（即单词的最后一个字）
        # S：单个字

        # 定义一个状态映射字典。方便我们定位状态在列表中对应位置
        status2num={'B':0,'M':1,'E':2,'S':3}

        # 定义状态转移矩阵。总共4个状态，所以4x4
        A=np.zeros((4,4))

        #定义观测概率矩阵
        #在ord中，中文编码大小为65536，总共4个状态
        #所以B矩阵4x65536
        #就代表每一种状态（词性）得到观测状态（字）
        B=np.zeros((4,65536))

        # 初始状态，每一个句子的开头只有4中状态（词性）
        PI=np.zeros(4)

        with open(fileName,encoding='utf-8') as file:
            # 每一行读取
            # 如某一行语料为：   迈向  充满  希望  的  新  世纪 。
            # 语料库为我们进行好了切分，每一个词语用空格隔开。
            # 那么在这其中，我们将每个词语切分（包括标点符号）放在列表中。
            # 然后遍历列表每一个元素
            # 当列表词语长度为1的时候，如 '的'字，那么我们就认为状态为S（单个字）
            # 当列表长度为2的时候，如'迈向'，我们认为'迈'为B，'向'为E
            # 当长度为3以上时候，如'实事求是'，我们认为'实'为B，'事求'两个字均为M，'是'为E


            # 我们遍历完毕所有的语料，就可以按照公式10.39,40,41来获取A，B，PI
            # 其实这三个公式的本质是统计出频数/总数
            # 如10.39，公式上半部分是从1-T-1时刻，t时刻状态为qi，t+1时刻为qj状态的总概率。
            # 那么由似然可以知道，该总概率是由 1-T-1时刻，t时刻状态为qi，t+1时刻为qj状态出现次数/ 观测序列对应状态序列总数
            # 下方也类似。 两者相除，分母均为观测序列对应状态序列总数，可以相互抵消
            # 就可以变为 1-T-1时刻，t时刻状态为qi，t+1时刻为qj状态出现次数/1-T-1时刻，t时刻状态为qi出现次数
            # 所以一下我们只需要统计出现频数，然后除总次数即可。


            # date: 12.26
            # 刚刚上面的说错了QAQ 今天在回顾的时候，发现那时候自己理解还不够啊
            # 我们的训练集是已经标注完毕了，所以我们的学习算法是有监督学习
            # 所以就是直接极大似然，频数/总数   就可以得出模型三要素的参数了
            for line in file.readlines():
                wordStatus=[]#用于保存该行所有单词的状态
                words=line.strip().split() #除去前后空格，然后依照中间空格切分为单词

                for i,word in enumerate(words):

                    # 根据长度判断状态
                    if len(word)==1:
                        status='S'# 保存每一个单词状态
                        # 使用ord找到该字对应编码
                        # 更新B矩阵
                        # B代表了每一个状态到对应观测结果的可能性
                        # 先统计频数
                        code=ord(word)
                        B[status2num[status[0]]][code]+=1

                    else:
                        # 当长度为2，M*0。这样可以一起更新
                        status='B'+(len(word)-2)*'M'+'E'
                        # 使用ord找到该字对应编码
                        # 更新B矩阵
                        # B代表了每一个状态到对应观测结果的可能性
                        # 先统计频数
                        for s in range(len(word)):
                            code=ord(word[s])
                            B[status2num[status[s]]][code]+=1

                    # i==0意味着这是句首。我们需要更新PI中每种状态出现次数
                    if i==0:
                        # status[0]表示这行第一个状态
                        # status2num将其映射到list对应位置
                        PI[status2num[status[0]]]+=1

                    # 使用extend，将status中每一个元素家在列表之中。而不是append直接将整个status放在后面
                    wordStatus.extend(status)

                # 遍历完了一行，然后更新矩阵A
                # A代表的是前一个状态到后一个状态的概率
                # 我们先统计频数
                for i in range(1,len(wordStatus)):
                    # wordStatus获得状态，使用status2num来映射到正确位置
                    A[status2num[wordStatus[i-1]]][status2num[wordStatus[i]]]+=1

        # 读取完毕文件，频数统计完成
        # 接下来计算概率
        # 我们面临的问题是：
        # 1.如果句子较长，许多个较小的数值连乘，容易造成下溢。对于这种情况，我们常常使用log函数解决。
        # 但是，如果有一些没有出现的词语，导致矩阵对应位置0，那么测试的时候遇到了，连乘中有一个为0，整体就为0。
        # 但是log0是不存在的，所以我们需要给每一个0的位置加上一个极小值（-3.14e+100)，使得其有定义。

        # 计算PI向量
        total=sum(PI)
        for i in range(len(PI)):
            if PI[i]==0:
                PI[i]=-3.14e+100
            else:
                # 别忘了去取对数
                PI[i]=np.log(PI[i]/total)

        # 计算A矩阵
        # 要注意每一行的和为1，即从某个状态向另外4个状态转移概率只和为1
        # 最后我们取对数
        for i in range(len(A)):
            total=sum(A[i])
            for j in range(len(A[i])):
                if A[i][j]==0:
                    A[i][j]=-3.14e+100
                else:
                    A[i][j]=np.log(A[i][j]/total)
        # 更新B矩阵
        # B矩阵中，每一行只和为1
        # 即某一个状态到所有观测结果只和为1
        # 最后我们取对数
        for i in range(len(B)):
            total=sum(B[i])
            for j in range(len(B[i])):
                if B[i][j]==0:
                    B[i][j]=-3.14e+100
                else:
                    B[i][j]=np.log(B[i][j]/total)

        # 返回三个参数
        return (PI,A,B)


    def word_partition(self, content):
        article = content

        HMM_parameter = self.param
        '''
        使用维特比算法进行预测（即得到路径中每一个最有可能的状态）
        :param HMM_parameter: PI,A,B隐马尔可夫模型三要素
        :param article: 需要分词的文章,以数组的形式传入，每一个元素是一行
        :return: 分词后的文章
        '''
        PI,A,B=HMM_parameter
        article_partition = [] #分词之后的文章

        # 我们需要计算的是Ψ（psi），δ（delta）
        # delta对应于公式10.44,45.psi对应于公式10.46

        for line in article:
            # 定义delta，psi
            # delta一共长度为每一行长度，每一位有4种状态
            delta=[[0 for _ in range(4)] for _ in range(len(line))]
            # psi同理
            psi=[[0 for _ in range(4)] for _ in range(len(line))]


            for t in range(len(line)):
                if t==0:
                    # 初始化psi
                    psi[t][:]=[0,0,0,0]
                    for i in range(4):
                        # !!! 注意这里是加号，因为之前log处理了
                        delta[t][i]=PI[i]+B[i][ord(line[t])]

                #依照两个公式更细delta和psi
                #注意每一个时刻的delta[t][i]代表的是到当前时刻t，结束状态为i的最有可能的概率
                #psi[t][i]代表的是当前时刻t，结束状态为i，在t-1时刻最有可能的状态（S，M，E，B）
                else:
                    for i in range(4):
                        # 一共4中状态，就不写for循环一个个求出在的max了，直接写成列表了

                        # !!! 划重点，注意这里概率之间的计算用的加号
                        # 因为之前我们进行了log处理，所以之前的概率相乘变成了log相加

                        # temp=[delta[t-1][0]+A[0][i],delta[t-1][1]+A[1][i],delta[t-1][2]+A[2][i],delta[t-1][3]+A[3][i]]
                        temp=[delta[t-1][j]+A[j][i] for j in range(4)] #写成列表生成式吧，短一点。和上面一样的
                        #求出max在乘以b
                        # b[i][ot]中，ot就是观测结果，即我们看到的字
                        # 我们使用ord将其对应到编码，然后就可以获得他在观测概率矩阵中，由状态i到观测结果（ord（line[t]))的概率了
                        delta[t][i]=max(temp)+B[i][ord(line[t])]

                        #求psi
                        #可以注意到，psi公式中，所求的是上一个最有可能的概率
                        #argmax中的值就是上方的temp，所以我们只需要获得temp最大元素的索引即可
                        psi[t][i]=temp.index(max(temp))

            # 遍历完毕这一行了，我们可以计算每个词对应的状态了
            # 依照维比特算法步骤4，计算最优回溯路径
            # 我们保存的是索引，0，1，2，3。对应与B，M，E，S
            status=[] #用于保存最优状态链

            # 计算最优状态链
            # 最优的最后一个状态
            It=delta[-1].index(max(delta[-1]))
            status.append(It)
            # 这是后向的计算该最优路径
            # 所以我们使用insert，在列表最前方插入当前算出的最优节点。
            for t in range(len(delta)-2,-1,-1):
                #status[0]保存的是所求的当前t时刻的后一时刻（t+1），最有可能的状态
                #psi[t][i]表示t时刻，状态为i，t-1时刻最有可能的状态
                # 所以用psi[t+1][status[0]]就可以得出t时刻最有可能的状态
                It=psi[t+1][status[0]]
                status.insert(0,It)

            # 计算出了所有所有时刻最有可能的状态之后，进行分词
            # 遇到S，E我们就要在该词之后输出｜
            # 例如 我今天很开心 对应 S，B，E，S，B，E 输出 我｜今天｜很｜开心｜。
            # 只需要注意这一行最后不输出｜即可
            line_partition=[]
            start = 0
            for t in range(len(line)):
                if (status[t]==2 or status[t]==3) and t!=len(line)-1:
                    end = t
                    line_partition.append(line[start:end+1])
                    start = end+1
            line_partition.append(line[start:len(line)])
            # 结束输出，换行
            article_partition.append(line_partition)

        return article_partition


    def load_article(self, fileName):
        with open(fileName,encoding='utf-8') as file:
            # 按行读取
            test_article=[]
            for line in file.readlines():
                # 去除空格，以及换行符
                line=line.strip()
                test_article.append(line)
        return test_article


class Inverted_Index():

    def __init__(self , documents):
        self.documents = documents
        self.dict = self.build_dict()#统计每个词语数量的词典，例：中国，200
        self.Inverted_Index = self.build_Inverted_Indexs()#倒排索引
        # Inverted_Index：{中国，{（第一片文档出现次数）7，（第二片文档出现次数）3，（第一片文档出现次数）5}
        #                 {美国，{（第一片文档出现次数）9，（第二片文档出现次数）2，（第一片文档出现次数）1}

    def build_dict(self):#计数词典
        dict = {}
        for document in self.documents:
            for word in document:
                if word in dict:
                    dict[word] += 1
                else:
                    dict[word] = 0
        return dict

    def build_Inverted_Indexs(self):
        Inverted_Index = self.dict
        for word in Inverted_Index:
            Inverted_Index[word] = {}
            for j in range(len(self.documents)):
                Inverted_Index[word][j] = 0
        i = 0
        for doc in self.documents:
            self.build_Inverted_Index(doc, Inverted_Index, i)#i表示第i篇文档
            i += 1
        return Inverted_Index

    def build_Inverted_Index(self, document, Inverted_Index: dict, i):
        for word in document:
            Inverted_Index[word][i] += 1


class TFIDF():

    def __init__(self, documents, Inverted_Index, query):
        self.documents = documents
        self.Inverted_Index = Inverted_Index#倒排索引。
        self.Idf = self.compute_Idf()#统计每个词语idf值的词典，例：中国，100
        # Inverted_Index：{中国，{（第一片文档出现次数）7，（第二片文档出现次数）3，（第一片文档出现次数）5}}
        #                 {美国，{（第一片文档出现次数）9，（第二片文档出现次数）2，（第一片文档出现次数）1}}
        self.score = self.compute_score(query)

    def compute_Idf(self):
        idf = {}
        df = {}
        for doc in self.documents:
            for word in doc:
                if word in idf and word not in df:
                    idf[word] += 1
                    df[word] = 1
                elif word not in idf and word not in df:
                    idf[word] = 1
                    df[word] = 1
            df.clear()
        for word in idf:
            idf[word] = math.log(len(self.documents) / (idf[word] + 1))
        return idf

    def compute_score(self, query):
        score = {}
        print()
        i = 0
        j = 0
        for word in query:
            if word not in self.Inverted_Index:
                prstr = '%s关键词无效，不做考察'
                continue
            for docnum in self.Inverted_Index[word]:
                str = '%s' % (i)
                if j == 0:
                    score[str] = self.Inverted_Index[word][i] * self.Idf[word].real
                else:
                    score[str] += self.Inverted_Index[word][i] * self.Idf[word].real
                i += 1
            i = 0
            j += 1
        score = sorted(score.items(), key=lambda d: d[1], reverse=True)#排好序的“（文档序号，得分）”序列
        score = score[0:10]#取前十位
        return score


class SNOWNLP():

    def __init__(self, score, corpus):
        self.score = score
        self.corpus = corpus
        self.keywords = []
        self.sentences = []

    def extract(self, no):
        doc = self.corpus[no]
        s = SnowNLP(doc)
        self.keywords = s.keywords(5)  # 提取前5个关键词
        self.sentences = s.summary(3)  # 提取前3个句子作为摘要


def Load_data(filename):
    df_news = pd.read_csv(filename, sep='\\s+', skiprows=[0], header=None, error_bad_lines=False)
    df_news.columns = ['category', 'content']
    df_news.to_csv('classification_corpus_800_new_result.txt', sep=',', index=None)
    # print(df_news.head())
    content = df_news.content.values.tolist()
    return df_news, content


class Drop_stopword():  #

    def __init__(self, content_S, wordpath):
        self.contents = (pd.DataFrame({'content_S': content_S})).content_S.values.tolist()
        self.stopwords = self.load_stopwords(wordpath)
        self.contents_clean = self.drop_stopwords()  ###使用停用词

    def load_stopwords(self, wordpath):
        stopwords = pd.read_csv(wordpath, index_col=False, sep="\t", quoting=3, names=['stopword'],
                                encoding='utf-8')  # list
        return stopwords.stopword.values.tolist()

    def drop_stopwords(self):
        contents_clean = []
        for line in self.contents:
            line_clean = []
            for word in line:
                if word in self.stopwords:
                    continue
                line_clean.append(word)
            contents_clean.append(line_clean)
        return contents_clean


class Module():

    def __init__(self, contents_clean, df_news):
        self.contents_clean = contents_clean
        self.df_news = df_news
        self.df_train = self.create_Module()
        self.vectorizer, self.classifier = self.create_classifier()

    def create_Module(self):
        df_train = pd.DataFrame({'contents_clean': self.contents_clean, 'label': self.df_news['category']})
        label_mapping = {"classIT": "互联网", "classHealth": "健康", "classsports": "体育", "classEducation": "教育",
                         "classJob": "工作", "classArt": "美术", "classMilitary": "军事"}
        df_train['label'] = df_train['label'].map(label_mapping)  ##变换label
        # print(df_train.head())
        return df_train

    def create_classifier(self):

        """创建训练、测试集"""
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(self.df_train['contents_clean'].values,
                                                            self.df_train['label'].values, test_size=0.2,
                                                            random_state=1)
        """train_test_split(train_data,train_target,test_size, random_state)是sklearn包的model_selection模块中提供的随机划分训练集和测试集的函数；
        使用train_test_split函数可以将原始数据集按照一定比例划分训练集和测试集对模型进行训练
        x,y是原始的数据集。
        X_train,y_train 是原始数据集划分出来作为训练模型的，fit模型的时候用。
        X_test,y_test 这部分的数据不参与模型的训练，而是用于评价训练出来的模型好坏，score评分的时候用。
        y_test是标签集合

        train_target：所要划分的样本结果
        test_size：样本占比，如果是整数的话就是样本的数量
        random_state：是随机数的种子。随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
        test_size=0.2 测试集的划分比例。如果为浮点型，则在0.0-1.0之间，代表测试集的比例；如果为整数型，则为测试集样本的绝对数量；如果没有，则为训练集的补充。 
        random_state：是随机数的种子。固定随机种子时，同样的代码，得到的训练集数据相同。不固定随机种子时，同样的代码，得到的训练集数据不同。"""

        words = []
        for line_index in range(len(x_train)):
            try:
                # x_train[line_index][word_index] = str(x_train[line_index][word_index])
                words.append(' '.join(x_train[line_index]))
            except:
                print(line_index)

        test_words = []
        for line_index in range(len(x_test)):
            try:
                # x_train[line_index][word_index] = str(x_train[line_index][word_index])
                test_words.append(' '.join(x_test[line_index]))
            except:
                print(line_index)

        # TF-IDF模型
        # from sklearn.feature_extraction.text import TfidfVectorizer
        # vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=False)

        # """构建贝叶斯模型"""
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer  = CountVectorizer(analyzer='word', max_features=4000,  lowercase = False)
        vectorizer.fit(words)
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
        classifier.fit(vectorizer.transform(words), y_train)
        return vectorizer, classifier

    def predict(self, text):
        text = self.vectorizer.transform([text])
        result = self.classifier.predict(text)
        return result


def Load_wordlist(fname):
    wordlist = list()
    with open(fname, encoding='utf-8') as fread:
        read_data = fread.read()
        wordlist = read_data.split()
    return wordlist


def Load_corpus(corpus_load_path):
    corpus = []
    with open(corpus_load_path, encoding='utf-8') as fread:
        for line in fread:
            corpus.append(line.strip())
    return corpus


def Process(SnowNLP_ob, Module):
    ten_result = []
    for word in SnowNLP_ob.score:#题目、摘要、评分、分类
        result = ""
        no = word[0]#文档编号
        kind = str((Module.predict(SnowNLP_ob.corpus[int(word[0])]))[0])
        score = str(word[1])
        SnowNLP_ob.extract(int(no))
        title = ""
        for i in range(5):
            title = title + SnowNLP_ob.keywords[i]
        summary = ""
        for j in range(3):
            summary = summary + SnowNLP_ob.sentences[j]

        result += "文档编号："
        result += no
        result += "\n"

        result += "分类："
        result += kind
        result += "\n"

        result += "评分："
        result += score
        result += "\n"

        result += "标题："
        result += title
        result += "\n"

        result += "摘要："
        result += summary
        result += "\n"

        print(result)
        ten_result.append(result)

    return ten_result


class INIT:

    def __init__(self):
        HMM_ob = HMM('pku_training.utf8')
        self.corpus = HMM_ob.load_article('ir_corpus_1000_shuffled_new.txt')
        WordSegmentation_result = HMM_ob.word_partition(self.corpus)
        self.InvertedIndex = Inverted_Index(WordSegmentation_result)  # 倒排索引结果
        df_news, content = Load_data(r'classification_corpus_800_new.txt')
        WordSegmentation_result = HMM_ob.word_partition(content)  # 分词结果
        self.Module_ob = Module(Drop_stopword(WordSegmentation_result, "stoplist_utf8.txt").contents_clean, df_news)


class Search(QMainWindow, search.Ui_MainWindow):

    def __init__(self):
        super(Search, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("摆渡一下")


class Result(QMainWindow, result.Ui_MainWindow):

    def __init__(self):
        super(Result, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("搜索结果")
        self.ten_result = []
        self.btn_quit.clicked.connect(self.quit)

    def quit(self):
        self.close()

    def Process_display_instruction(self, corpus, InvertedIndex, Module_ob, keyword):
        TFIDF_ob = TFIDF(InvertedIndex.documents, InvertedIndex.Inverted_Index, keyword)
        SNOWNLP_ob = SNOWNLP(TFIDF_ob.score, corpus)
        self.ten_result = Process(SNOWNLP_ob, Module_ob)
        ms_diaplay.update_date.emit()
        self.Display_data()

    def Display_data(self):
        ten_result = self.ten_result
        self.label_1.setText(ten_result[0])
        self.label_2.setText(ten_result[1])
        self.label_3.setText(ten_result[2])
        self.label_4.setText(ten_result[3])
        self.label_5.setText(ten_result[4])
        self.label_6.setText(ten_result[5])
        self.label_7.setText(ten_result[6])
        self.label_8.setText(ten_result[7])
        self.label_9.setText(ten_result[8])
        self.label_10.setText(ten_result[9])


class Action:

    def __init__(self,corpus, InvertedIndex, Module_ob, searchWindow, resultWindow):
        self.corpus = corpus
        self.InvertedIndex = InvertedIndex
        self.Module_ob = Module_ob
        self.searchWindow = searchWindow
        self.resultWindow = resultWindow
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.keyword = []
        # 初始化信号，用于子线程更改UI
        global ms_diaplay
        ms_diaplay = MySignal()
        ms_diaplay.update_date.connect(self.resultWindow.Display_data)

    def SW_search_event(self):
        text = self.searchWindow.text_edit.text()
        self.keyword = jieba.lcut(text)
        self.searchWindow.text_edit.setText("")
        self.stopEvent.set()
        self.resultWindow.show()
        self.searchWindow.close()
        th = threading.Thread(target=self.Diplay_event)
        th.start()
        self.stopEvent.set()

    def Diplay_event(self):
        resultWindow.Process_display_instruction(self.corpus, self.InvertedIndex, self.Module_ob, self.keyword)

    def RW_return_event(self):
        self.searchWindow.show()
        self.resultWindow.close()





if __name__ == '__main__':


    begin_time = time.process_time()

    app = QApplication(sys.argv)
    INIT_ob = INIT()
    searchWindow = Search()
    searchWindow.show()
    resultWindow = Result()#INIT_ob.corpus, INIT_ob.InvertedIndex, INIT_ob.Module_ob
    Action_ob = Action(INIT_ob.corpus, INIT_ob.InvertedIndex, INIT_ob.Module_ob, searchWindow, resultWindow)

    mid_time = time.process_time()
    run_time = mid_time - begin_time
    print('至此运行时间：', mid_time)

    searchWindow.btn_search.clicked.connect(Action_ob.SW_search_event)
    resultWindow.btn_return.clicked.connect(Action_ob.RW_return_event)
    resultWindow.btn_quit.clicked.connect(resultWindow.close)
    sys.exit(app.exec())

    end_time = time.process_time()
    run_time = end_time - begin_time
    print('该程序运行时间：', run_time)
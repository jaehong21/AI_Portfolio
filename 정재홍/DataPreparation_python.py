#!/usr/bin/env python
# coding: utf-8

# 한국어-영어 번역 관련 '자연어 처리' 데이터셋은 'AI 허브'라는 사이트에서 다운로드 받았다. <br />
# AI 허브에서는 음성/자연어 분야부터 컴퓨터 비전, 헬스케어 등 많은 분야의 국내 데이터셋 등을 무료로 다운받을 수 있다. <br />
# (회원가입 절차를 필요로 한다) <br />
# 그래서, seq2seq(Sequence-to-Sequence)를 이용한 한영 기계번역기 제작을 위해서 AI 허브에서 한영 구어체 데이터셋을 다운로드했다. <br />
# 그리고, 진행하는 프로젝트 이외에도 공식 Pytorch 사이트에서 제공하는 seq2seq 튜토리얼에서 제공하는 영어-프랑스어 데이터셋 또한 <br />
# 테스트를 위해서 추가적으로 다운로드 받았다. <br />
# 
# <br /> <br />
# 
# 
# 영어-프랑스어 번역 데이터셋 출처: https://download.pytorch.org/tutorial/data.zip
# 한국어-영어 번역 데이터셋 출처: https://aihub.or.kr/aidata/87

# In[14]:


import pandas as pd
from pandas import DataFrame as df
import re
import random


# In[1]:


excel_1 = pd.read_excel("2_구어체(1).xlsx")
excel_2 = pd.read_excel("2_구어체(2).xlsx")
excel_result = excel_1

excel_result.to_csv('kor-eng.txt', sep='\t', index=False)


# In[2]:


SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else: 
            self.word2count[word] += 1


# <h4>one-hot encoding 및 데이터셋 정제 준비</h4> <br />
# 이번 task에서는 one-hot encoding을 이용하려고 한다. <br />one-hot encoding은 문자를 단어 0으로 이루어진 엄청 큰 벡터에서 특정 단어가 가진 특정한 index에만 1을 넣는 one-hot vector로 나타내는 것을 말한다. <br />
# 학습 이전에 위 과정을 쉽게 하기 위하여, 클래스 'Lang'을 제작하였다. 단어를 index에 대응시킨 word2index와 index에 단어를 대응시킨 index2word를 선언한다.  <br />
# seq2seq에 필수적으로 들어가는 토큰인 "SOS"와 "EOS"는 미리 추가해놓고, 이를 포함하여 n_words는 2로 초기화시켜 놓았다. <br />
# 함수 addSentence는 ' '를 기준으로 문장을 쪼개서 단어를 받는 함수이다. <br />
# addWord는 addSentence로부터 받은 word를 word2index와 index2word에 추가시키고, n_words에 1을 더해준다. <br />
# 문장으로부터 받은 단어가 이미 word2index에 있다면, n_words에만 1 더해준다.

# In[3]:


def normalizeString(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^A-Za-z0-9가-힣.!?]+", r" ", s)
    return s


# <h4>문장 전처리</h4> <br />
# 장부호와 알파벳을 제외한 모든 단어를 제거해준다. <br /> 
# 다시 한번 언급하자면, 컴퓨터는 소문자와 대문자가 섞여있다면, 다른 단어로 인식한다. <br />
# 또한, 문장부호가 붙어있는 것도 마찬가지이다. 그리하여, 납득할만한 결과를 얻기 위하여 이러한 데이터 전처리 과정이 필수적이다. <br />
# (한국어를 다룰 때에는 형태소 분석기 사용을 추천한다)

# In[4]:


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else: 
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs


# <h4>엑셀파일로부터 데이터 추출</h4> <br />
# 본격적으로 data를 추출하는 단계이다. data.zip의 압축을 풀고, python 파일과 같은 디렉토리 내에 kor-eng.txt와 같이 저장해준다.<br /> .read().strip().split('\n)을 통해서 메모장에 있는 파일을 줄 단위로 받아온다. (.strip()은 문자열의 양 끝에 있는 공백 혹은 \n을 제거해주는 함수이다. <br />
# 그 다음에 한 줄에서 한글 문장 한 개와 영어 문장 한 개 짝을 이루어서 받아오는 pair를 만들어준다. <br />
# 메모장에서 한 줄에 한글과 영어가 tab으로 구분되어 있어 .split('\t')으로 pair를 만든다.

# In[5]:


MAX_LENGTH = 15
'''
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
'''

kor_prefixes= (
    "나는", "난",
    "그는", "그녀는",
    "우리는", "그들은",
    "너는", "넌",
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[0].startswith(kor_prefixes)
    # and p[1].startswith(eng_prefixes)
    # 만약 reverse=False면 p[0].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# 본래는 모든 데이터를 학습시키는 데에는 꽤 오랜 시간을 요구한다. 이러한 시간을 조금 줄이고자, 몇가지 필터링을 추가시키고자 한다. <br />
# 1. 문장의 길이가 MAXLENGTH 이하인 문장만 학습한다. <br />
# 2. eng_prefixes를 선언하여, 이의 형태로 시작하는 문장만 골라서 학습한다. <br />

# In[6]:


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# In[11]:


input_lang, output_lang, pairs = prepareData('kor', 'eng', False)
print(random.choice(pairs))


# In[19]:


dataset = df(pairs)
dataset.columns = ['kor', 'eng']
dataset


# In[21]:


dataset.to_excel('2_Data preparation_Dataset.xlsx')
dataset.to_csv('2_Data preparation_Dataset.csv')


# In[ ]:





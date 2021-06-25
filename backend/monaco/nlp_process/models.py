from django.db import models
from wordcloud import WordCloud, STOPWORDS
from dataclasses import dataclass
import pandas as pd
from icecream import ic
from abc import *
import random
import json
import numpy as np
import googlemaps
import matplotlib.pyplot as plt
import platform
from PIL import Image
import nltk
from konlpy.corpus import kobill
from konlpy.tag import Okt; t = Okt()

@dataclass
class File(object):

    context: str
    fname: str
    dframe: object



    @property
    def context(self) -> str: return self._context

    @context.setter
    def context(self, context): self._context = context

    @property
    def fname(self) -> str: return self._fname

    @fname.setter
    def fname(self, fname): self._fname = fname

    @property
    def dframe(self) -> object: return self._dframe

    @dframe.setter
    def dframe(self, dframe): self._dframe = dframe





class Reader(File):

    def new_file(self, file) -> str:
        return file.context + file.fname

    def txt(self, file) -> str:
        return open(f'{self.new_file(file)}').read()

    def img(self, file):
        return np.array(Image.open(f'{self.new_file(file)}'))

    def kobil(self, file) -> str:
        return open(f'{self.new_file(file)}').read()



class Service(Reader):

    def __init__(self):
        self.file = File()
        self.reader = Reader()

    def printer(self):
        file = self.file
        reader = self.reader
        file.context = './data/'
        file.fname = '09. alice_mask.png'
        alice = reader.img(file)
        stopwords = set(STOPWORDS)
        stopwords.add("said")
        path = "c:/Windows/Fonts/malgun.ttf"
        from matplotlib import font_manager, rc
        if platform.system() == 'Darwin':
            rc('font', family='AppleGothic')
        elif platform.system() == 'Windows':
            font_name = font_manager.FontProperties(fname=path).get_name()
            rc('font', family=font_name)
        else:
            print('Unknown system... sorry~~~~')

        plt.figure(figsize=(8, 8))
        plt.imshow(alice, cmap=plt.cm.gray, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def show_present(self):

        f = self.file
        r = self.reader
        f.context = './data/'
        path = "c:/Windows/Fonts/malgun.ttf"

        if platform.system() == 'Darwin':
            rc('font', family='AppleGothic')
        elif platform.system() == 'Windows':
            font_name = font_manager.FontProperties(fname=path).get_name()
            rc('font', family=font_name)
        else:
            print('Unknown system... sorry~~~~')

        plt.rcParams['axes.unicode_minus'] = False

        tmp1 = 'https://search.naver.com/search.naver?where=kin'
        html = tmp1 + '&sm=tab_jum&ie=utf8&query={key_word}&start={num}'

        response = urlopen(html.format(num=1, key_word=urllib.parse.quote('여친 선물')))
        # ic(response)
        soup = BeautifulSoup(response, "html.parser")
        # ic(soup)
        tmp = soup.find_all('strong')
        # ic(tmp)
        tmp_list = []
        for line in tmp:
            tmp_list.append(line.text)
        # ic(tmp_list)

        present_candi_text = []
        for n in tqdm.tqdm(range(1, 1000, 10)):
            response = urlopen(html.format(num=n, key_word=urllib.parse.quote('여자 친구 선물')))
            soup = BeautifulSoup(response, "html.parser")
            tmp = soup.find_all('strong')
            for line in tmp:
                present_candi_text.append(line.text)
            time.sleep(0.5)
        # ic(present_candi_text)

        t = Okt()
        present_text = ''
        for each_line in present_candi_text[:10000]:
            present_text = present_text + each_line + '\n'

        tokens_ko = t.morphs(present_text)
        # ic(tokens_ko)

        ko = nltk.Text(tokens_ko, name='여자 친구 선물')
        ic(len(ko.tokens))
        ic(len(set(ko.tokens)))
        ko.vocab().most_common(100)

        plt.figure(figsize=(15, 6))
        ko.plot(50)
        plt.show()

        data = ko.vocab().most_common(300)
        # for win : font_path='c:/Windows/Fonts/malgun.ttf'
        # /Library/Fonts/AppleGothic.ttf
        wordcloud = WordCloud(font_path=path,
                              relative_scaling=0.2,
                              # stopwords=STOPWORDS,
                              background_color='white',
                              ).generate_from_frequencies(dict(data))
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()

        f.fname = '09. heart.jpg'
        mask = r.img(f)
        image_colors = ImageColorGenerator(mask)
        data = ko.vocab().most_common(200)

        # for win : font_path='c:/Windows/Fonts/malgun.ttf'
        # /Library/Fonts/AppleGothic.ttf
        wordcloud = WordCloud(font_path=path,
                              relative_scaling=0.1, mask=mask,
                              background_color='white',
                              min_font_size=1,
                              max_font_size=100).generate_from_frequencies(dict(data))

        default_colors = wordcloud.to_array()
        plt.figure(figsize=(12, 12))
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def harrypotter(self):
        file = self.file
        reader = self.reader
        file.context = './data/'
        file.fname = '해리포터.txt'
        doc_ko = reader.kobil(file)
        print(doc_ko)

    def harrypotter_series(self):
        file = self.file
        reader = self.reader
        file.context = './data/'
        file.fname = '해리포터.txt'
        doc_ko = reader.kobil(file)
        tokens_ko = t.nouns(doc_ko)
        print(tokens_ko)

    def harrypotter_voca(self):
        file = self.file
        reader = self.reader
        file.context = './data/'
        file.fname = '해리포터.txt'
        doc_ko = reader.kobil(file)
        tokens_ko = t.nouns(doc_ko)
        ko = nltk.Text(tokens_ko, name='해리포터')
        print(len(ko.tokens)) # 문서길이
        print(len(set(ko.tokens)))
        print(ko.vocab)
    def harrypotter_plot(self):
        file = self.file
        reader = self.reader
        file.context = './data/'
        file.fname = '해리포터.txt'
        doc_ko = reader.kobil(file)
        tokens_ko = t.nouns(doc_ko)
        ko = nltk.Text(tokens_ko, name='해리포터')
        plt.rc('font', family='Malgun Gothic')
        plt.figure(figsize=(12, 6))
        ko.plot(50)
        plt.show()



if __name__ == '__main__':
    ns = Service()
    # ns.printer()
    # ns.harrypotter()
    # ns.harrypotter_series()
    # ns.harrypotter_voca()
    # ns.harrypotter_plot()
    ic(ns.alice_printer())

# Create your models here.

from django.db import models
from abc import *
from dataclasses import dataclass
import json
import pandas as pd
import googlemaps
from selenium import webdriver
from icecream import ic

@dataclass
class FileDTO(object):

    context: str
    fname: str
    url: str
    dframe: object


    @property
    def context(self) -> str: return self._context

    @context.setter
    def context(self, context):
        self._context = context

    @property
    def fname(self) -> str: return self._fname

    @fname.setter
    def fname(self, fname): self._fname = fname

    @property
    def dframe(self) -> object: return self._dframe

    @dframe.setter
    def dframe(self, dframe): self._dframe = dframe

    @property
    def url(self) -> object: return self._url

    @url.setter
    def url(self, url): self._url = url

class PrinterBase(metaclass=ABCMeta):
    @abstractmethod
    def dframe(self):
        pass

class ReaderBase(metaclass=ABCMeta):

    @abstractmethod
    def new_file(self):
        pass

    @abstractmethod
    def csv(self):
        pass

    @abstractmethod
    def xls(self):
        pass

    @abstractmethod
    def json(self):
        pass

class ScraperBase(metaclass=ABCMeta):

    @abstractmethod
    def driver(self):
        pass

class Printer(PrinterBase):

    def dframe(self, this):
        ic(type(this))
        ic(this.columns)
        ic(this.head())
        ic(this.isnull().sum())

class Reader(ReaderBase):

    def new_file(self, file) -> str:
        return file.context + file.fname

    def csv(self, file) -> object:
        return pd.read_csv(f'{self.new_file(file)}.csv', encoding='UTF-8', thousands=',')

    def xls(self, file, header, usecols) -> object:
        return pd.read_excel(f'{self.new_file(file)}.xls', header=header, usecols=usecols)



    def json(self, file) -> object:
        return json.load(open(f'{self.new_file(file)}.json', encoding='UTF-8'))

    def gmaps(self) -> object:
        return googlemaps.Client(key='')

class Scraper(ScraperBase):

    def driver(self) -> object:
        return webdriver.Chrome('C:/Program Files/Google/Chrome/chromedriver')

    def auto_login(self):
        pass


if __name__ == '__main__':
    ic('test')
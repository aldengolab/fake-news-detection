import newspaper
import csv
import pandas as pd
import re
import os
import time
import pdb
from multiprocessing import Pool, Process, Lock, Queue

CREDIBLE = './credible.csv'
NONCREDIBLE = './noncredible.csv'
OUTPUT = './new_articles_threading.csv'

class ParallelScraper:

    def __init__(self, credible=CREDIBLE, noncredible=NONCREDIBLE, output=OUTPUT):
        self.credible = pd.read_csv(credible)
        self.noncredible = pd.read_csv(noncredible)
        self.sites = Queue()
        for site in [(x, 0, 0) for x in self.credible.url] + [(x, 1, 0) for x in self.noncredible.url]:
            self.sites.put(site)
        self.output = output
        self.queue = Queue()
        self.uid = 0
        self.print_lock = Lock()
        self.write_lock = Lock()
        self.uid_lock = Lock()
        self.process_pool = []
        self.collect_pool = []

    def run(self, collect_threads=3, process_threads=3):
        '''
        Starts threads to collect articles and threads to read them.
        '''
        print('Starting collection...')
        self.manage_collect(collect_threads)
        time.sleep(120)
        print('Starting processing...')
        self.manage_process(process_threads)
        for p in self.collect_pool:
            p.join()
        if not self.queue.empty():
            # If there's still articles to process, restart processing
            self.manage_process(process_threads)
        for p in self.process_pool:
            p.join()

    def manage_process(self, process_threads):
        '''
        Start given number of threads to multi-process articles.
        '''
        while not self.queue.empty():
            for i in range(process_threads):
                p = Process(target=self.process_articles, args=())
                p.start()
                self.process_pool.append(p)
        self.print_lock.acquire()
        print('No articles found. Ending processing.')
        self.print_lock.release()

    def manage_collect(self, collect_threads):
        '''
        Start a given number of threads to multi-process collection.
        '''
        for i in range(collect_threads):
            p = Process(target=self.collect_articles, args=())
            p.start()
            self.collect_pool.append(p)

    def collect_articles(self):
        '''
        Collects articles from sites, downloads, and adds them to queue for processing.
        '''
        while not self.sites.empty():
            url, label, tries = self.sites.get()
            paper = newspaper.build('http://www.' + url)
            if len(paper.articles) == 0 and tries <= 5:
                tries += 1
                self.sites.put((url, label, tries))
                self.print_lock.acquire()
                print('Error collecting paper from {}; moving to back of queue.'.format(url))
                self.print_lock.release()
            if len(paper.articles) == 0 and tries > 5:
                continue
            else:
                for i in range(len(paper.articles)):
                    article = paper.articles[i]
                    self.queue.put((article, label))

    def process_articles(self):
        '''
        Processes articles in queue.
        '''
        uid = self.get_uid()
        article, label = self.queue.get()
        try:
            row = self.read_article(article, uid, label)
            self.write_to_file(row, self.output)
        except Exception as e:
            print('Error downloading or reading article.')
            print(e)

    def write_to_file(self, row, output, method='a'):
        '''
        Writes result to file.
        '''
        if not os.path.isfile(output) and method == 'a':
            self.write_to_file(['uid','content','source','url','date','author','label'], output, 'w')
        self.write_lock.acquire()
        with open(output, method) as f:
            writer = csv.writer(f)
            writer.writerow(row)
        self.write_lock.release()

    def get_uid(self):
        '''
        Gets a uid for the article.
        '''
        self.uid_lock.acquire()
        uid = self.uid
        self.uid += 1
        self.uid_lock.release()
        return uid

    def read_article(self, article, uid, label):
        '''
        Downloads and reads article. Returns row of information.
        '''
        article.download()
        time.sleep(15)
        article.parse()
        content = article.text
        try:
            author = article.author
        except:
            author = None
        title = article.title
        url = article.url
        source = re.search(r'(?<=://)([\w\d]+\.)[a-zA-z0-9]+[.a-z]*', url).group(0)
        date = article.publish_date
        return [uid, content, source, url, date, author, label]

if __name__ == '__main__':
    scraper = ParallelScraper()
    scraper.run()

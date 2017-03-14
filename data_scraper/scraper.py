import newspaper
import csv
import pandas as pd
import re
import os
import time

CREDIBLE = './credible.csv'
NONCREDIBLE = './noncredible.csv'
OUTPUT = './new_articles.csv'

def main():
    '''
    Runs through all urls from CREDIBLE and NONCREDIBLE, reads articles, then
    writes to file.
    '''
    credible = pd.read_csv(CREDIBLE)
    noncredible = pd.read_csv(NONCREDIBLE)
    queue = [(x, 0, 0) for x in credible.url] + [(x, 1, 0) for x in noncredible.url]
    uid = 0
    while len(queue) > 0:
        url, label, tries = queue.pop(0)
        paper = newspaper.build('http://www.' + url)
        print('Scraping {}'.format(url))
        if len(paper.articles) == 0 and tries <= 5:
            tries += 1
            queue.append((url, label, tries))
            print('Error collecting paper; moving to back to queue.')
        if len(paper.articles) == 0 and tries > 5:
            continue
        else:
            for i in range(len(paper.articles)):
                if uid % 10 == 0:
                    print('.')
                try:
                    row = read_article(paper.articles[i], uid, label)
                    write(row, OUTPUT)
                except Exception as e:
                    print('Error reading article')
                    print(e)
                uid += 1

def write(row, output, method='a'):
    '''
    Writes result to file.
    '''
    if not os.path.isfile(output) and method == 'a':
        write(['uid','content','source','url','date','author','label'], output, 'w')
    with open(output, method) as f:
        writer = csv.writer(f)
        writer.writerow(row)

def read_article(article, uid, label):
    '''
    Downloads and reads article. Returns row of information.
    '''
    article.download()
    time.sleep(5)
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
    main()

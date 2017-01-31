import json
import csv
import pandas as pd

cred = pd.read_csv('credible.csv')
noncred = pd.read_csv('noncredible.csv')
noncred.reset_index(level=0, inplace=True)
cred.columns = ['site','type']
noncred.columns = ['site', 'lang', 'type','notes', 'tmp']
cred['clean_site'] = cred['site'].apply(lambda x: x.split('.')[0].lower())
noncred['clean_site'] = noncred['site'].apply(lambda x: x.split('.')[0].lower())

true = list(cred['clean_site'])
false = list(noncred['clean_site'])
data = []
with open('signalmedia-1m.jsonl') as f:
    for line in f:
        l = json.loads(line)
        #?print(l)
        s = l['source'].replace(' ','').lower()
        if s in true and s not in false:
           print("True ",s)
           l['label']=0
           data.append(l)
        elif s in false and s not in true:
            print("False ",s)
            l['label']=1
            data.append(l)
        else:
            pass
    print(len(data))
    
df = pd.DataFrame(data)
df.to_csv('articles1.csv')

#with open('sources.csv','w',newline='') as f:
#    cw = csv.writer(f)
#    sources = list(sources)
#    for s in sources:
#        print(s)
#        cw.writerow([s])


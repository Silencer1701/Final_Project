import re

from stanfordcorenlp import StanfordCoreNLP
from xml.etree import ElementTree as ET
import os
import glob
import json

nlp = StanfordCoreNLP(r'C:\Users\Silencer\Desktop\Final Project\Code\stanford-corenlp-4.5.1', lang='en')
path = 'C:\\Users\\Silencer\\Desktop\\Final Project\\Code\\English\\'

cata = ['bc', 'bn', 'cts', 'nw', 'un', 'wl']



dict1 = {}
dict2 = {}
dict3 = {}

for x in cata:
    files_path = glob.glob(os.path.join(path + x, '*.sgm'))
    for f in files_path:
        tree = ET.parse(f)
        # 获xml文件的内容取根标签
        root = tree.getroot()
        list1 = []
        list2 = []
        list3 = []
        list4 = []
        count = 0

        for ele in root.iter(tag='SPEAKER'):
            '''a = nlp.pos_tag(ele.tail)
            list1.append(a)

            b = nlp.dependency_parse(ele.tail)
            list2.append(b)

            c = nlp.ner(ele.tail)
            list3.append(c)'''

            stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
            print(ele.tail)
            ele.tail = re.sub(stop,'',ele.tail)
            print(ele.tail)
            d = nlp.word_tokenize(ele.tail)
            list4.append(d)

            '''dict1[root.find('DOCID').text] = list1
            dict2[root.find('DOCID').text] = list2
            dict3[root.find('DOCID').text] = list3'''

            count = count + 1
    '''js1 = json.dumps(dict1)
    file1 = open('dict\\pos_tag_' + x + '.txt', 'w')
    file1.write(js1)
    file1.close()

    js2 = json.dumps(dict2)
    file2 = open('dict\\dependency_parse_' + x + '.txt', 'w')
    file2.write(js2)
    file2.close()

    js3 = json.dumps(dict3)
    file3 = open('dict\\ner_' + x + '.txt', 'w')
    file3.write(js3)
    file3.close()'''

    js4 = json.dumps(list4)
    file4 = open('dict\\tokenize_' + x + '.txt', 'w')
    file4.write(js4)
    file4.close()



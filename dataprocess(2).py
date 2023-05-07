from xml.etree import ElementTree as ET
import os
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords


class EnglishTextUtil:
    STOPCHARS = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'

    def __init__(self):
        self.stemmer = nltk.WordNetLemmatizer()  # Wordnet的词形变换，比nltk.PortStemmer更准确

    # 分词并记录词的位置
    def split_sentence(text):
        wordlist = []
        poslist = []

        total = len(text)
        i = 0
        while i < total:
            while i < total and (text[i] == ' ' or text[i] == '\t'):
                i += 1
            lastidx = i
            while i < total and text[i] != ' ' and text[i] != '\t':
                i += 1
            if i > lastidx:
                wordlist.append(text[lastidx:i])
                poslist.append(lastidx)
        return wordlist, poslist

    # nltk.pos_tag返回的词性转化为 wordnet词性
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    def norm_word(self, word):
        # 去掉I'm 或者 dad's之类的
        idx = word.find("'")
        if idx > 0:
            return word[0:idx]
        return word

    # 英文分词，词性和词形变化, 返回 (原始word, 词性变换，
    def process(self, text):
        # words = nltk.word_tokenize(text) # 分词 这个分词处理了 I'm, Dad's之类的
        words, offsets = EnglishTextUtil.split_sentence(text)  # 分词
        tags = nltk.pos_tag(words)  # 词性标注
        wordnet_tags = [EnglishTextUtil.get_wordnet_pos(tag) for word, tag in tags]  # 转化为wordnet的词性

        # 词形变换的结果
        stem_words = [self.stemmer.lemmatize(word, pos=tag) if bool(tag) else word for word, tag in
                      zip(words, wordnet_tags)]

        # 去掉's 之类的
        stem_words = [self.norm_word(word) for word in stem_words]

        # 去掉可能连着的标点符号
        stem_words = [re.sub(self.STOPCHARS, '', word) for word in stem_words]

        ret = [(word, norm_word, offset, tag[1], wordnet_tag) for word, norm_word, offset, tag, wordnet_tag in
               zip(words, stem_words, offsets, tags, wordnet_tags)]
        return ret

def load_ace2005(datapath="English"):
    etu_processer = EnglishTextUtil()

    sources = os.listdir(datapath)
    totalcount_mention = 0
    totalcount_event = 0
    totalcount_doc = 0
    out_event_mentions = []
    for source in sources:
        subdir = os.path.join(datapath, source, "timex2norm")
        subdir_files = os.listdir(subdir)
        docid_list = [file[:-4] for file in subdir_files if file.endswith(".sgm")]
        for docid in docid_list:
            apffile = os.path.join(subdir, docid + ".apf.xml")

            # tree.findall('.//p')
            tree = ET.parse(apffile)
            root = tree.getroot()
            events = root.findall("./document/event")
            ret_events = []
            for event in events:
                event_type = event.attrib["TYPE"]
                event_subtype = event.attrib["SUBTYPE"]
                event_mentions = event.findall("./event_mention")
                ret_mentions = []
                for mention in event_mentions:
                    charseq = mention.find("ldc_scope/charseq")
                    textstart = int(charseq.attrib["START"])
                    textend = int(charseq.attrib["END"])
                    text = charseq.text
                    text = text.replace("\n", " ") # xml中空行去掉换成空格

                    anchor = mention.find("anchor/charseq")
                    anchorstart = int(anchor.attrib["START"])
                    anchorend = int(anchor.attrib["END"])
                    anchor = anchor.text
                    anchor = anchor.replace("\n", " ")  # xml中空行去掉换成空格

                    textinfo = etu_processer.process(text)  # 四元组 word, norm_word, tag, offset, wordnet_tag
                    words = [(word, norm_word, wordnet_tag, anchorend-textstart>offset>=anchorstart-textstart)
                             for word, norm_word, offset, tag, wordnet_tag in textinfo]

                    ret_mentions.append({"text": text, "anchor": anchor, "words": words})

                    # 一句话中可能出现多次触发词，只选择一个
                    if text[anchorstart-textstart:].find(anchor) != 0:
                        print("====", apffile)
                        print(anchorstart, textstart, anchorstart-textstart, anchor, "<<<", text)
                        print(text.find(anchor))
                        continue
                    # assert(text[anchorstart-textstart:].find(anchor) == 0)

                event_info = {"type": event_type, "subtype": event_subtype, "mentions": ret_mentions}
                totalcount_mention += len(ret_mentions)
                ret_events.append(event_info)
                totalcount_event += 1
                for mention in ret_mentions:
                    mention["type"] = event_type
                    mention["subtype"] = event_subtype
                out_event_mentions.extend(ret_mentions)
            # print(docid, len(ret_events), ret_events)
            totalcount_doc += 1
    print(totalcount_doc, totalcount_event, totalcount_mention)

    # 去重
    ret_event_mentions = []
    textset = dict()
    for item in out_event_mentions:
        text = item.get("text")
        if text not in textset:
            textset[text] = item
            ret_event_mentions.append(item)
        else:
            print("+++>", text)
            print(item["type"], item["subtype"], item["anchor"])
            item = textset.get(text)
            print(item["type"], item["subtype"], item["anchor"])
    print("return total", len(ret_event_mentions))
    return ret_event_mentions


import json
def write_datas(datas):
    with open("eventdata1.json", "w", encoding="utf8") as writer:
        for event in datas:
            ss = json.dumps(event, ensure_ascii=False)
            writer.write(ss+"\n")


if __name__ == '__main__':
    event_mentions = load_ace2005()
    write_datas(event_mentions)


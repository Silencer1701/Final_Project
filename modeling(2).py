# encoding:utf-8
import json
from collections import defaultdict
import numpy as np
import re
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from crf import CRF
from nltk.corpus import stopwords
import sys


START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK = "<UNK>"
EVENTTYPE_IDX_MAP = {"N": 0, "Life": 1, "Movement": 2, "Transaction": 3, "Business": 4, "Conflict": 5, "Contact": 6,
                 "Personnel": 7, "Justice": 8, START_TAG: 9, STOP_TAG: 10}
POSTAG_IDX_MAP = {"": 0, "n": 1, "v": 2, "a": 3, "<PAD>": 4}
STOPCHARS = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'

def pad_datas(vocab, max_seqlen, inputdata):
    wordlist = inputdata["words"]
    postaglist = inputdata.get("postags")
    taglist = inputdata.get("tags")
    # 补齐到最大长度 + START 和 END
    out_tagids = []
    if taglist is not None:
        for tags in taglist:
            tagids = [EVENTTYPE_IDX_MAP.get(START_TAG)] + [EVENTTYPE_IDX_MAP.get(tag, 0) for tag in tags]
            if len(tagids) < max_seqlen:
                tagids += [EVENTTYPE_IDX_MAP.get(STOP_TAG)] * (max_seqlen - len(tagids))
            out_tagids.append(tagids)

    # 词补齐到最大长度 + START 和 END
    out_wordsids = []
    defaultid = vocab.get(UNK)  # 缺省的词
    for words in wordlist:
        ids = [vocab.get(START_TAG)] + [vocab.get(word, defaultid) for word in words]
        if len(ids) < max_seqlen:
            ids += [vocab.get(STOP_TAG)] * (max_seqlen - len(ids))
        out_wordsids.append(ids)

    # 词性补齐到最大长度 + START 和 END
    out_postagids = []  # 词性
    default_postag = 0  # 缺省的词性
    pad_postag = POSTAG_IDX_MAP["<PAD>"]
    for postags in postaglist:
        # print("postags", postags)
        ids = [pad_postag] + [POSTAG_IDX_MAP.get(tag, default_postag) for tag in postags]
        if len(ids) < max_seqlen:
            ids += [pad_postag] * (max_seqlen - len(ids))
        # print("postagids", ids)
        out_postagids.append(ids)

    return {"wordids": out_wordsids, "postagids": out_postagids, "tagids": out_tagids}

from dataprocess import EnglishTextUtil
def preprocess_texts(texts, vocab):
    texts_words = []
    texts_postags = []
    etu_processer = EnglishTextUtil()
    for text in texts:
        # words, words_pos = split_sentence(text)
        # words = [re.sub(STOPCHARS, '', word) for word in words]

        textinfo = etu_processer.process(text)  # 四元组 word, norm_word, tag, offset, wordnet_tag
        words = [norm_word for word, norm_word, offset, tag, wordnet_tag in textinfo]
        postags = [wordnet_tag for word, norm_word, offset, tag, wordnet_tag in textinfo]
        texts_words.append(words)
        texts_postags.append(postags)
    special_words = [START_TAG, STOP_TAG, UNK]
    max_seqlen = max([len(words) for words in texts_words]) + len(special_words) - 1 # 去掉
    outpudata = pad_datas(vocab, max_seqlen, {"words": texts_words, "postags": texts_postags})
    return outpudata

# 载入数据
def load_eventdata(datafile):
    with open(datafile, 'r') as r:
        source_datas = [json.loads(line) for line in r]
    print("texts", len(source_datas))

    out_wordslist = []
    out_taglist = []
    out_postaglist = []  # 词性标注

    for sentence_data in source_datas:
        text = sentence_data["text"]
        anchor = sentence_data["anchor"]
        main_event_type = sentence_data.get('type')
        sub_event_type = sentence_data.get('subtype')
        wordsinfo = sentence_data.get('words')

        words = [item[1] for item in wordsinfo]  # 词形变换后的结果
        postags = [item[2] for item in wordsinfo]
        tags = ['N' for _ in range(len(words))]  # 缺省标签

        for k, (word, normword, postag, anchorflag) in enumerate(wordsinfo):
            if anchorflag:
                tags[k] = main_event_type

        out_wordslist.append(words)
        out_taglist.append(tags)
        out_postaglist.append(postags)

    print("loaded datas", len(out_wordslist), len(out_taglist), len(out_postaglist))

    # 计算词表，计算最大序列长度，词表中前2个保留给
    special_words = [START_TAG, STOP_TAG, UNK]
    vocab = build_vocab(out_wordslist, special_words)
    max_seqlen = max([len(words) for words in out_wordslist]) + len(special_words) - 1 #去掉UNK
    print("**vocab", len(vocab), vocab.get(START_TAG), vocab.get(STOP_TAG))

    inputdata = {"words": out_wordslist, "postags": out_postaglist, "tags": out_taglist}
    padded_datas = pad_datas(vocab, max_seqlen, inputdata)
    out_wordsids = padded_datas.get("wordids")
    out_postagids = padded_datas.get("postagids")
    out_tagids = padded_datas.get("tagids")

    # for i in range(10):
    #     words = [item[1] for item in source_datas[i]["words"]]
    #     print(words)
    #     print(out_wordsids[i])
    #     print(out_postagids[i])
    #     print(out_tagids[i])

    # 校验数据
    for words, tags, wordids, tagids in zip(out_wordslist, out_taglist, out_wordsids, out_tagids):
        # print(words, tags, wordids, tagids)
        assert (len(words) == len(tags))
        assert (len(wordids) == len(wordids))
        assert (len(wordids) == max_seqlen)

    # 信息
    output = inputdata
    for key, val in padded_datas.items():
        output[key] = val
    print("vocab", len(vocab), "max_seqlen", max_seqlen)
    return vocab, max_seqlen, output

# 建立词表，并加入保留的词表
def build_vocab(text_words, special_words):
    wordcounter = defaultdict(int)
    for words in text_words:
        for word in words:
            wordcounter[word] += 1

    wordarr = sorted(wordcounter.items(), key=lambda x:x[1], reverse=True)
    print("vocab", len(wordarr))

    # 英语停用词
    eng_stopwords = set(stopwords.words('english'))
    # 去掉词频少于3的
    wordarr = list(filter(lambda x: (x[0] not in eng_stopwords) and x[1] >= 3, wordarr))
    print("filter vocab", len(wordarr))

    # 特俗词
    special_wordcount = len(special_words)
    vocab = {word: i+special_wordcount for i, (word, count) in enumerate(wordarr)}
    for i, word in enumerate(special_words):
        vocab[word] = i
    return vocab


# 把数据集划分成训练集、验证集和测试集， inputdatas是dict，每个元素是一个list
def split_dataset(inputdatas, train_ratio, val_ratio):
    keys = list(inputdatas.keys())
    total = len(inputdatas[keys[0]])
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # 随机序号列表
    idxes = list(range(total))
    np.random.seed(42)
    np.random.shuffle(idxes)
    np.random.shuffle(idxes)
    np.random.shuffle(idxes)

    trainset = dict()
    valset = dict()
    testset = dict()
    for key, datas in inputdatas.items():
        trainset[key] = [datas[i] for i in idxes[0:train_size]]
        valset[key] = [datas[i] for i in idxes[train_size:train_size+val_size]]
        testset[key] = [datas[i] for i in idxes[train_size+val_size:]]

    print("split_dataset", total, train_size, val_size)
    return trainset, valset, testset


# 保存词表
def save_vocab(vocab, outfile):
    wordarr = sorted(vocab.items(), key=lambda x: x[1])
    with open(outfile, "w", encoding="utf8") as writer:
        for word, idx in wordarr:
            writer.write("%s %s\r\n" %(word, idx))


def load_vocab(vocabfile):
    r = open(vocabfile, "r", encoding="utf8")
    vocab = {}
    for line in r:
        line = line.strip()
        if len(line) <= 0:
            continue
        ss = line.split()
        if len(ss) != 2:
            continue
        word = ss[0]
        idx = int(ss[1])
        vocab[word] = idx
    return vocab


class BiLSTM_CRF(nn.Module):
    # num_labels：输出类型数量  hidden_dim：双向lstm的总的隐藏层维度
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, use_postag):
        super(BiLSTM_CRF, self).__init__()
        self.batch_first = True
        self.use_postag = use_postag  # 是否使用词性
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.postag_embeds = nn.Embedding(len(POSTAG_IDX_MAP), embedding_dim) if use_postag else None
        self.num_tags = num_tags
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=self.batch_first)
        self.classifier = nn.Linear(hidden_dim, self.num_tags)
        self.crf = CRF(self.num_tags, batch_first=self.batch_first)

    def forward(self, inputdata):
        wordids = inputdata["ids"]
        embeds = self.word_embeds(wordids)

        if self.use_postag:
            postagids = inputdata["postagids"]
            # print("postagids", postagids.shape, postagids)
            # print("embeds", embeds.shape, embeds)
            postag_embeds = self.postag_embeds(postagids)
            # print("postag_embeds", postag_embeds.shape, postag_embeds)
            embeds = embeds.add(postag_embeds)
            # print("combined embeds", embeds.shape, embeds)

        # print("embeds", embeds.shape)
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # lstm输出 (output
        sequence_output, (ht, hc) = self.lstm(embeds)
        # print("lstm_out", sequence_output.shape, ht.shape, hc.shape)

        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)

        labels = inputdata.get("labels")
        if labels is not None:  # 训练的情况
            # , mask=attention_mask
            # print("logits", logits)
            # print("labels", labels)
            loss = self.crf(emissions=logits, tags=labels)
            predicts = self.crf.decode(logits)
            # print("train res:", labels.shape, predicts.shape, "label", labels[0])
            # print("predicts", predicts[0][0])
            outputs = (-1 * loss,) + outputs
        else:
            # print("sequence_output", sequence_output.shape)
            # print("logits", logits.shape, logits)
            tags = self.crf.decode(logits)
            # print("tags", tags)
            tags = tags.squeeze(0).cpu().numpy().tolist()
            outputs = tags
        #                     tags = self.model.crf.decode(logits, inputs['attention_mask'])
        #                     tags = tags.squeeze(0).cpu().numpy().tolist()
        return outputs


EMBEDDING_DIM = 100
HIDDEN_DIM = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_dataset(model, dataset, use_postag):
    batch_size = 128
    datakeys = list(dataset.keys())
    # print("eval dataset", datakeys, len(dataset[datakeys[0]]))
    wordsids = dataset["wordids"]
    tagids = dataset["tagids"]
    postagids = dataset["postagids"]

    start_tagid = EVENTTYPE_IDX_MAP[START_TAG]
    stop_tagid = EVENTTYPE_IDX_MAP[STOP_TAG]

    model.eval()

    total_errcnt = 0    # 词语级别错误数
    word_acc = 0  # 词语级别的正确率
    sentence_correct_count = 0  # 句子级别的正确数
    for start in range(0, len(wordsids), batch_size):
        end = min(len(wordsids), start + batch_size)
        batch_wordsids = wordsids[start:end]
        targets = tagids[start:end]
        wordid_feats = torch.tensor(batch_wordsids, dtype=torch.long).to(device)
        if use_postag:
            batch_postagids = postagids[start:end]
            postag_feats = torch.tensor(batch_postagids, dtype=torch.long).to(device)
            predicts = model({"ids": wordid_feats, "postagids": postag_feats})
        else:
            predicts = model({"ids": wordid_feats})

        for ti, pi in zip(targets, predicts):
            # print("++>", ti)
            # print("==>", pi)
            accarr = [1 if targ == pred else 0 for targ, pred in zip(ti, pi)
                   if (targ not in [start_tagid, stop_tagid])]  # and (targ != 0 or pred != 0)

            class_accarr = [1 if targ == pred else 0 for targ, pred in zip(ti, pi)
                      if (targ not in [start_tagid, stop_tagid])]  # and (targ != 0 or pred != 0)

            acc = sum(accarr) / len(accarr)
            total_errcnt += int(len(accarr) - sum(accarr))
            if sum(accarr) == len(accarr):
                sentence_correct_count += 1
            # print("**", acc, sum(accarr), len(accarr), accarr)
            word_acc += acc
    sentence_acc = sentence_correct_count / len(wordsids)  # 句子级别正确率
    word_acc = word_acc / len(wordsids)  # 单词级别正确率
    return word_acc, sentence_acc, total_errcnt


def train(use_postag=True):
    datafile = "eventdata.json"

    # 全部数据
    vocab, max_seqlen, inputdatas = load_eventdata(datafile)

    # 划分数据集
    trainset, devset, testset = split_dataset(inputdatas, 0.8, 0.1)
    vocab_size = len(vocab)
    num_tags = len(EVENTTYPE_IDX_MAP)

    model = BiLSTM_CRF(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_tags, use_postag=use_postag)
    optimizer = optim.SGD(model.parameters(), lr=0.02, weight_decay=1e-4)
    optimizer2 = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    batch_size = 1024
    epoches = 1000  # 600

    model = model.to(device)
    model.train()
    print("model", model)

    datakeys = list(trainset.keys())
    print("trainset", datakeys, len(trainset[datakeys[0]]))
    wordsids = trainset["wordids"]
    tagids = trainset["tagids"]
    postagids = trainset["postagids"]

    save_vocab(vocab, "myModel.vocab")

    best_acc = 0
    best_epoch = 0

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(epoches):  # again, normally you would NOT do 300 epochs, it is toy data
        epoch_loss = 0
        model.train()
        for start in range(0, len(wordsids), batch_size):
            end = min(len(wordsids), start+batch_size)
            batch_wordsids = wordsids[start:end]
            batch_tagids = tagids[start:end]


            # print("====words", np.shape(batch_wordsids), batch_wordsids[0])
            # print(np.shape(batch_tagids), batch_tagids[0])
            # print(np.shape(batch_postagids), batch_postagids[0])


            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            # model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            targets = torch.tensor(batch_tagids, dtype=torch.long).to(device)
            wordid_feats = torch.tensor(batch_wordsids, dtype=torch.long).to(device)

            if use_postag:
                batch_postagids = postagids[start:end]
                postag_feats = torch.tensor(batch_postagids, dtype=torch.long).to(device)
                loss, outputs = model({"ids": wordid_feats, "postagids": postag_feats, "labels": targets})
            else:
                loss, outputs = model({"ids": wordid_feats, "labels": targets})
            # Step 3. Run our forward pass.
            # print('feats', feats.shape, feats[0])
            # print('targets', targets.shape, targets[0])


            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            if epoch < 600:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()

            epoch_loss += loss

        word_acc, sentence_acc, errcnt = eval_dataset(model, devset, use_postag)
        print("epoch %d train loss %.3f eval word acc: %.5f  sentence acc: %.5f errcnt %d [best:%d %.5f]"
                %(epoch, epoch_loss, word_acc, sentence_acc, errcnt, best_epoch, best_acc))

        # 保存最好的模型
        if word_acc > best_acc:
            torch.save(model, 'myModel.pt')
            best_acc = word_acc
            best_epoch = epoch
    word_acc, sentence_acc, errcnt = eval_dataset(model, testset, use_postag)
    print("test word acc: %.5f, sentence acc: %.5f, errcnt %d" %(word_acc, sentence_acc, errcnt))



    return model, vocab


def load_model(modelfile='myModel.pt', vocabfile="myModel.vocab"):
    vocab = load_vocab(vocabfile)
    model = torch.load(modelfile)
    return vocab, model

def predict(model=None, vocab=None, use_postag=True):
    if model is None or vocab is None:
        vocab, model = load_model()

    # 模型
    model.eval()
    model.to(device)
    print("vocab", len(vocab), "model", model)

    # Check predictions after training
    print("Please input...")
    while True:
        text = input()
        featdatas = preprocess_texts([text], vocab)
        print("featdatas", list(featdatas.keys()))
        wordsids = featdatas["wordids"]
        postagids = featdatas["postagids"]

        # print(out_wordsids)
        id_feat = torch.tensor(wordsids, dtype=torch.long).to(device)
        feats = {"ids": id_feat}
        if use_postag:
            feats["postagids"] = torch.tensor(postagids, dtype=torch.long).to(device)

        # print("feat", feat)
        with torch.no_grad():
            res = model(feats)
            print(res)


def batch_test(model=None, vocab=None, use_postag=True):
    if model is None or vocab is None:
        vocab, model = load_model()

    # 模型
    model.eval()
    model.to(device)
    print("vocab", len(vocab), "model", model)

    datafile = "eventdata.json"
    vocab, max_seqlen, inputdatas = load_eventdata(datafile)
    wordslist = inputdatas["words"]
    texts = [" ".join(words) for words in wordslist]

    for text in texts:
        featdatas = preprocess_texts([text], vocab)

        wordsids = featdatas["wordids"]
        postagids = featdatas["postagids"]

        # print(out_wordsids)
        id_feat = torch.tensor(wordsids, dtype=torch.long).to(device)
        feats = {"ids": id_feat}
        if use_postag:
            feats["postagids"] = torch.tensor(postagids, dtype=torch.long).to(device)

        with torch.no_grad():

            word_acc, sentence_acc, errcnt = eval_dataset(model, feats, use_postag)
            print("test word acc: %.5f, sentence acc: %.5f, errcnt %d" % (word_acc, sentence_acc, errcnt))
            # res = model(feats)[0]
            # # if res[0]!=9 and res[-1]!=10:
            # flag = False
            # for i in range(1, len(res) - 1):
            #     if res[i] != 0:
            #         flag = True
            # if flag:
            #     print(text, res)


if len(sys.argv) > 1 and sys.argv[1] == "train":
    model, vocab = train(True)
    predict(model, vocab)
elif len(sys.argv) > 1 and sys.argv[1] == "test":
    model, vocab = None, None
    batch_test(model, vocab)
else:
    model, vocab = None, None
    predict(model, vocab)

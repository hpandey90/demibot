UNK = 'unk'
VOCAB_SIZE = 8000
MAXQ =  50
MINQ =  2
MAXA =  50
MINA =  2

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space included

from collections import defaultdict
import numpy as np



'''
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''
def get_id2line():
    lines = open('./data/raw_data/movie_lines.txt', encoding='utf-8', errors ='ignore').read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line


'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''
def get_conversations():
    conv = open('./data/raw_data/movie_conversations.txt', encoding='utf-8', errors ='ignore').read().split('\n')
    convs = []
    for line in conv[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    return convs


'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''
#def gather_dataset(convs, id2line):
    #return questions,answers


def gather_dataset(convs, id2line):
    questions = []; answers = []

    for conv in convs:
        if len(conv) %2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i%2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers


def filter_line(line,whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])

def filter_data(qseq,aseq):
    filtered_q, filtered_a =[], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    for i in range(raw_data_len):
        qlen,alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= MINQ and qlen <= MAXQ:
            if alen >= MINA and alen <= MAXA:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    print('*filtered conv :')

    for q,a in zip(filtered_q[147:152], filtered_a[147:152]):
        print('q : [{0}]; a : [{1}]'.format(q,a))

    return filtered_q,filtered_a


def process_data():

    id2line = get_id2line()
    print('>> gathered id2line dictionary.\n')
    convs = get_conversations()
    print(convs[151:156])
    print('>> gathered conversations.\n')

    questions, answers = gather_dataset(convs,id2line)

    # change to lower case (just for en)
    questions = [ line.lower() for line in questions ]
    answers = [ line.lower() for line in answers ]

    for q,a in zip(questions[175:180],answers[175:180]):
        print('q : [{0}]; a :[{1}]'.format(q,a))

    #filter unwanted characters
    questions = [ filter_line(line,EN_WHITELIST) for line in questions ]
    answers = [ filter_line(line,EN_WHITELIST) for line in answers ]

    #discard long or short sentences
    que, ans = filter_data(questions,answers)

    #seperating words for tokeninzation
    print('\n>> Segment lines into words')
    qtokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in que ]
    atokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in ans ]
    print('\n:: Sample from segmented list of words')

    for q,a in zip(qtokenized[175:180], atokenized[175:180]):
        print('q : [{0}]; a : [{1}]'.format(q,a))

    #print(questions[121:125],answers[121:125])


if __name__ == '__main__':
    process_data()

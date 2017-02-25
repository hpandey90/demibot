UNK = 'unk'
VOCAB_SIZE = 8000

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

def process_data():

    id2line = get_id2line()
    print('>> gathered id2line dictionary.\n')
    convs = get_conversations()
    print(convs[121:125])
    print('>> gathered conversations.\n')

    questions, answers = gather_dataset(convs,id2line)

    # change to lower case (just for en)
    questions = [ line.lower() for line in questions ]
    answers = [ line.lower() for line in answers ]

    print(questions[121:125],answers[121:125])


if __name__ == '__main__':
    process_data()

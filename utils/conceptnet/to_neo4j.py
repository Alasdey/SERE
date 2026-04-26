'''
Copy from https://github.com/redsk/neo_concept/tree/master
'''


import json
import os
import re
import sys

from tqdm import tqdm

nodeid = 0
nodes = {}
nf = 0

triplets = set()


def add_node(node):
    global nodes
    global nf
    if node in nodes:
        nodes[node] = nodes[node] + 1
    else:
        nodes[node] = 1
        nf.write(node + ',' + "Concept\n")


def isEnglish(concept):
    if concept.split('/')[2] == 'en':
        return True
    return False


def remove_comma(concept):
    return re.sub(r',', r' ', concept)


def cn5ToCSV(inputDir, ALL_LANGUAGES=False):
    global nodes
    global nodeid
    global nf
    nodes = {}
    nodeid = 0

    global sources
    sources = {}

    nf = open('nodes.csv', "w", encoding='utf-8')
    nf.write('id:ID,:LABEL\n')
    ef = open('edges.csv', "w", encoding='utf-8')
    ef.write(':START_ID,:END_ID,:TYPE,weight:float\n')

    for filename in [os.path.join(inputDir, 'assertions.csv')]:
        total_line = 0

        with open(filename, "r", encoding='utf-8') as f:
            for _ in f:
                total_line += 1
        print(total_line)
        with open(filename, "r", encoding='utf-8') as f:
            for line in tqdm(f, total=total_line):
                tokens = line[0:-1].split('\t')

                if (ALL_LANGUAGES or (isEnglish(tokens[2]) and isEnglish(tokens[3]))):
                    fromN = remove_comma(esc(tokens[2].split('/')[3])).strip()
                    toN = remove_comma(esc(tokens[3].split('/')[3])).strip()

                    add_node(fromN)
                    add_node(toN)

                    relType = remove_comma(esc(tokens[1]).split('/')[2])
                    weight = remove_comma(escFloat(str(json.loads(tokens[4])["weight"])))

                    triplet = fromN + ',' + toN + ',' + relType + ',' + weight + '\n'
                    if fromN != toN and triplet not in triplets:
                        ef.write(triplet)

                        triplets.add(triplet)

    nf.close()
    ef.close()

def esc(s):
    s = re.sub(r"\\", "", s)
    s = re.sub(r"\"", "\\\"", s)
    return s


def escFloat(s):
    return re.sub(r"L", "", s)


def main():
    numargs = len(sys.argv)
    if numargs == 3:
        cn5ToCSV(sys.argv[1], True)
    if numargs == 2:
        cn5ToCSV(sys.argv[1], False)
    if numargs < 2 or numargs > 3:
        print("Usage:\npython convertcn.py <input directory> [ALL_LANGUAGES]\n")


if __name__ == "__main__":
    main()

import argparse
import os
import random
from collections import defaultdict
from pprint import pprint

import nltk
from tqdm import tqdm

import utils.dataset_utils
import utils.io

SQUAD_VERSION_TWO = "v2.0"


class StatType(object):
    NUMBER_OF_DOCS = 'number of docs'
    AVG_TRUNCATED_DOC_LENGTH = 'average length of context, i.e. after doc truncation, (chars)'
    AVG_DOC_LENGTH = 'average length of documents (chars)'
    NUMBER_OF_QUESTIONS = 'number of questions'
    AVG_QUERY_LENGTH = 'average length of questions (chars)'
    NUMBER_QUESTIONS_WITH_NO_ANSWER = 'number of questions with NO consensus answer'
    NUMBER_OF_NON_NULL_ANSWERS = 'number of non null consensus answers'
    AVG_ANSWER_LENGTH = 'average length of answers (chars)'


def get_text(qad, domain):
    local_file = os.path.join(args.web_dir,
                              qad['Filename']) if domain == 'SearchResults' else os.path.join(
        args.wikipedia_dir, qad['Filename'])
    return utils.io.get_file_contents(local_file, encoding='utf-8')


def select_relevant_portion(text):
    paras = text.split('\n')
    selected = []
    done = False
    for para in paras:
        sents = sent_tokenize.tokenize(para)
        for sent in sents:
            words = nltk.word_tokenize(sent)
            for word in words:
                selected.append(word)
                if len(selected) >= args.max_num_tokens:
                    done = True
                    break
            if done:
                break
        if done:
            break
        selected.append('\n')
    st = ' '.join(selected).strip()
    return st


def add_triple_data(datum, page, domain):
    qad = {'Source': domain}
    for key in ['QuestionId', 'Question', 'Answer']:
        qad[key] = datum[key]
    for key in page:
        qad[key] = page[key]
    return qad


def get_qad_triples(data):
    qad_triples = []
    for datum in data['Data']:
        for key in ['EntityPages', 'SearchResults']:
            for page in datum.get(key, []):
                qad = add_triple_data(datum, page, key)
                qad_triples.append(qad)
    return qad_triples


def convert_to_squad_format(qa_json_file, squad_file):
    stats = defaultdict(float)
    qa_json = utils.dataset_utils.read_triviaqa_data(qa_json_file)
    stats[StatType.NUMBER_OF_QUESTIONS] = len(qa_json)
    qad_triples = get_qad_triples(qa_json)

    random.seed(args.seed)
    random.shuffle(qad_triples)

    data = []

    for qad in tqdm(qad_triples):
        stats[StatType.NUMBER_OF_DOCS] += 1
        qid = qad['QuestionId']

        text = get_text(qad, qad['Source'])
        stats[StatType.AVG_DOC_LENGTH] += len(text)
        selected_text = select_relevant_portion(text)
        stats[StatType.AVG_TRUNCATED_DOC_LENGTH] += len(selected_text)

        question = qad['Question']
        stats[StatType.AVG_QUERY_LENGTH] += len(question)
        para = {'context': selected_text, 'qas': [{'question': question, 'answers': []}]}
        data.append({'paragraphs': [para]})
        qa = para['qas'][0]
        qa['id'] = utils.dataset_utils.get_question_doc_string(qid, qad['Filename'])
        qa['qid'] = qid

        ans_string, index = utils.dataset_utils.answer_index_in_document(qad['Answer'],
                                                                         selected_text)
        if index == -1:
            stats[StatType.NUMBER_QUESTIONS_WITH_NO_ANSWER] += 1
            qa['is_impossible'] = True
            # if qa_json['Split'] == 'train':
            #     continue
        else:
            qa['is_impossible'] = False
            qa['answers'].append({'text': ans_string, 'answer_start': index})
            stats[StatType.AVG_ANSWER_LENGTH] += len(ans_string)
            stats[StatType.NUMBER_OF_NON_NULL_ANSWERS] += 1

            # if qa_json['Split'] == 'train' and len(data) >= args.sample_size and qa_json['Domain'] == 'Web':
            #     break

    stats[StatType.AVG_TRUNCATED_DOC_LENGTH] /= float(stats[StatType.NUMBER_OF_DOCS])
    stats[StatType.AVG_DOC_LENGTH] /= float(stats[StatType.NUMBER_OF_DOCS])
    stats[StatType.AVG_ANSWER_LENGTH] /= float(stats[StatType.NUMBER_OF_NON_NULL_ANSWERS])
    stats[StatType.AVG_QUERY_LENGTH] /= float(stats[StatType.NUMBER_OF_DOCS])
    pprint(stats)

    squad = {'data': data, 'version': SQUAD_VERSION_TWO}
    utils.io.write_json_to_file(squad, squad_file)
    print('Added', len(data))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--triviaqa_file', help='Triviaqa file')
    parser.add_argument('--squad_file', help='Squad file')
    parser.add_argument('--wikipedia_dir', help='Wikipedia doc dir')
    parser.add_argument('--web_dir', help='Web doc dir')

    parser.add_argument('--seed', default=10, type=int, help='Random seed')
    parser.add_argument('--max_num_tokens', default=800, type=int,
                        help='Maximum number of tokens from a document')
    # parser.add_argument('--sample_size', default=80000, type=int, help='Random seed')
    parser.add_argument('--tokenizer', default='tokenizers/punkt/english.pickle',
                        help='Sentence tokenizer')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    sent_tokenize = nltk.data.load(args.tokenizer)
    convert_to_squad_format(args.triviaqa_file, args.squad_file)

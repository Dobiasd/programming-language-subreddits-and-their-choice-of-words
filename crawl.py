#!/usr/bin/env python
# -*- coding: ascii -*-

"""crawl subreddit comments
"""

__author__ = 'Tobias Hermann'
__version__ = '1.0.0'

from collections import Counter
import datetime
import csv
import cPickle
import functools
import pprint
import operator
import os
import re
import sqlite3
import subprocess
import numpy as np
import matplotlib.pyplot as plt

languages = [
    ('actionscript', ['actionscript']),
    ('ada', ['ada']),
    ('asm', ['assembler', 'asm']),
    ('brainfuck', ['brainfuck']),
    ('c_programming', ['c_programming']), # 'c'
    ('clojure', ['clojure']),
    ('cobol', ['cobol']),
    ('cpp', ['c++', 'cpp']),
    ('csharp', ['c#', 'c sharp', 'c-sharp', 'csharp']),
    ('d_language', ['dlang', 'd_language']), # 'd'
    ('dartlang', ['dart', 'dartlang']),
    ('delphi', ['delphi']),
    ('elm', ['elm', 'elm-lang']),
    ('erlang', ['erlang']),
    ('forth', ['forth']),
    ('fortran', ['fortran']),
    ('fsharp', ['f#', 'f sharp', 'f-sharp', 'fsharp']),
    ('golang', ['golang']), # 'go'
    ('groovy', ['groovy']),
    ('haskell', ['haskell']),
    ('haxe', ['haxe']),
    ('java', ['java']),
    ('javascript', ['javascript']), # 'js'
    ('lisp', ['lisp']),
    ('lua', ['lua']),
    ('mathematica', ['mathematica']),
    ('matlab', ['matlab']),
    ('objectivec', ['objective-c', 'objective c', 'objectivec']),
    ('ocaml', ['ocaml']),
    ('pascal', ['pascal']),
    ('perl', []), # 'perl'
    ('php', ['php']),
    ('prolog', ['prolog']),
    ('python', ['python']),
    ('ruby', ['ruby']),
    ('rust', ['rust']),
    ('scala', ['scala']),
    ('scheme', ['scheme']),
    ('scratch', ['scratch']),
    ('shell', []), # 'shell'
    ('smalltalk', []), # 'smalltalk'
    ('sml', ['standard ml', 'sml']),
    ('sql', ['sql']),
    ('swift', ['swift']),
    ('tcl', ['tcl']),
    ('visualbasic', ['visual basic', 'visualbasic', 'vb'])
    ]

subreddits = map(lambda x: x[0], languages)

emotions = [
    ('beautiful', 'ugly'),
    ('clean', 'dirty'),
    ('clever', 'stupid'),
    ('friend', 'enemy'),
    ('genius', 'idiot'),
    ('good', 'bad'),
    ('happy', 'sad'),
    ('heaven', 'hell'),
    ('interesting', 'boring'),
    ('joy', 'grief'),
    ('love', 'hate'),
    ('pretty', 'disgusting'),
    ('pride', 'shame'),
    ('relief', 'frustration'),
    ('simple', 'complicated'),
    ('sympathy', 'cruelty'),
    ('concise', 'verbose'),
    ('awesome', 'terrible'),
    ('helpful', 'asshole'),
    ('simplification', 'boilerplate'),
    ('fun', 'fuck'),
    ('nice', 'damn'),
    ('perfect', 'crap'),
    ('cool', 'shit'),
    ('great', 'suck'),
    ('thank you', 'fuck you'),
    ('pleasure', 'pain'),
    ('terse', 'bloated'),
    ('easy', 'difficult'),
    ('exciting', 'tedious'),
    ('fast', 'slow'),
    ('better', 'worse'),
    ('hope', 'fear'),
    ('succinct', 'long-winded'),
    ('awesomeness', 'sucks'),
    ('funny', 'fucking'),
    ('nicer', 'shitty')
    ]

positive_emotions = sorted(map(operator.itemgetter(0), emotions))
negative_emotions = sorted(map(operator.itemgetter(1), emotions))

#http://en.wiktionary.org/wiki/Appendix:English_internet_slang
internet_slang_words = [
    'afaik',
    'afaiwafk',
    'asap',
    'flamer',
    'ftw',
    'fyi',
    'gtfo',
    'imo',
    'imho',
    'lame',
    'leet',
    'lol',
    'omg',
    'omfg',
    'stfu',
    'stfw',
    'tl;dr',
    'troll',
    'wtf',
    'yagni',
    'yolo',
    'zomg',
    'rtfm',
    'rofl'
    ]

opposite_words = [
#http://en.wikipedia.org/wiki/Contrasting_and_categorization_of_emotions
    'abstract', 'concrete',
    'algorithm', 'data',
    'book', 'course',
    'bug', 'feature',
    'church', 'turing',
    'compiler', 'interpreter',
    'desire', 'disgust',
    'efficient', 'effective',
    'general', 'specific',
    'hobby', 'work',
    'know', 'guess',
    'library', 'application',
    'nerd', 'geek',
    'open', 'closed',
    'practice', 'theory',
    'pure', 'impure',
    'refactoring', 'rewrite',
    'software', 'hardware',
    'solution', 'workaround',
    'static', 'dynamic',
    'syntax', 'semantics',
    'thanks', 'please',
    'think', 'feel']

single_words = [
    'analysis',
    'autism',
    'blog',
    'category',
    'cloud',
    'coffee',
    'community',
    'cost',
    'deadline',
    'debug',
    'design',
    'device',
    'distributed',
    'documentation',
    'error',
    'esoteric',
    'finance',
    'freak',
    'function',
    'functional',
    'github',
    'goto',
    'hacker',
    'homework',
    'idea',
    'idiom',
    'information',
    'intelligence',
    'language',
    'learn',
    'logic',
    'machine',
    'math',
    'model',
    'money',
    'newb',
    'newbie',
    'object',
    'optimal',
    'overhead',
    'paralell',
    'pattern',
    'performance',
    'pragmatic',
    'problem',
    'procedural',
    'progress',
    'quality',
    'reason',
    'religion',
    'robotics',
    'safety',
    'school',
    'science',
    'sicp',
    'sorry',
    'speed',
    'types',
    'understand',
    'usability',
    'user',
    'web']



# http://en.wikipedia.org/wiki/List_of_emoticons
emoticons = [
    ':-)',
    ';-)',
    ':)',
    ';)',
    '=)',
    ':>',

    ':-(',
    ';-/',
    ":'-(",
    ":'(",


    '^_^',
    '^_-',
    '-_^',
    '-_-',
    'O_O',
    'o_o',
    '^.^',
    '(._.)',
    '(,_,)',
    'T_T'
    ]





# http://www.reddit.com/r/redditdev/comments/2c1iv5/how_to_get_all_comments_ever_written_in_a/


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def utf_to_ascii(str):
    return str.encode('ascii', 'ignore')

def replace_unwanted_chars(str):
    return str.replace(',', '-').replace('\r', '-').replace('\n', '-').replace(':', '-')

def submission_meta(submission):
    return 'created_utc:' + str(int(submission.created_utc)) +\
           '\nups:' + str(submission.ups) +\
           '\ndowns:' + str(submission.downs) +\
           '\ntitle:' + replace_unwanted_chars(utf_to_ascii(submission.title)) +\
           '\nnum_comments:' + str(submission.num_comments) +\
           '\nauthor:' + replace_unwanted_chars(utf_to_ascii(str(submission.author))) +\
           '\nselfText:' + replace_unwanted_chars(utf_to_ascii(submission.selftext))

def comment_meta(comment):
    return 'created_utc:' + str(int(comment.created_utc)) +\
           '\nups:' + str(comment.ups) +\
           '\ndowns:' + str(comment.downs) +\
           '\nauthor:' + replace_unwanted_chars(utf_to_ascii(str(comment.author))) +\
           '\nbody:' + replace_unwanted_chars(utf_to_ascii(comment.body))

def write_str_to_file(path, str):
    with open(path, 'w') as text_file:
        text_file.write(utf_to_ascii(str))

def get_comments():
    import praw
    r = praw.Reddit('Comment Scraper 1.0 by u/_Daimon_ see '
                    'https://praw.readthedocs.org/en/latest/'
                    'pages/comment_parsing.html')

    make_dir('comments')

    for i, subreddit in enumerate(subreddits):
        with open('submissions/' + subreddit + '.txt') as text_file:
            ids = text_file.read().split('\n')
            ids = filter(None, ids)

        #r.login('bot_username', 'bot_password')
        for j, submission_id in enumerate(ids):
            try:
                submission = r.get_submission(submission_id=submission_id)
                subm_dir_base = 'comments/' + subreddit
                submission_dir = subm_dir_base + '/' + submission_id
                make_dir(submission_dir)
                write_str_to_file(submission_dir + '.txt', (submission_meta(submission)))
                submission.replace_more_comments(limit=None)
                flat_comments = list(praw.helpers.flatten_tree(submission.comments))
                print 'submission %s (%d/%d, %d comments, subreddit %d/%d - %s)' %\
                    (submission_id, j + 1, len(ids), len(flat_comments), i + 1,
                     len(subreddits), subreddit)
                for comment in flat_comments:
                    if not hasattr(comment, 'id') or not hasattr(comment, 'body'):
                        continue
                    comment_path = submission_dir + '/' + comment.id + ".txt"
                    write_str_to_file(comment_path, comment_meta(comment))
            except Exception as e:
                print "Exception:", str(e)
                print e.__doc__
                print e.message


def get_today_start_as_unix_timestamp():
    now = datetime.datetime.now()
    sec_since_epoch = (now.date() - datetime.date(1970, 1, 1)).total_seconds()
    return int(sec_since_epoch)

def get_submission_ids():
    days_to_go_back = 365
    startT = get_today_start_as_unix_timestamp()
    startT = 1406764800
    bash_script_file_name = 'temp_parse_submission_ids.sh'
    with open(bash_script_file_name, 'w') as bash_script:
        bash_script.write('mkdir days\n')
        bash_script.write('mkdir submissions\n')
        for subreddit in subreddits:
            bash_script.write('mkdir days/%s\n' % subreddit)
            bash_script.write('mkdir comments/%s\n' % subreddit)
            t = startT
            day_cnt = 0
            while t > 1119312000:
                day_cnt += 1
                if day_cnt > days_to_go_back:
                    break
                newT = t - (60*60*24)
                # Limit is set to 1000 posts,
                # although reddit only shows 100 at a time.
                # But usually this should be enough
                # to get the most important posts per day.
                bash_script.write('wget -t 32 -T 16 "http://www.reddit.com/r/%s/search?q=timestamp:%d..%d&sort=top&restrict_sr=on&syntax=cloudsearch&limit=1000" -O days/%s/%d-%d\n' % (subreddit, newT, t, subreddit, newT, t))
                t = newT
            bash_script.write("grep -r days/%s -e comments | perl -pe 's/%s\\/comments\\/([a-z0-9]{4,7})/\\nregex_marker_start\\1regex_marker_end\\n/gi' | grep regex_marker_start | perl -pe 's/regex_marker_start(.*)regex_marker_end/\\1/g' | sort | uniq > submissions/%s.txt\n" % (subreddit, subreddit, subreddit))

    #subprocess.call(['bash', bash_script_file_name])
    #os.remove(bash_script_file_name)

def load_comment(path):
    lines = [line.strip() for line in open(path)]
    pairs = [line.split(":", 1) for line in lines]
    return dict(pairs)

def load_comments(subreddit):
    comments = {}
    mainDir = "comments/" + subreddit
    for submission_id in os.listdir(mainDir):
        path = os.path.join(mainDir, submission_id)
        if os.path.isdir(path):
            subDirPath = path
            for fileName in os.listdir(subDirPath):
                if fileName.endswith(".txt"):
                    filePath = os.path.join(subDirPath, fileName)
                    comment_id = fileName[:-4]
                    comments[comment_id] = load_comment(filePath)
                    comments[comment_id]['submission_id'] = submission_id
    return comments


def pickle_comments():
    allComments = {}
    for subreddit, aliases in languages:
        print subreddit
        allComments[subreddit] = load_comments(subreddit)
    write_str_to_file("analysis/all_comments_dict.pickle",
        cPickle.dumps(allComments))

def minus_pad(s):
    return '-' + s + '-'

def prepare_comment_body(s):
    return re.sub(r'[^a-zA-Z+#_-]', '-', s)

def comments_to_db():
    all_comments = cPickle.load(open("analysis/all_comments_dict.pickle", "rb"))
    print 'len(all_comments):', len(all_comments)

    dbPath = 'analysis/comments.db'
    if os.path.exists(dbPath):
        os.remove(dbPath)
    conn = sqlite3.connect(dbPath)
    c = conn.cursor()
    # todo: submissions table erstellen und befuellen?

    c.execute('''CREATE TABLE subreddits (name text)''')
    c.execute('''CREATE TABLE aliases(alias text, subreddit text)''')

    for subreddit, aliases in languages:
        c.execute('INSERT INTO subreddits VALUES (?)', (subreddit, ))
        for alias in aliases:
            c.execute('INSERT INTO aliases VALUES (?, ?)', (alias, subreddit))


    c.execute('''CREATE TABLE comments
             (id text, subreddit text, submission_id text, created_utc integer, ups integer, downs integer, author text, body text)''')
    for subreddit, comments in all_comments.iteritems():
        print subreddit
        for comment_id, comment in comments.iteritems():
            command = 'INSERT INTO comments VALUES(?, ?, ?, ?, ?, ?, ?, ?)'
            c.execute(command, (comment_id,
                                subreddit,
                                comment['submission_id'],
                                comment['created_utc'],
                                comment['ups'],
                                comment['downs'],
                                comment['author'],
                                minus_pad(prepare_comment_body(comment['body']))))

    conn.commit()
    conn.close()

def get_mention_count(c, mentioner, mentionee):
    command = "SELECT COUNT(comments.body) from comments, aliases WHERE comments.body like ('%-' || aliases.alias || '-%') AND comments.subreddit = ? AND aliases.subreddit = ?"
    c.execute(command, (mentioner, mentionee))
    return int(c.fetchone()[0])


def cache_mutual_mentions():
    dbPath = 'analysis/comments.db'
    conn = sqlite3.connect(dbPath)
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS cached_mutual_mentions''')
    c.execute('''CREATE TABLE cached_mutual_mentions (mentionee text, mentioner text, cnt integer)''')
    for mentionee, _ in languages:
        print mentionee,
        for mentioner, _ in languages:
            result = get_mention_count(c, mentioner, mentionee)
            print result,
            c.execute('INSERT INTO cached_mutual_mentions VALUES (?, ?, ?)',
                (mentionee, mentioner, result))
        print

    conn.commit()
    conn.close()

def cache_subreddit_comment_counts():
    dbPath = 'analysis/comments.db'
    conn = sqlite3.connect(dbPath)
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS cached_subreddit_comment_counts''')
    c.execute('''CREATE TABLE cached_subreddit_comment_counts (subreddit text, cnt integer)''')
    for subreddit in subreddits:
        command = "SELECT COUNT(*) from comments WHERE comments.subreddit == ?"
        c.execute(command, (subreddit, ))
        result = int(c.fetchone()[0])
        print subreddit, result
        c.execute('INSERT INTO cached_subreddit_comment_counts VALUES (?, ?)',
                (subreddit, result))
    conn.commit()
    conn.close()

def print_table(table):
    result = ""
    for row in table:
        for cell in row:
            result += cell + ","
        result = result[:-1]
        result += "\n"
    return result

def relative_word_count(c, subreddit, word):
    command =\
        "SELECT \
        (10000.0 * COUNT(*) / cached_subreddit_comment_counts.cnt) as result\
        FROM comments, cached_subreddit_comment_counts\
        WHERE comments.subreddit = ? \
            AND cached_subreddit_comment_counts.subreddit = ? \
            AND body like ?"
    c.execute(command, (subreddit, subreddit, '%-' + word + '-%'))
    res = c.fetchone()[0]
    if not res:
        return 0
    return int(res)

def get_subreddit_comment_count(c, subreddit):
    command = "SELECT cnt from cached_subreddit_comment_counts \
        where subreddit = ?"
    c.execute(command, (subreddit,))
    return int(c.fetchone()[0])

def print_word_table(c, subreddits, words):
    sanitized_words = map(prepare_comment_body, words)
    sanitized_words = filter(lambda x: x, sanitized_words)
    print ' '.join(["subreddit"] + sanitized_words + ["sum"])
    for subreddit in subreddits:
        cnt_sum = 0
        print subreddit,
        for word in sanitized_words:
            res = relative_word_count(c, subreddit, word)
            cnt_sum += res
            print res,
        print cnt_sum

def transpose_table(mat):
    return zip(*mat)

def show_mutual_mentions(c):
    big_languages = filter(lambda (s, _): isSubredditBig(c, s), languages)
    print ' '.join(map(operator.itemgetter(0), [("subreddit", [])] + big_languages))
    for mentionee, _ in big_languages:
        print mentionee,
        for mentioner, _ in big_languages:
            command = "SELECT cnt from cached_mutual_mentions WHERE mentioner = ? and mentionee = ?"
            c.execute(command, (mentioner, mentionee))
            print int(c.fetchone()[0]),
        print
    print

def who_by_whom(c):
    print 'who is mentioned by whom how often by others relative to respective others comment count'
    command =\
        'SELECT mentioner,\
            mentionee,\
            (10000.0 * cached_mutual_mentions.cnt / cached_subreddit_comment_counts.cnt) as result\
        FROM cached_mutual_mentions, cached_subreddit_comment_counts\
        WHERE cached_mutual_mentions.mentioner == cached_subreddit_comment_counts.subreddit\
            AND cached_mutual_mentions.mentioner != cached_mutual_mentions.mentionee\
            AND cached_subreddit_comment_counts.cnt > 100\
        ORDER BY result DESC'
    c.execute(command)
    write_str_to_file("analysis/who_by_whom.csv", print_table(c.fetchall()))

def who_himself(c):
    print 'who mentions himself the most often relative to own comment count'
    command =\
        'SELECT mentioner,\
            (10000.0 * cached_mutual_mentions.cnt / cached_subreddit_comment_counts.cnt) as result\
        FROM cached_mutual_mentions, cached_subreddit_comment_counts\
        WHERE cached_mutual_mentions.mentioner == cached_subreddit_comment_counts.subreddit\
            AND cached_mutual_mentions.mentioner == cached_mutual_mentions.mentionee\
        ORDER BY result DESC, mentionee'
    c.execute(command)
    write_str_to_file("analysis/who_himself.csv", print_table(c.fetchall()))

def who_by_others(c):
    print 'who is mentioned how often by others relative to all others comment count'
    command =\
        'SELECT mentionee,\
        (10000.0 * SUM(cached_mutual_mentions.cnt) / SUM(cached_subreddit_comment_counts.cnt)) as result\
            FROM cached_mutual_mentions, cached_subreddit_comment_counts\
        WHERE cached_mutual_mentions.mentioner == cached_subreddit_comment_counts.subreddit\
            AND cached_mutual_mentions.mentioner != cached_mutual_mentions.mentionee\
        GROUP BY mentionee\
        ORDER BY result DESC'
    c.execute(command)
    write_str_to_file("analysis/who_by_others.csv", print_table(c.fetchall()))

def count_word_mentions(c):
    big_subreddits = filter(functools.partial(isSubredditBig, c), subreddits)
    print_word_table(c, big_subreddits, positive_emotions)
    print_word_table(c, big_subreddits, negative_emotions)
    print_word_table(c, big_subreddits, internet_slang_words)
    print_word_table(c, big_subreddits, single_words + opposite_words)

def isSubredditBig(c, subreddit):
    return get_subreddit_comment_count(c, subreddit) >= 1000

def cache_db_results():
    cache_subreddit_comment_counts()
    cache_mutual_mentions()

def analyse_comments():
    dbPath = 'analysis/comments.db'
    conn = sqlite3.connect(dbPath)
    c = conn.cursor()

    who_by_whom(c)
    who_himself(c)
    who_by_others(c)
    show_mutual_mentions(c)
    count_word_mentions(c)

    conn.close()

def draw_word_mentions(name, columns, colors, sorted_by_sum, filename):
    reader = csv.DictReader(open('analysis/' + name + '.csv'))

    output = []
    for row in reader:
        #print row
        d = {}
        for col in columns + ['subreddit']:
            d[col] = row[col]
        output.append(d)

    def dict_sum(d):
        c = d.copy();
        c.pop('subreddit')
        return sum(map(int,c.values()))


    if sorted_by_sum:
        output.sort(key=lambda x: dict_sum(x), reverse = True)
    else:
        output.sort(key=lambda x: x['subreddit'], reverse = True)

    subreddits = []
    dataset = []
    for row in output:
        subreddit = row['subreddit']
        subreddits.append(subreddit)
        dataset_row = {}
        counts = []
        for column in columns:
            dataset_row[column] = int(row[column])
        dataset.append(dataset_row)

    data_orders = [columns] * len(subreddits)

    names = sorted(dataset[0].keys())

    values = np.array([[data[name] for name in order] for data,order in zip(dataset, data_orders)])
    lefts = np.insert(np.cumsum(values, axis=1),0,0, axis=1)[:, :-1]
    orders = np.array(data_orders)
    bottoms = np.arange(len(data_orders))

    for name, color in zip(names, colors):
        idx = np.where(orders == name)
        value = values[idx]
        left = lefts[idx]
        plt.bar(left=left, height=0.8, width=value, bottom=bottoms,
                color=color, orientation="horizontal", label=name)
    plt.yticks(bottoms+0.4, subreddits)
    plt.legend(loc="best", bbox_to_anchor=(1.0, 1.00))

    #plt.subplots_adjust(right=0.95)
    #plt.subplots_adjust(left=1.85)
    plt.savefig('img/' + filename + '.png', bbox_inches='tight')
    #plt.show()
    plt.close()

def draw_graphs():
    # colors from http://colorschemedesigner.com/csd-3.5/
    draw_word_mentions('cursing',
        ['crap', 'fuck', 'hate', 'shit'],
        ['#66A3D2', '#7373D9', '#61D7A4', '#FFC373'],
        True,
        'cursing')

    draw_word_mentions('happy',
        ['awesome', 'cool', 'fun', 'happy', 'helpful', 'interesting'],
        ['#FF9640', '#FFBF40', '#FF4040', '#33CCCC', '#FFB273', '#FF7373'],
        True,
        'happy')

    draw_word_mentions('words',
        ['abstract', 'category', 'pure', 'theory'],
        ['#FF7373', '#FFB273', '#5CCCCC', '#67E667'],
        True,
        'abstract_concepts')

    draw_word_mentions('words',
        ['hardware'],
        ['#5577CC'],
        True,
        'hardware')

    draw_word_mentions('words',
        ['nerd'],
        ['#67E667'],
        True,
        'nerd')

    #todo: mentioned div tiobe: http://matplotlib.org/examples/pie_and_polar_charts/pie_demo_features.html

def main():
    #get_comments()
    #get_submission_ids()
    #pickle_comments()
    #comments_to_db()
    #cache_db_results()
    #analyse_comments()
    draw_graphs()

if __name__ == '__main__':
    main()

# possible todos:
#  orthography check
#  caps lock usage
#  distance between words (pyramid filter)
#  word clouds
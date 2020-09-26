
import os
import json
import sys


def quality_to_scale(quality):
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100
    if quality < 50:
        quality = 5000 // quality
    else:
        quality = 200 - quality * 2
    return quality


def scale_qtable(qtable, quality):
    scale = quality_to_scale(quality)
    res = [0] * 64
    for i in range(64):
        temp = (qtable[i] * scale + 50) // 100
        if temp <= 0:
            temp = 1
        if temp > 255:
            temp = 255
        res[i] = temp
    return res


def print_qtable(qtable, fp = sys.stdout):
    for i in range(64):
        fp.write('%4d ' % qtable[i])
        if i % 8 == 7:
            fp.write('\n')
    fp.write('\n')


def load_qtables(args):
    qtable = {}
    with open(args.qtable, 'r') as f:
        for line in f.readlines():
            tid, table = line.split(':')
            qtable[tid] = json.loads(table)
    return qtable


def join_path(*dirs):
    if len(dirs) == 0:
        return ''
    path = dirs[0]
    for d in dirs[1:]:
        path = os.path.join(path, d)
    return path


def make_filepath(fpath, dir_name=None, ext_name=None, tag=None):
    if dir_name is None:
        dir_name = os.path.dirname(fpath)
        if dir_name == '':
            dir_name = '.'
    fname = os.path.basename(fpath)
    base, ext = os.path.splitext(fname)
    if ext_name is None:
        ext_name = ext
    elif ext_name != '' and ext_name[0] != '.':
        ext_name = '.' + ext_name
    name = base
    if tag == '':
        name = name.split('-')[0]
    elif tag is not None:
        name = '%s-%s' % (name, tag)
    if ext_name != '':
        name = '%s%s' % (name, ext_name)
    return join_path(dir_name, name)

import argparse
import os
import traceback
import pandas
import subprocess
import sys

# cmd_pref = ' '.join(sys.argv[1:])

parser = argparse.ArgumentParser()
parser.add_argument('--seq', action='store_true', default=False)
parser.add_argument('opts', nargs=argparse.REMAINDER)
args = parser.parse_args()


cmd_pref = ' '.join(args.opts)

df = pandas.read_csv('../output/100doh_detectron/by_obj/very_good.csv')
index_list = df['index'].tolist()
if args.seq:
    index_list = [e[:-2] for e in index_list]

# index_list = [line.strip() for line in open('configs/to_run.config')]
for index in index_list:
    cmd = cmd_pref + ' ' + index
    print(cmd)
    try:
        subprocess.check_call(cmd.split(' '))
    except subprocess.CalledProcessError as exec:
        print(traceback.format_exc())
        print(exec)
        continue

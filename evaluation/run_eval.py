"""Generate human evaluation numbers."""
import argparse
import collections
import csv
import glob
import json
import numpy as np
import os
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Insert a description of this script.')
  parser.add_argument('domain', choices=['yelp', 'caption', 'amazon'])
  parser.add_argument('pred_dir')
  parser.add_argument('results_file')
  parser.add_argument('--num-per-class', '-n', type=int, 
                      help='Number of examples per class (default=all)')
  parser.add_argument('--out-file', '-o', help='Print to this file')
  parser.add_argument('--out-json', '-j', help='Write json data (for Vega)')
  parser.add_argument('--tsv-file', '-t', help='Write tsv file (for stats)')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def read_preds():
  preds = collections.defaultdict(list)
  for fn in glob.glob(os.path.join(OPTS.pred_dir, 'sentiment.*')):
    model = os.path.basename(fn).split('.')[3]
    y_label = os.path.basename(fn).split('.')[2]
    with open(fn) as f:
      for i, line in enumerate(f):
        if OPTS.num_per_class is not None and i == OPTS.num_per_class:
          break
        toks = line.strip().split('\t')
        orig = toks[0].strip().lower()
        mod = toks[1].strip().lower()
        if OPTS.domain == 'caption':
          target = 'humorous' if y_label == '1' else 'romantic'
        else:
          target = 'negative' if y_label == '1' else 'positive'
        preds[model].append((orig, target, mod))
  return preds

def read_human_scores():
  human_scores = {}
  with open(OPTS.results_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
      origs = row['Input.origs'].split('\t')
      mods = row['Input.mods'].split('\t')
      target = row['Input.target']
      responses = row['Answer.responses'].split('\t')
      for (orig, mod, r) in zip(origs, mods, responses):
        key = '\t'.join((orig, target, mod))
        cur_scores = collections.OrderedDict()
        for x in r.split(' '):
          k, v = x.split('=')
          cur_scores[k] = int(v)
        success = int(all(v > 3 for v in cur_scores.values()))
        cur_scores['success'] = success
        human_scores[key] = cur_scores
  return human_scores

def main():
  preds = read_preds()
  human_scores = read_human_scores()
  eval_obj = {}
  out_lines = []
  out_json = []
  tsv_lines = []
  for model in preds:
    eval_obj[model] = collections.defaultdict(list)
    for (orig, target, mod) in preds[model]:
      key = '\t'.join((orig, target, mod))
      cur_scores = human_scores[key]
      json_ex = dict(cur_scores)  # clone
      json_ex['model'] = model
      out_json.append(json_ex)
      for cat, val in cur_scores.iteritems():
        eval_obj[model][cat].append(val)
      score_str = ' '.join('%s=%d' %  (k, v) for k, v in cur_scores.iteritems())
      out_lines.append('\t'.join((model, orig, target, mod, score_str)))
    print model
    for cat in eval_obj[model]:
      mean = np.mean(eval_obj[model][cat])
      std_err = np.std(eval_obj[model][cat]) / np.sqrt(len(eval_obj[model][cat]) - 1)
      print '  %s: %.3f +/- %.3f' % (cat, mean, std_err)
      tsv_lines.append('\t'.join((OPTS.domain, cat, model, '%.3f' % mean)))
    print
  if OPTS.out_file:
    with open(OPTS.out_file, 'w') as f:
      for line in out_lines:
        print >> f, line
  if OPTS.out_json:
    with open(OPTS.out_json, 'w') as f:
      json.dump(out_json, f)
  if OPTS.tsv_file:
    with open(OPTS.tsv_file, 'w') as f:
      for line in tsv_lines:
        print >> f, line


if __name__ == '__main__':
  OPTS = parse_args()
  main()


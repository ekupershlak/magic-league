#!/usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function

import re
import sys

import html
import requests

RO = 1024
TOP = RO // 2

try:
  rss = open('rss').read()
except IOError:
  rss = html.unescape(
      html.unescape(requests.get('https://mtgbracket.tumblr.com/rss').text))
  open('rss', 'w').write(rss)

batches = re.findall(f'Round of {RO} - Batch (\\d+) results.*?'
                     f'Visual results are(.*?)Full results', rss, re.DOTALL)
batches.sort(key=lambda b: int(b[0]))
batch_number, post = batches[-1]
batch_number = int(batch_number)
post = re.sub('’', "'", post)
post = re.sub(r'\s+', ' ', post)

cardname_pattern = r"\w[\w '\-,/\.]*?"
result_pattern = r'\b({0}) defeats ({0}) with \d{{2}}\.\d{{2}}%'.format(
    cardname_pattern)
x = [a for a, b in re.findall(result_pattern, post, re.DOTALL | re.UNICODE)]
if len(x) < 16:
  print('WARNING: Found {} winners.'.format(len(x)), file=sys.stderr)

filename = f'top-{TOP:d}-batch-{batch_number:02d}'
print(f'Writing to {filename}')
out = open(filename, 'w')
for card in x:
  print(card, file=out)
  # print(card)

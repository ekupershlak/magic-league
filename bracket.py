# -*- encoding: utf-8 -*-

from __future__ import print_function

import re
import sys

import html
import requests

try:
  rss = open('rssl').read()
except IOError:
  rss = requests.get('https://mtgbracket.tumblr.com/rss').text
  open('rss', 'w').write(rss)

rsses = re.findall(r'Round of 2048 - Batch (\d+) results.*?'
                   r'Visual results are(.*?)</description>', rss, re.DOTALL)

batch_number, post = rsses[0]
post = html.unescape(html.unescape(post))
post = re.sub('’', "'", post)
post = re.sub(r'\s+', ' ', post)

cardname = r"\w[\w '\-,/\.]*?"
x = [
    a
    for a, b in re.findall(r'\b({0}) defeats ({0}) with \d{{2}}\.\d{{2}}%'.
                           format(cardname), post, re.DOTALL | re.UNICODE)
]
if len(x) < 32:
  print('WARNING: Found {} winners.'.format(len(x)), file=sys.stderr)

out = open('top-1024-batch-{:02d}'.format(int(batch_number)), 'w')
for card in x:
  print(card, file=out)
  # print(card)

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input

import numpy as np

import os
import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.brown.py import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx
#!/usr/bin/python
from lib.sentence_parser import get_nouns
from config import YAKSHA as yksh

print "Hey! This is "+yksh+"."
while True:
    sentence = raw_input("You: ")
    print get_nouns(sentence)
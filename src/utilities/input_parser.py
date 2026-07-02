import argparse
# set up parser
# make parser global to read parameters
parser = argparse.ArgumentParser()
parser.add_argument('-ct1', nargs=1, default=None)
parser.add_argument('-co', nargs=1, default=None)
parser.add_argument('-ct2', nargs='?', default=None)
parser.add_argument('-yml_inp', nargs=1, default=None)
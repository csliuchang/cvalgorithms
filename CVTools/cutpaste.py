import argparse


def cutpaste():
    pass





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_dir", default='', type=str, help='postive example for image cutpaste')
    parser.add_argument("--neg_dir", default='', type=str, help='negtive example for image cutpaste')
    parser.add_argument("--")
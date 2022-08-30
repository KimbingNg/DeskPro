import argparse
parser = argparse.ArgumentParser(description='MGN')

parser.add_argument("--config", type=str, default="", help='config path')
parser.add_argument('--resume', type=str, default='', help='folder name to load')
parser.add_argument("--test", action='store_true', default=False, help='test')
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

parser.add_argument("--local_rank", default=-1)

args = parser.parse_args()

for arg in vars(args):
     if vars(args)[arg] == 'True':
          vars(args)[arg] = True
     elif vars(args)[arg] == 'False':
          vars(args)[arg] = False

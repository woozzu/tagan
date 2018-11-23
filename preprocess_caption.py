import os
import argparse
import fasttext

from data import ConvertCapVec


parser = argparse.ArgumentParser()
parser.add_argument('--caption_root', type=str, required=True,
                    help='root directory that contains captions')
parser.add_argument('--fasttext_model', type=str, required=True,
                    help='pretrained fastText model (binary file)')
parser.add_argument('--max_nwords', type=int, default=50,
                    help='maximum number of words (default: 50)')
args = parser.parse_args()


if __name__ == '__main__':
    caption_root = args.caption_root.split('/')[-1]
    if (caption_root + '_vec') not in os.listdir(args.caption_root.replace(caption_root, '')):
        os.makedirs(args.caption_root + '_vec')
        print('Loading a pretrained fastText model...')
        word_embedding = fasttext.load_model(args.fasttext_model)
        print('Making vectorized caption data files...')
        ConvertCapVec().convert_and_save(args.caption_root, word_embedding, args.max_nwords)

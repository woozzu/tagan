import os
import argparse
import fastText
from PIL import Image
import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import Generator
from data import split_sentence_into_words


parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--text_file', type=str, required=True,
                    help='text file that contains descriptions')
parser.add_argument('--fasttext_model', type=str, required=True,
                    help='pretrained fastText model (binary file)')
parser.add_argument('--generator_model', type=str, required=True,
                    help='pretrained generator model')
parser.add_argument('--output_root', type=str, required=True,
                    help='root directory of output')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
args = parser.parse_args()


if __name__ == '__main__':
    if not args.no_cuda and not torch.cuda.is_available():
        print('Warning: cuda is not available on this machine.')
        args.no_cuda = True
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    print('Loading a pretrained fastText model...')
    word_embedding = fastText.load_model(args.fasttext_model)

    print('Loading a pretrained model...')
    G = Generator().to(device)
    G.load_state_dict(torch.load(args.generator_model))
    G.eval()

    transform = transforms.Compose([
        transforms.Resize(136),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    print('Loading test data...')
    filenames = os.listdir(args.img_root)
    img = []
    for fn in filenames:
        im = Image.open(os.path.join(args.img_root, fn))
        im = transform(im)
        img.append(im)
    img = torch.stack(img)
    save_image(img, os.path.join(args.output_root, 'original.jpg'), pad_value=1)
    img = img.mul(2).sub(1).to(device)

    html = '<html><body><h1>Manipulated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Description</b></td><td><b>Image</b></td></tr>'
    html += '\n<tr><td>ORIGINAL</td><td><img src="{}"></td></tr>'.format('original.jpg')
    with open(args.text_file, 'r') as f:
        texts = f.readlines()

    for i, text in enumerate(texts):
        text = text.replace('\n', '')
        words = split_sentence_into_words(text)
        txt = torch.tensor([word_embedding.get_word_vector(w) for w in words], device=device)
        txt = txt.unsqueeze(1)
        txt = txt.repeat(1, img.size(0), 1)
        len_txt = torch.tensor([len(words)], dtype=torch.long, device=device)
        len_txt = len_txt.repeat(img.size(0))

        output, _ = G(img, (txt, len_txt))

        out_filename = 'output_%03d.jpg' % i
        save_image(output.mul(0.5).add(0.5), os.path.join(args.output_root, out_filename), pad_value=1)
        html += '\n<tr><td>{}</td><td><img src="{}"></td></tr>'.format(text, out_filename)

    with open(os.path.join(args.output_root, 'index.html'), 'w') as f:
        f.write(html)
    print('Done.')

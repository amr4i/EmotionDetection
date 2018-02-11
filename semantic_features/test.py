# System libs
import os
import pickle
import argparse
# Numerical libs
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from scipy.io import loadmat
from scipy.misc import imread, imresize, imsave
from scipy.ndimage import zoom
from sklearn.preprocessing import normalize

# Our libs
from models import ModelBuilder
from utils import colorEncode


# forward func for testing
def forward_test_multiscale(nets, img, args):
    (net_encoder, net_decoder) = nets

    pred = torch.zeros(1, args.num_class, img.size(2), img.size(3))
    pred = Variable(pred, volatile=True).cuda()

    for scale in args.scales:
        img_scale = zoom(img.numpy(),
                         (1., 1., scale, scale),
                         order=1,
                         prefilter=False,
                         mode='nearest')

        # feed input data
        input_img = Variable(torch.from_numpy(img_scale),
                             volatile=True).cuda()

        # forward
        pred_scale = net_decoder(net_encoder(input_img),
                                 segSize=(img.size(2), img.size(3)))

        # average the probability
        pred = pred + pred_scale / len(args.scales)

    return pred

def get_feature_vector(img, pred, args):
    img = img[0]
    pred = pred.data.cpu()[0]
    for t, m, s in zip(img,
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

    # prediction
    pred_ = np.argmax(pred.numpy(), axis=0) + 1
    vec_ = np.apply_along_axis(np.bincount, 1, pred_, minlength=150)
    vec = normalize(np.sum(vec_, axis=0).reshape(-1,1), axis=0).ravel()
    
    return vec


def test(nets, args):
    # switch to eval mode
    for net in nets:
        net.eval()

    outDict = {}

    with open(args.test_img_file , 'r') as trainFile:

        files = trainFile.readlines()
        
        for file in files:
            filename, file_label = file.split(",")
            filepath = os.path.join(args.test_img_directory, filename)

            # loading image, resize, convert to tensor
            img = imread(filepath, mode='RGB')
            h, w = img.shape[0], img.shape[1]
            s = 1. * args.imgSize / min(h, w)
            img = imresize(img, s)
            img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            img = img_transform(img)
            img = img.view(1, img.size(0), img.size(1), img.size(2))
            
            # forward pass
            pred = forward_test_multiscale(nets, img, args)
            
            # visualization
            # visualize_test_result(img, pred, args)

            # feature vector 
            vec = get_feature_vector(img, pred, args)
            outDict[filename]=vec

    with open('../semantic_features.pickle', 'wb') as handle:
        pickle.dump(outDict, handle, protocol=pickle.HIGHEST_PROTOCOL)




def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=args.arch_encoder,
                                        fc_dim=args.fc_dim,
                                        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(arch=args.arch_decoder,
                                        fc_dim=args.fc_dim,
                                        segSize=args.segSize,
                                        weights=args.weights_decoder,
                                        use_softmax=True)

    nets = (net_encoder, net_decoder)
    for net in nets:
        net.cuda()

    # single pass
    test(nets, args)

    print('Done! Output is saved in {}'.format(args.result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    # parser.add_argument('--id', required=True,
                        # help="a name for identifying the model to load")
    parser.add_argument('--suffix', default='_best.pth',
                        help="which snapshot to load")
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='psp_bilinear',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--test_img_file', required=True)
    parser.add_argument('--test_img_directory', required=True)
    parser.add_argument('--weights_encoder', required=True)
    parser.add_argument('--weights_decoder', required=True)

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize')
    parser.add_argument('--imgSize', default=384, type=int,
                        help='resize input image')
    parser.add_argument('--segSize', default=-1, type=int,
                        help='output image size, -1 = keep original')

    args = parser.parse_args()
    print(args)

    # scales for evaluation
    args.scales = (0.5, 0.75, 1, 1.25, 1.5)

    # absolute paths of model weights
    # args.weights_encoder = os.path.join(args.ckpt, args.id, 'encoder' + args.suffix)
    # args.weights_decoder = os.path.join(args.ckpt, args.id, 'decoder' + args.suffix)
    
    # args.weights_encoder = '/ckpt/encoder_best.pth'
    # args.weights_decoder = './ckpt/decoder_best.pth'

    main(args)

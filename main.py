from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import json
import pandas as pd
from scipy import random
from sklearn import preprocessing
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler

from retrieval_model import RFOP


print('Training')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def read_data(FLAGS):
    train_file_face = '../%s/%s/%s_faces_train.csv'%(FLAGS.ver, FLAGS.train_lang, FLAGS.train_lang)
    train_file_voice = '../%s/%s/%s_voices_train.csv'%(FLAGS.ver, FLAGS.train_lang, FLAGS.train_lang)
    
    print('Reading Train Faces')
    img_train = pd.read_csv(train_file_face, header=None)
    img_train = np.asarray(img_train)
    img_train = img_train[:, :-1]
    
    print('Reading Voices')
    voice_train = pd.read_csv(train_file_voice, header=None)
    voice_train = np.asarray(voice_train)

    train_label = voice_train[:, -1:]
    train_label = np.asarray(train_label)

    voice_train = voice_train[:, :-1]

   
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)
    print("Train file length", len(img_train))
        
    print('Shuffling\n')
    combined = list(zip(img_train, voice_train, train_label))
    img_train = []
    voice_train = []
    train_label = []
    random.shuffle(combined)
    img_train[:], voice_train, train_label[:] = zip(*combined)
    combined = [] 
    img_train = np.asarray(img_train).astype(np.float)
    voice_train = np.asarray(voice_train).astype(np.float)
    train_label = np.asarray(train_label)
    
    
    return img_train, voice_train, train_label


def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main(face_train, voice_train, train_label):

    if FLAGS.ver == 'v1':
        n_class = 64
    elif FLAGS.ver == 'v2':
        n_class = 78 
    else:
        n_class = 50
    
    model = RFOP(FLAGS, face_train.shape[1], voice_train.shape[1], n_class)
    model.apply(init_weights)

    ####################
    # load round1 weights (used in 2nd pass)
    '''
    weights = torch.load('./first_pass/checkpoint_best.pth.tar')
    sd = weights['state_dict']
    keys = list(sd.keys())
    for key in keys:
        if 'logits_layer' in key:
            del sd[key]
    msg = model.load_state_dict(sd, strict=False)
    print(msg)
    '''
    ###########################3

    ce_loss = nn.CrossEntropyLoss().cuda()
    opl_loss = OrthogonalProjectionLoss().cuda()
    
    if FLAGS.cuda:
        model.cuda()
        ce_loss.cuda()    
        opl_loss.cuda()
        cudnn.benchmark = True

    optimizer = optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=0.2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=FLAGS.epochs, eta_min=1e-8)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    epoch=1
    num_of_batches = (len(train_label) // FLAGS.batch_size)


    save_dir = '%s_%s_log'%(FLAGS.ver, FLAGS.train_lang)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    eer_list = []
    #prev_weight = None

    while (epoch < FLAGS.epochs):
        loss_per_epoch = 0
        loss_plot = []
        print('Epoch %03d'%(epoch))
        for idx in tqdm(range(num_of_batches)):
            
            face_feats, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, face_train)
            voice_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)

            '''
            if epoch < 5:
                if prev_weight is not None:
                    sd = model.state_dict()
                    for key in sd.keys():
                        sd[key] = (sd[key] + prev_weight[key])/2.0

                    model.load_state_dict(sd)
            '''

            loss_tmp, loss_opl, loss_soft, s_fac, d_fac = train(face_feats, voice_feats, batch_labels, 
                                                                model, optimizer, ce_loss, opl_loss)
            
            #prev_weight = model.state_dict()

            loss_per_epoch += loss_tmp
        
        scheduler.step()

        loss_per_epoch/=num_of_batches
        
        loss_plot.append(loss_per_epoch)
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict()}, save_dir, 'checkpoint_%04d_%0.3f.pth.tar'%(epoch, loss_per_epoch))
        
        print('==> Epoch: %d/%d Loss: %0.2f LR:%0.6f, '%(epoch, FLAGS.epochs, loss_per_epoch, scheduler.get_lr()[-1]))
        
        loss_per_epoch = 0
        epoch += 1
            
    return
    
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self):
        super(OrthogonalProjectionLoss, self).__init__()
        self.device = (torch.device('cuda') if FLAGS.cuda else torch.device('cpu'))

    def forward(self, features, labels):
        
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]

        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1-pos_pairs_mean) + 0.9*neg_pairs_mean

        return loss, pos_pairs_mean, neg_pairs_mean


def train(face_feats, voice_feats, labels, model, optimizer, ce_loss, opl_loss):
    
    average_loss = RunningAverage()
    soft_losses = RunningAverage()
    opl_losses = RunningAverage()

    s_fac = 0
    d_fac = 0

    model.train()
    face_feats = torch.from_numpy(face_feats).float()
    voice_feats = torch.from_numpy(voice_feats).float()

    labels = torch.from_numpy(labels).long()

    if FLAGS.cuda:
        face_feats, voice_feats, labels = face_feats.cuda(), voice_feats.cuda(), labels.cuda()

    face_feats, voice_feats, labels = Variable(face_feats), Variable(voice_feats), Variable(labels)

    comb, face_embeds, voice_embeds, face_f, voice_f = model(face_feats, voice_feats)
    
    loss_opl, s_fac, d_fac = opl_loss(comb[0], labels)
    
    loss_soft = ce_loss(comb[1], labels)

    loss_mse = (face_f[0] - voice_f[0])**2
    loss_mse = loss_mse.mean()

    loss = loss_soft*0.2 + loss_opl*0.78 + loss_mse*0.02

    optimizer.zero_grad()
    
    loss.backward()
    average_loss.update(loss.item())
    opl_losses.update(loss_opl.item())
    soft_losses.update(loss_soft.item())
    
    optimizer.step()

    return average_loss.avg(), opl_losses.avg(), soft_losses.avg(), s_fac, d_fac

class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0. 

    def update(self, val):
        self.value_sum += val 
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average
 
def save_checkpoint(state, directory, filename):
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random Seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='CUDA Training')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--ver', default='v3', type=str, help='Dataset version')
    parser.add_argument('--train_lang', default='English', type=str, help='Training language')
    parser.add_argument('--unheard_lang', default='German', type=str, help='Test language')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Max number of epochs to train, number')
    parser.add_argument('--dim_embed', type=int, default=256, help='Embedding Size')

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
        
    face_train, voice_train, train_label = read_data(FLAGS)

    main(face_train, voice_train, train_label)
    

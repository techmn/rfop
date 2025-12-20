import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim).cuda()
        #self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.normalize(x) 
        return x


def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out), 
                        nn.BatchNorm1d(f_out),
                        nn.ReLU(),
                        #nn.Dropout(p=0.1), # uncomment for German train
                        nn.Linear(f_out, f_out))


class FusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, 3, padding=1)

    def forward(self, face, voice):
        
        ff = nn.functional.gelu(face)
        vf = nn.functional.gelu(voice)

        w = ff + vf
        
        x = torch.cat([(face*w).unsqueeze(1), (voice*w).unsqueeze(1)], dim=1)
        x = self.conv(x).squeeze(1)
        x = nn.functional.gelu(x)

        return x, ff, vf


##################################################################

class RFOP(nn.Module):
    def __init__(self, args, face_feat_dim, voice_feat_dim, n_class):
        super(RFOP, self).__init__()
        self.embed_dim = args.dim_embed

        self.voice_branch = EmbedBranch(voice_feat_dim, args.dim_embed)
        self.face_branch = EmbedBranch(face_feat_dim, args.dim_embed)
        
        self.fusion_layer = FusionBlock(dim=self.embed_dim)
        self.res_mix = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.logits_layer = nn.Linear(self.embed_dim, n_class)

        if args.cuda:
            self.cuda()

    def forward(self, faces, voices):

        voices_feats = self.voice_branch(voices)
        faces_feats = self.face_branch(faces)

        fused_feats, faces_feats_e, voices_feats_e = self.fusion_layer(faces_feats, voices_feats) # fusion of features

        fused_feats = torch.nn.functional.relu(self.res_mix(fused_feats))

        logits = self.logits_layer(fused_feats)

        return [fused_feats, logits], faces_feats_e, voices_feats_e, [faces_feats], [voices_feats]


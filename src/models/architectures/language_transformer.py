import os
from transformers import BertTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import pdb
# from src.train.train_cvae import parser
from src.parser.generate import parser
parameters = parser()
device = "cpu"
language_option = '1'
# device = parameters["device"]
# language_option = parameters["language"]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1 / (lengths[..., None] - 1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)



flag3d_coarse_action_description = {}
flag3d_text = []
i = 0
for text_name in os.listdir('data/flag3d_txt/'):
    if text_name.endswith('00' + language_option + '.txt'):
        f = open('data/flag3d_txt/' + text_name)
        line = f.readline()
        flag3d_coarse_action_description[i] = line
        flag3d_text.append(line)
        i += 1
# print(flag3d_text)
# print(flag3d_coarse_action_description)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# if language_option == '1':
#     max_length = 512
bert_input = tokenizer(flag3d_text, padding='max_length',
                       max_length=512,
                       truncation=True,
                       return_tensors="pt")
actions_text_features = bert_input['input_ids'].to(device)
print(actions_text_features.size())
# model, preprocess = clip.load("ViT-B/32", device=device)
# with torch.no_grad():
#     actions_text_features = model.encode_text(actions_text_features)
#     print("text_features", actions_text_features.shape)


# model, preprocess = clip.load("ViT-B/32", device=device)
# text = clip.tokenize(flag3d_text).to(device)
# print('text', text.shape)
# with torch.no_grad():
#     actions_text_features = model.encode_text(text)
#     print("text_features", actions_text_features.shape)
# humanact12_coarse_action_description = {
#     0: "A man is warming up",
#     1: "A man is walking",
#     2: "A man is running",
#     3: "A man is jumping",
#     4: "A man is drinking",
#     5: "A man is lifting dumbbell",
#     6: "A man is siting",
#     7: "A man is eating",
#     8: "A man is turning steering wheel",
#     9: "A man is putting a phone call",
#     10: "A man is boxing",
#     11: "A man is throwing",
#
# }
# device ="cpu"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# text = clip.tokenize(
#     ["A man is warming up", "A man is walking", "A man is running", "A man is jumping", "A man is drinking",
#      "A man is lifting dumbbell", "A man is siting", "A man is eating", "A man is turning steering wheel",
#      "A man is putting a phone call", "A man is boxing", "A man is throwing"]).to(device)
# print('text', text.shape)

class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        if self.ablation == "average_encoder":
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.sigma_layer = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
            self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

        #     self.model, preprocess = clip.load("ViT-B/16")
        self.mu_layer = nn.Linear(512, self.latent_dim)
        self.sigma_layer = nn.Linear(512, self.latent_dim)

    def forward(self, batch):
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        #    pdb.set_trace()

        #    batch_label = y.tolist()
        #    batch_language = []
        #    for i in range(y.shape[0]):
        #        label = batch_label[i]
        #        language = humanact12_coarse_action_description[label]
        #        batch_language.append(language)
        #    pdb.set_trace()
        #    model, preprocess = clip.load("ViT-B/16")
        #    text = model.tokenize(['hello'])
        #    y_text = clip.tokenize(batch_language).to(y.device)
        #    with torch.no_grad():
        #        text_features = self.model.encode_text(y_text)
        text_features = actions_text_features[y]
        text_features = text_features.to(torch.float)
        token_mu = self.mu_layer(text_features)
        token_sigma = self.sigma_layer(text_features)
        # embedding of the skeleton
        x = self.skelEmbedding(x)

        # only for ablation / not used in the final model
        if self.ablation == "average_encoder":
            # add positional encoding
            x = self.sequence_pos_encoder(x)

            # transformer layers
            final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
            # get the average of the output
            z = final.mean(axis=0)

            # extract mu and logvar
            mu = self.mu_layer(z)
            logvar = self.sigma_layer(z)
        else:
            # adding the mu and sigma queries
            #   xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)
            xseq = torch.cat((token_mu[None], token_sigma[None], x), axis=0)
            # add positional encoding
            xseq = self.sequence_pos_encoder(xseq)

            # create a bigger mask, to allow attend to mu and sigma
            muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
            maskseq = torch.cat((muandsigmaMask, mask), axis=1)

            final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
            mu = final[0]
            logvar = final[1]

        return {"mu": mu, "logvar": logvar}


class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        else:
            self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)

        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        #   self.model, preprocess = clip.load("ViT-B/16")
        self.actionlayer = nn.Linear(512, self.latent_dim)

    def forward(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        # batch_label = y.tolist()
        # batch_language = []
        # for i in range(y.shape[0]):
        #     label = batch_label[i]
        #     language = humanact12_coarse_action_description[label]
        #     batch_language.append(language)

        # y_text = clip.tokenize(batch_language).to(y.device)
        # with torch.no_grad():
        #     text_features = self.model.encode_text(y_text)
        text_features = actions_text_features[y]
        text_features = text_features.to(torch.float)
        token_a = self.actionlayer(text_features)
        #    pdb.set_trace()
        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            yoh = F.one_hot(y, self.num_classes)
            z = torch.cat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]  # sequence of size 1
        else:
            # only for ablation / not used in the final model
            if self.ablation == "concat_bias":
                # sequence of size 2
                z = torch.stack((z, self.actionBiases[y]), axis=0)
            # z = torch.stack((z, token_a), axis=0)
            else:
                # shift the latent noise vector to be the action noise
                z = z + token_a
                #    z = z + self.actionBiases[y]
                z = z[None]  # sequence of size 1

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries)

        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)

        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)

        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)

        batch["output"] = output
        return batch

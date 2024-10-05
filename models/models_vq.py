import torch
import torch.nn.functional as F
import importlib
from einops import rearrange
from torch.nn import Embedding
from models.discriminator import NLayerDiscriminator, weights_init
from models.lpips import LPIPS
#from models.encoder_decoder import Encoder, Decoder, Decoder_Cross
from models.vit_encoder import Encoder
from models.decoder import ViTDecoder, CNNDecoder
import torch.nn as nn
import tome
from models.source_recovery import source_recovery as Recovery
from models.cls_head import MlpHead



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def apply_merge(model, args):
    tome.patch.timm(model)
    model.r = args.merge_ratio
    return model



class VQModel(torch.nn.Module):
    def __init__(self,
                 args,
                 enconfig,
                 ddconfig,
                 k2lconfig,
                 embed_dim=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.args = args
        
        self.stage = args.stage
        self.merge_ratio = args.merge_ratio

        self.o_encoder = Encoder(**enconfig)
        self.encoder = apply_merge(self.o_encoder, args)
        #print(self.encoder._tome_info)
        
        if ddconfig.pop('type', 'ViT').lower() == 'vit':
            self.decoder = ViTDecoder(**ddconfig)
        else:
            self.decoder = CNNDecoder(**ddconfig)
        print(self.decoder)
        self.discriminator = NLayerDiscriminator(input_nc=3,
                                                n_layers=2,
                                                use_actnorm=False,
                                                ndf=64
                                                ).apply(weights_init)
        
        embed_dim = args.embed_dim  # update embed_dim for VQ codebook
        if args.merge_ratio != 0:
            self.recovery = Recovery(**k2lconfig)

        #self.recovery = Recovery(**k2lconfig)
        self.projection = nn.Linear(embed_dim,embed_dim)

        self.perceptual_loss = LPIPS().eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
        self.perceptual_weight = args.rate_p        
        self.quantize_type = args.quantizer_type
        self.cls_head = MlpHead(dim=enconfig.embed_dim, dropout=enconfig.head_dropout)
        
        print("****Using Quantizer: %s"%(args.quantizer_type))
        self.criterion = torch.nn.CrossEntropyLoss()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        codebook_dim = embed_dim
        if args.tuning_codebook == 2: ## Random
            print("****Using Tuned Random Codebook****")
            print("Word Number:%d" %(args.n_vision_words))
            print("Feature Dim:%d" %(embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, embed_dim)
            self.tok_embeddings.weight.data.uniform_(-1.0 / args.n_vision_words, 1.0 / args.n_vision_words)
            self.tok_embeddings.weight.requires_grad = True
        
        elif args.tuning_codebook == -2: ##Random Fix
            print("****Using Fix Random Codebook****")
            print("Word Number:%d" %(args.n_vision_words))
            print("Feature Dim:%d" %(embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, embed_dim)
            self.tok_embeddings.weight.data.uniform_(-1.0 / args.n_vision_words, 1.0 / args.n_vision_words)
            self.tok_embeddings.weight.requires_grad = False

        elif args.tuning_codebook == 0:
            print("****Using Fix Initialized Codebook****")
            checkpoint = torch.load(args.local_embedding_path, map_location="cpu")
            args.n_vision_words = checkpoint.shape[0]
            codebook_dim = checkpoint.shape[1]
            print("Word Number:%d" %(args.n_vision_words))
            print("Feature Dim:%d" %(embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, checkpoint.shape[1])
            self.tok_embeddings.weight.data = checkpoint
            self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.float()
            self.tok_embeddings.weight.requires_grad = False

        elif args.tuning_codebook == 1:
            print("****Tuning Initialized Codebook****")
            checkpoint = torch.load(args.local_embedding_path, map_location="cpu")
            args.n_vision_words = checkpoint.shape[0]
            codebook_dim = checkpoint.shape[1]
            print("Word Number:%d" %(args.n_vision_words))
            print("Feature Dim:%d" %(embed_dim))
            self.tok_embeddings = Embedding(args.n_vision_words, checkpoint.shape[1])
            self.tok_embeddings.weight.data = checkpoint
            self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.float()
            self.tok_embeddings.weight.requires_grad = True

        self.e_dim = embed_dim
        # print('self.e_dim', self.e_dim)
        self.remap = remap
        self.sane_index_shape = sane_index_shape
        #self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        #self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)


        if self.quantize_type == "ema":
            self.decay = 0.99
            self.eps = 1e-5
            self.cluster_size = torch.nn.Parameter(torch.zeros(args.n_vision_words), requires_grad = False)
            self.embed_avg = torch.nn.Parameter(self.tok_embeddings.weight.clone(), requires_grad = False)
            self.update = True
            self.tok_embeddings.weight.requires_grad = False
            self.num_tokens = args.n_vision_words

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, discriminator_weight, last_layer=None):

        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * discriminator_weight
        return d_weight

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.tok_embeddings.weight.data.copy_(embed_normalized) 


    def quantize(self, z, temp=None, rescale_logits=False, return_logits=False):

        # reshape z -> (batch, height, width, channel) and flatten
        #z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
        if self.args.use_cblinear != 0:
            tok_embeddings_weight = self.codebook_projection(self.tok_embeddings.weight)
        else:
            tok_embeddings_weight = self.tok_embeddings.weight
        tok_embeddings_weight = torch.nn.functional.normalize(tok_embeddings_weight, dim=-1)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(tok_embeddings_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(tok_embeddings_weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        if self.quantize_type == "ema":
            z_q = self.tok_embeddings(min_encoding_indices).view(z.shape)
            encodings = F.one_hot(min_encoding_indices, self.num_tokens).type(z.dtype)     
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))
            min_encodings = None
            #EMA cluster size
            encodings_sum = encodings.sum(0)            
            self.cluster_size_ema_update(encodings_sum)
            #EMA embedding average
            embed_sum = encodings.transpose(0,1) @ z_flattened            
            self.embed_avg_ema_update(embed_sum)
            #normalize embed_avg and update weight
            self.weight_update(self.num_tokens)
            loss = F.mse_loss(z_q.detach(), z) 
        else:
            min_encodings = None
            perplexity = None
            z_q = F.embedding(min_encoding_indices, tok_embeddings_weight).view(z.shape)
            loss = torch.mean((z_q.detach()-z)**2) + 0.33 * torch.mean((z_q - z.detach()) ** 2)
            #loss = torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
    
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (d, min_encodings, min_encoding_indices)
    
    def forward(self, input, global_input, label_cls, data_iter_step, step=0, is_val=False):
        
        #encoder_feature = self.quant_conv(self.encoder(input))
        quant, qloss, [_, _, tk_labels], enc_h = self.encode(input)

        #quant doesnt have cls token
        #source = self.encoder._tome_info["source"]
        #source = self.encoder_tome_info["source"]
        #source = source[:,1:,1:]

        ###Training GPT
        if self.stage == 2: 
            return quant, tk_labels.view(input.shape[0], -1)
        

        if self.merge_ratio != 0 :
            source = self.encoder._tome_info["source"]
            source = source[:,1:,1:]
            quant = self.recovery(quant,source)

        #quant = self.recovery(quant)
        
        dec = self.decode(quant)
        
        ###Loss
        rec_loss = torch.mean(torch.abs(input.contiguous() - dec.contiguous()))
        
        p_loss = torch.mean(self.perceptual_loss(input.contiguous(), dec.contiguous()))
        
        cls_loss = 0
        
        if self.cls_head is not None:
            enc_h = enc_h[:, 0, :]  # cls_token
            pred = self.cls_head(enc_h)
            pred = F.softmax(pred, dim=1)
            cls_loss = F.cross_entropy(pred, label_cls, reduction='mean')
        
        if step == 0: #Upadte Generator
            logits_fake = self.discriminator(dec)
            g_loss = -torch.mean(logits_fake)

            if is_val:
                loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + 0 * g_loss + self.args.rate_cls * cls_loss
                return loss, rec_loss, qloss, p_loss, g_loss, cls_loss, tk_labels.view(input.shape[0], -1), dec
            
            d_weight = self.calculate_adaptive_weight(rec_loss + self.perceptual_weight * p_loss, g_loss, self.args.rate_d, last_layer=self.decoder.conv_out.weight)
            
            if data_iter_step > self.args.disc_start:
                loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + d_weight * g_loss + self.args.rate_cls * cls_loss
            else:
                loss = rec_loss + self.args.rate_q * qloss + self.perceptual_weight * p_loss + 0 * g_loss + self.args.rate_cls * cls_loss

            return loss, rec_loss, qloss, p_loss, g_loss, cls_loss, tk_labels, dec
        else: #Upadte Discriminator
            logits_real =  self.discriminator(input.contiguous().detach().clone())
            logits_fake = self.discriminator(dec.detach().clone())
            d_loss = self.hinge_d_loss(logits_real, logits_fake)
            loss = d_loss + 0 * (rec_loss + qloss + p_loss)

            return loss, rec_loss, qloss, p_loss, d_loss, cls_loss, tk_labels, dec


    def encode(self, input):
        #h = self.quant_conv(self.encoder(input))
        h = self.encoder(input)
        # if self.e_dim == 768 and self.args.tuning_codebook != -1:
        h_none_cls = h[:,1:,:]
        h_none_cls = self.projection(h_none_cls)
        if self.args.tuning_codebook != -1:
            h_q = h_none_cls / h_none_cls.norm(dim=1, keepdim=True)
        else:
            h_q = h_none_cls.clone()
        quant, emb_loss, info = self.quantize(h_q)
        return quant, emb_loss, info, h

    def decode(self, quant, global_c_features=None):
        #quant = self.post_quant_conv(quant)
        #quant = quant[:, 1:, :]
        if self.merge_ratio == 0:
            quant = quant[:, 1:, :]
        dec = self.decoder(quant)

        return dec
    def token_recovery(self,quant):
        rec_tokens = self.recovery(quant)
        return rec_tokens

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def decode_code(self, code_b):
        quant_b = self.quantize.embedding(code_b)
        dec = self.decode(quant_b)
        return dec

    def classification(self,input):
        z = self.encoder(input)
        y = self.cls_head(z)
        return y

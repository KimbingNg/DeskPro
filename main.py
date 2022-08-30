import re
import copy

import torch
import numpy as np
import pytorch_lightning as pl

import data
from loss import make_loss
from model.deskpro import deskpro
from optim import make_optimizer, make_scheduler
from utils.functions import evaluation


def cuda(t):
    if isinstance(t, list) or isinstance(t, tuple):
        return [item.cuda() for item in t]
    return t.cuda()

class Model(pl.LightningModule):
    def __init__(self, cfg, loader):
        super(Model, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.galleryset
        self.queryset = loader.queryset
        self.data_manager = loader


        self.model = deskpro.DeskPro(cfg)
        sd = {}
        for k, v in torch.load( cfg.pre_train, map_location='cpu')['state_dict'].items():
            k = k.replace('module.', '')
            if k.startswith('face_net'):
                k = k.replace('face_net.', '')
                sd[k] = v
        self.model.face_net.load_state_dict(sd, strict=False)
        self.loss = make_loss(cfg)
        self.save_hyperparameters(cfg)

    def forward(self, *args, **kwargs):
        foo = self.model(*args, **kwargs)
        return foo


    def training_step(self, x, *args, **kwargs):
        imgs, pid, camid, img_paths = x
        outputs = self.model(imgs)
        outputs, kd_loss, attn_loss = outputs['out'], outputs['kd_loss'], outputs['attn_loss']
        loss = self.loss.compute(outputs, pid)
        self.log('reid_loss', loss, on_step=False, on_epoch=True)
        loss += kd_loss * self.cfg.kd_loss.alpha
        self.log('kd_loss', kd_loss, on_step=False, on_epoch=True)
        loss += attn_loss * self.cfg.mse.mse_weight
        self.log('attn_loss', attn_loss, on_step=False, on_epoch=True)
        self.log('step', self.current_epoch, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, *args, **kwargs):
        ...

    def configure_optimizers(self):
        optimizer = make_optimizer(self.cfg, self.model)
        ret = {'optimizer': optimizer}

        scheduler = make_scheduler(self.cfg, optimizer, last_epoch=-1)
        ret.update({"lr_scheduler": {'scheduler': scheduler, "interval": "epoch"}})

        return ret

    def find_best_model(self, stream):
        import os
        model_path = os.path.join(self.loggers[-1].log_dir, 'checkpoints')
        if not os.path.exists(model_path):
            return None, None
        models = []
        for f in os.listdir(model_path):
            file_name = os.path.basename(f)
            metric = f'{stream}_mAP'
            if metric in file_name:
                reg = f'{metric}=(?P<val>.*).ckpt'
                search = re.search(reg, file_name)
                val = float(search.groupdict()['val'])
                models.append((val, f))
        if len(models) == 0:
            return None, None
        models.sort()
        return models[-1][0], os.path.join(model_path, models[-1][-1])


    def load_stream(self, model_path, stream):
        if model_path is None:
            return
        dic = {'face': 'lr_face_net', 'body':'nocloth_net'}
        module_name = dic[stream]
        sd = torch.load(model_path, map_location='cpu')['state_dict']
        sd = {k:v for k, v in sd.items() if module_name in k}
        self.load_state_dict(sd, strict=False)


    def validation_epoch_end(self, outputs) -> None:
        feat = self.validate_features()
        if cfg.forward_mode in ['face', 'all']:
            face_ret = self.validates(feat, 'face')
            self.log_dict(face_ret)
        if cfg.forward_mode in ['body', 'all']:
            body_ret = self.validates(feat, 'body')
            self.log_dict(body_ret)
        if cfg.forward_mode == 'all':
            sd = copy.deepcopy(self.state_dict())
            loaded = False
            face_mAP, face_model_path = self.find_best_model('face')
            if face_mAP is not None and face_mAP > face_ret['face_mAP']:
                self.load_stream(face_model_path, 'face')
                loaded = True
            body_mAP, body_model_path = self.find_best_model('body')
            if body_mAP is not None and body_mAP > body_ret['body_mAP']:
                self.load_stream(body_model_path, 'body')
                loaded = True
            if loaded:
                feat = self.validate_features()
            ret = self.validates(feat, 'all')
            self.log_dict(ret)
            self.load_state_dict(sd)

    def validate_features(self):
        qf, query_ids, query_cams = self.extract_feature(self.query_loader)
        gf, gallery_ids, gallery_cams = self.extract_feature(self.test_loader)
        return (qf, query_ids, query_cams), (gf, gallery_ids, gallery_cams)

    def get_feat_by_stream(self, feat, stream):
        assert stream in ['all', 'body', 'face']
        if stream == 'all' or cfg.forward_mode == stream:
            return feat
        assert cfg.forward_mode == 'all'
        (qf, query_ids, query_cams), (gf, gallery_ids, gallery_cams) = feat
        if stream == 'face':
            return (qf[:, :512*7], query_ids, query_cams), (gf[:, :512*7], gallery_ids, gallery_cams)
        elif stream == 'body':
            return (qf[:, 512*7:], query_ids, query_cams), (gf[:, 512*7:], gallery_ids, gallery_cams)


    def validates(self, feat, stream=None):
        (qf, query_ids, query_cams), (gf, gallery_ids, gallery_cams) \
            = self.get_feat_by_stream(feat, stream)

        inference_mode = 0
        if inference_mode == 0:
            dist = 1 - torch.mm(qf.cuda(), gf.cuda().t()).cpu().numpy()
        elif inference_mode == 1:
            dist = torch.zeros(qf.shape[0], gf.shape[0])
            for i in range(qf.shape[0]):
                for j in range(gf.shape[0]):
                    a = qf[i:i+1]
                    b = gf[j:j+1]
                    if not torch.any(a[:, :512*7]):
                        a = a[:, 512*7:]
                        b = b[:, 512*7:]
                    dist[i][j] = 1 - torch.mm(a, b.t())
            dist = dist.numpy()
        else:
            raise
        r, m_ap = evaluation(dist, query_ids, gallery_ids, query_cams, gallery_cams, 50)

        rank = [1, 5, 10]

        ret_dict = {
            f'{stream}_mAP': m_ap,
            'step': self.current_epoch
        }
        for i in rank:
            ret_dict[f'{stream}_R{i}'] = r[i - 1]
        return ret_dict

    def _parse_data_for_eval(self, data):
        imgs = cuda(data[0])
        pids = data[1]
        camids = data[2]
        return imgs, pids, camids

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        pids, camids = [], []

        for d in loader:
            input_img, pid, camid = self._parse_data_for_eval(d)

            outputs = self.model(input_img)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            f1 = outputs.data.cpu()

            model_input = []
            for inputs in input_img:
                inputs = inputs
                try:
                    i0 = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
                except:
                    i0 = inputs
                model_input.append(i0.cuda())


            outputs = self.model(model_input)
            if isinstance(outputs, dict):
                outputs = outputs['out']

            f2 = outputs.data.cpu()

            ff = f1 + f2
            if ff.dim() == 3:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)  # * np.sqrt(ff.shape[2])
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.transpose(1, 2).reshape(ff.size(0), -1)

            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
            pids.extend(pid)
            camids.extend(camid)

        return features, np.asarray(pids), np.asarray(camids)

class DataModule(pl.LightningDataModule):

    def __init__(self, loader):
        super(DataModule, self).__init__()
        self.loader = loader

    def train_dataloader(self):
        return self.loader.train_loader

    def val_dataloader(self):
        return [torch.ones(1, 1)]


if __name__ == '__main__':
    from config import cfg
    from pytorch_lightning.callbacks import ModelCheckpoint

    loader = data.ImageDataManager(cfg)
    data_module = DataModule(loader)
    model = Model(cfg, loader)

    from pytorch_lightning import loggers

    tb_logger = loggers.TensorBoardLogger(
        save_dir='log',
        name='',
        version=cfg.tag,
    )
    trainer = pl.Trainer(precision=16,
                         num_sanity_val_steps=0,
                         logger=tb_logger,
                         check_val_every_n_epoch=5,
                         gpus=[0],
                         callbacks=[
                             ModelCheckpoint(
                                 monitor='all_mAP',
                                 filename='{epoch}-{all_mAP:.4f}',
                                 save_top_k=2,
                                 verbose=True,
                                 mode='max',
                                 save_last=False,
                             ),
                             ModelCheckpoint(
                                 monitor='body_mAP',
                                 filename='{epoch}-{body_mAP:.4f}',
                                 save_top_k=2,
                                 verbose=True,
                                 mode='max',
                                 save_last=False,
                             ),
                             ModelCheckpoint(
                                 monitor='face_mAP',
                                 filename='{epoch}-{face_mAP:.4f}',
                                 save_top_k=2,
                                 verbose=True,
                                 mode='max',
                                 save_last=False,
                             ),
                         ]
                         )

    trainer.fit(model, data_module)
    # trainer.validate(model, data_module, ckpt_path='log/prcc/checkpoints/best.ckpt')
    # trainer.validate(model, data_module, ckpt_path='log/Celeb/checkpoints/best.ckpt')

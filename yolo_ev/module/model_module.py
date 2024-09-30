import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig

from .model.yolox.yolox import YOLOX
from yolo_ev.utils.lr_scheduler import LRScheduler
from yolo_ev.module.model.yolox.utils.boxes import postprocess
from yolo_ev.utils.eval.evaluation import to_coco_format, evaluation

from yolo_ev.module.data.dataset.coco.coco_classes import COCO_CLASSES

class ModelModule(pl.LightningModule):

    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config
        self.model = YOLOX(self.full_config.model)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
                    
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()

    def forward(self, x, targets=None):
        return self.model(x, targets)
    
    def training_step(self, batch, batch_idx):
        self.model.train()
        imgs, targets, _, _ = batch
        imgs = imgs.to(torch.float32)
        targets = targets.to(torch.float32)
        
        outputs = self(imgs, targets)
        loss = outputs["total_loss"]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['train_loss'].mean()
        self.log('epoch_train_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        imgs, targets, img_info, _ = batch
        imgs = imgs.to(torch.float32)
        targets = targets.to(torch.float32)
        
        predictions = self(imgs, targets)
        processed_pred = postprocess(prediction=predictions,
                                     num_classes=self.full_config.model.head.num_classes,
                                     conf_thre=self.full_config.model.postprocess.conf_thre,
                                     nms_thre=self.full_config.model.postprocess.nms_thre)
        
        height, width = img_info
        classes = COCO_CLASSES
        num_data = len(targets) ## = batch_size
        gt, pred = to_coco_format(gts=targets, detections=processed_pred, categories=classes, height=height, width=width)
        # scores ('AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L')
        scores = evaluation(Gt=gt, Dt=pred, num_data=num_data)
        
        # APをvalidation lossの代わりにする（val_lossの代わりにスコアをモニタリング）
        self.log('val_loss', scores['AP'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return scores

    def on_validation_epoch_end(self):
        # avg_val_loss = self.trainer.callback_metrics['val_loss'].mean()
        # self.log('epoch_val_loss', avg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        pass
        
    def configure_optimizers(self):
        # Learning rate を設定
        lr = self.full_config.scheduler.warmup_lr

        # パラメータグループを作成
        pg0, pg1, pg2 = [], [], []

        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        # オプティマイザーを設定
        optimizer = torch.optim.SGD(
            pg0,
            lr=lr,
            momentum=self.full_config.optimizer.momentum,
            nesterov=True
        )
        optimizer.add_param_group({"params": pg1, "weight_decay": self.full_config.optimizer.weight_decay})
        optimizer.add_param_group({"params": pg2})

        # スケジューラーの設定
        scheduler = {
            'scheduler': LRScheduler(
                optimizer=optimizer,
                name=self.full_config.scheduler.name,
                lr=lr,
                iters_per_epoch=self.full_config.training.max_step,
                total_epochs=self.full_config.training.max_epoch,
                warmup_epochs=self.full_config.scheduler.warmup_epochs,
                warmup_lr_start=self.full_config.scheduler.warmup_lr,
                no_aug_epochs=self.full_config.scheduler.no_aug_epochs,
                min_lr_ratio=self.full_config.scheduler.min_lr_ratio,
            ),
            'interval': 'step',  
            'frequency': 1,  # how often to update the scheduler
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}



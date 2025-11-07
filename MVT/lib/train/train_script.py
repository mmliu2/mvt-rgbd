import os
# loss function related
from lib.utils.box_ops import giou_loss, box_pixelwise_metrics
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer, LTRTrainerDepth
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.mobilevit_track.mobilevit_track import build_mobilevit_track
from lib.models.mobilevit_track.mobilevit_track_depth import build_mobilevit_track_depth
# forward propagation related
from lib.train.actors import MobileViTTrackActor, MobileViTTrackActorDepth
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for MobileViT-Track'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg

    # alert! there are two config files: experiments/tracker-name.yaml and lib.config.tracker-name.config.py,
    # and both files contain identical information regrading the backbone, head, training and testing params etc.
    # The training code reads both files, but the priority is given to experiments/tracker-name.yaml file
    config_module.update_config_from_file(settings.cfg_file)

    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == 'mobilevit_track':
        net = build_mobilevit_track(cfg)
    elif settings.script_name == 'mobilevit_track_depth':
        net = build_mobilevit_track_depth(cfg, training=True)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    # if torch.cuda.is_available():
    #     net.cuda()
    #     if settings.local_rank != -1:
    #         # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
    #         net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
    #         settings.device = torch.device("cuda:%d" % settings.local_rank)
    #     else:
    #         settings.device = torch.device("cuda:0")
    # else:
    #     settings.device = torch.device("cpu")
    net = net.to(settings.device)



    state_dict = torch.load(os.path.join(settings.save_dir, "checkpoints/train/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing/MobileViT_Track_ep0100_state_dict.pt"), map_location="cpu")
    state_dict = {name:state_dict[name] for name in state_dict if 'backbone.conv_1' not in name}
    print(f"Loaded {len(state_dict)} tensors from checkpoint:\n")

    # TODO_CHECKPOINT mvt
    missing, unexpected = net.load_state_dict(state_dict, strict=False)

    if settings.freeze:
        for name, param in net.named_parameters():
            if 'prompt' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    print()
    # Total number of parameters
    total_params = sum(p.numel() for p in net.parameters())
    print("Total parameters:", total_params)
    # Number of trainable parameters
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)
    # Number of frozen parameters
    frozen_params = total_params - trainable_params
    print("Frozen parameters:", frozen_params)
    print()






    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    if  settings.script_name == "mobilevit_track":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = MobileViTTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

        # if cfg.TRAIN.DEEP_SUPERVISION:
        #     raise ValueError("Deep supervision is not supported now.")

        # Optimizer, parameters, and learning rates
        optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
        use_amp = getattr(cfg.TRAIN, "AMP", False)
        trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    elif settings.script_name == "mobilevit_track_depth":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(), 'pixelwise': box_pixelwise_metrics}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = MobileViTTrackActorDepth(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

        # if cfg.TRAIN.DEEP_SUPERVISION:
        #     raise ValueError("Deep supervision is not supported now.")

        # Optimizer, parameters, and learning rates
        optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
        use_amp = getattr(cfg.TRAIN, "AMP", False)
        trainer = LTRTrainerDepth(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)
        # trainer = LTRTrainerDepthAdapt(actor, [loader_train, loader_val], loader_test, optimizer, settings, lr_scheduler, use_amp=use_amp)

    else:
        raise ValueError("illegal script name")

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True) # TODO_CHECKPOINT weight loading broken
    # trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True) 

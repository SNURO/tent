import logging

import torch
import torch.optim as optim

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

import time
import tent
import norm
import oracle
from utils import *

import ipdb

from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "oracle":
        logger.info("test-time adaptation: ORACLE")
        model = setup_oracle(base_model)
    # evaluate on each severity and type of corruption in turn
    for severity in cfg.CORRUPTION.SEVERITY:
        for corruption_type in cfg.CORRUPTION.TYPE:
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
            #TENT default optimizer는 last FC layer 을 parameter로 가지고 있지 않음
            if cfg.EXPERIMENTAL.LINEAR_RETRAIN and cfg.MODEL.ADAPTATION == "tent":
                params, param_names = tent.collect_params(model)
                model.optimizer = setup_optimizer(params)
            model.optimizer.param_groups[0]['lr'] = cfg.OPTIM.LR

            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            if cfg.EXPERIMENTAL.LINEAR_RETRAIN and cfg.MODEL.ADAPTATION == "oracle":
                logger.info("linear_finetune activated")
                tent.linear_retrain(model.model, freeze_bias=False)

            #check_freeze(model.model)
            
            start = time.time()
            if cfg.MODEL.ADAPTATION == "oracle":
                #acc = oracle_accuracy_multi(model, x_test, y_test, cfg.TEST.BATCH_SIZE, epochs=50, linear_retrain=True)
                acc = oracle_accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, iteration=10, linear_retrain=True)
            else:
                acc = clean_accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, iteration=cfg.ITERATION)

            #ipdb.set_trace()
            end = time.time()
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}      {end - start:.0f}s")

            if cfg.EXPERIMENTAL.LINEAR_RETRAIN and cfg.MODEL.ADAPTATION == "tent":
                logger.info("TENT linear_finetune activated")
                #TENT default optimizer는 last FC layer 을 parameter로 가지고 있지 않음. optimizer 변경하지 않으면 update 안됨
                tent.linear_retrain(model.model)
                model.optimizer=setup_optimizer(model.model.parameters())
                acc = oracle_accuracy_multi(model, x_test, y_test, cfg.TEST.BATCH_SIZE, epochs=50, linear_retrain = True)
                # linear_retrain only true when retraining on TENT, because forward is defined in confusing way
            
                end_finetune = time.time()
                err = 1. - acc
                logger.info(f"error % [{corruption_type}{severity}]_finetune: {err:.2%}      {end_finetune - end:.0f}s")

            #save_fc_params(model, 'freeze_bias_iter100')


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model

def setup_oracle(model):
    """Set up the baseline source model without adaptation."""
    model.train()
    model.requires_grad_(True)
    optimizer = setup_optimizer(model.parameters())
    oracle_model = oracle.Oracle(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return oracle_model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')

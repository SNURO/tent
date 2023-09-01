import logging

import torch
import torch.optim as optim

from robustbench.data import load_cifar10c, load_cifar10
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

import time
import tent
import norm
import oracle
from utils import *

import ipdb

from conf import cfg, load_cfg_fom_args


# 보류하고 bufr 로 넘어감
logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)

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
            #ipdb.set_trace()
            
            #TODO: load original cifar10 for histogram
            
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            #load_fc_params(model, 'no_bias_iter100')
            check_freeze(model.model)
            #disable_batchnorm(model)
            start = time.time()

            acc = clean_accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, iteration=cfg.ITERATION)
            end = time.time()
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}      {end - start:.0f}s")

        #save_fc_params(model, 'source')


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


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

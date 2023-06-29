import logging
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def Test(config):
    pass

def RigidBodyTranslations(config):
    """ Model rigid body translations, optimizing sigma. """
    from eryx.models import RigidBodyTranslations
    task = config.RigidBodyTranslations
    logger.debug('Setting up model')
    model = RigidBodyTranslations(config.setup.pdb_path, 
                                  config.setup.hsampling,
                                  config.setup.ksampling,
                                  config.setup.lsampling,
                                  res_limit=config.setup.res_limit,
                                  batch_size=config.setup.batch_size,
                                  n_processes=config.setup.n_processes,
                                  expand_friedel=task.get('expand_friedel') if task.get('expand_friedel') is not None else True)
    logger.debug('Optimizing model')
    ccs, sigmas = model.optimize(np.load(config.setup.exp_map), 
                                 task.sigmas_min, 
                                 task.sigmas_max, 
                                 n_search=task.n_search)
    model.plot_scan(output=os.path.join(config.setup.root_dir, "figs/scan_rigidbodytranslations.png"))
    np.save(os.path.join(config.setup.root_dir, "models/rigidbodytranslations.npy"), model.opt_map)
    np.save(os.path.join(config.setup.root_dir, "base/mtransform.npy"), model.transform)

def LiquidLikeMotions(config):
    """ Model liquid like motions, optimizing sigma and gamma. """
    from eryx.models import LiquidLikeMotions
    from parse_yaml import expand_sampling
    task = config.LiquidLikeMotions
    logger.debug('Setting up model')
    expand_sampling(config)
    model = LiquidLikeMotions(config.setup.pdb_path,
                              config.setup.hsampling,
                              config.setup.ksampling,
                              config.setup.lsampling,
                              res_limit=config.setup.res_limit,
                              batch_size=config.setup.batch_size,
                              n_processes=config.setup.n_processes,
                              asu_confined=task.asu_confined)
    logger.debug('Optimizing model')
    model.optimize(np.load(config.setup.exp_map),
                   task.sigmas_min,
                   task.sigmas_max,
                   task.gammas_min,
                   task.gammas_max,
                   ns_search=task.ns_search,
                   ng_search=task.ng_search)
    ntransform, extension = "ctransform", ""
    if task.asu_confined:
        ntransform, extension = "mtransform", "_asuconfined"
    model.plot_scan(output=os.path.join(config.setup.root_dir, f"figs/scan_liquidlikemotions{extension}.png"))
    np.save(os.path.join(config.setup.root_dir, f"models/liquidlikemotions{extension}.npy"), model.opt_map)
    np.save(os.path.join(config.setup.root_dir, f"base/{ntransform}.npy"), model.transform)

def OnePhonon(config):
    """ Model elastic network-based motions in the one-phonon approximation. """
    from eryx.models import OnePhonon
    from parse_yaml import expand_sampling
    task = config.OnePhonon
    logger.debug('Setting up model')
    expand_sampling(config, force_int=True)
    model = OnePhonon(config.setup.pdb_path,
                      config.setup.hsampling,
                      config.setup.ksampling,
                      config.setup.lsampling,
                      gnm_cutoff=task.gnm_cutoff,
                      gamma_intra=task.gamma_intra,
                      gamma_inter=task.gamma_inter,
                      batch_size=config.setup.batch_size,
                      n_processes=config.setup.n_processes,
                      expand_p1=task.get('expand_p1') if task.get('expand_p1') is not None else True)
    logger.debug('Computing diffuse map')
    Id = model.apply_disorder(use_data_adp=True)
    Id = Id.reshape(model.map_shape)
    np.save(os.path.join(config.setup.root_dir, f"models/onephonon.npy"), Id)

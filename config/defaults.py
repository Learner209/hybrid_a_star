import os
from yacs.config import CfgNode as CN
import numpy as np

_C = CN()

_C.x_resolution = 0.01
_C.y_resolution = 0.01
_C.start =  [10, 10]
_C.end = [100, 100]
_C.sx = 10
_C.sy = 10
_C.syaw = 120
_C.gx = 100
_C.gy = 100
_C.gyaw = 90
_C.yaw_reso = 5
_C.obsmap = 'map.npy'
_C.alpha = 0.4
_C.potential_field_weight = 0.3
_C.steering_penalty_weight = 0.8
_C.d_o_max = 0.05

_C.reeds_shepp = CN()
_C.reeds_shepp.max_curvature = 0.1
_C.reeds_shepp.step_size = 0.2
_C.reeds_shepp.max_length = 1000.0
_C.reeds_shepp.sample_rate = 3
_C.reeds_shepp.gear_cost = 100.0
_C.reeds_shepp.steer_angle_cost = 1.0
_C.reeds_shepp.steer_change_cost = 5.0

_C.output_dir = "traj_solver"

_C.num_workers = 0

_C.model = CN()
_C.model.device = "cuda:0"

_C.solver = CN()
_C.solver.num_epochs = 10000
_C.solver.max_lr = 0.001
_C.solver.end_lr = 0.0001
_C.solver.bias_lr_factor = 1
_C.solver.momentum = 0.9
_C.solver.weight_decay = 0.0005
_C.solver.weight_decay_bias = 0.0
_C.solver.gamma = 0.1
_C.solver.lrate_decay = 250
_C.solver.steps = (30000,)
_C.solver.warmup_factor = 1.0 / 3
_C.solver.warmup_iters = 500
_C.solver.warmup_method = "linear"
_C.solver.num_iters = 10000
_C.solver.min_factor = 0.1
_C.solver.log_interval = 1
_C.solver.show_interval = 500

_C.solver.optimizer = 'Adam'
_C.solver.scheduler = 'ConstantScheduler'
_C.solver.scheduler_decay_thresh = 0.00005
_C.solver.do_grad_clip = False
_C.solver.grad_clip_type = 'norm'  # norm or value
_C.solver.grad_clip = 1.0
_C.solver.batch_size = 256
_C.solver.traj_balance_loss_weight = 1
_C.solver.voronoi_potential_field_loss_weight = 10
_C.solver.obstacle_collision_field_loss_weight = 5
_C.solver.curvature_constrain_loss_weight = 4
_C.solver.steering_loss_weight = 50

#### IO configs#####
_C.solver.load_model = ""
_C.solver.load = ""
_C.solver.broadcast_buffers = False
_C.solver.dist = CN()

_C.dataloader = CN()
_C.dataloader.num_workers = 0
_C.dataloader.collator = 'DefaultBatchCollator'
_C.dataloader.pin_memory = False

_C.datasets = CN()
_C.datasets.train = ()
_C.datasets.test = ""

_C.input = CN()
_C.input.shuffle = True


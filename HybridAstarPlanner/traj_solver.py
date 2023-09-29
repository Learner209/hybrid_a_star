import torch
import torch.nn as nn
from loguru import logger

class TRAJSolver(nn.Module):
        def __init__(self, cfg, plan_planner, data, batch_size, H, W):
            super().__init__()
            self.plan_planner = plan_planner
            self.cfg = cfg
            self.batch_size = self.cfg.batch_size
            self.traj = nn.Parameter(data, requires_grad=True)
            self.seq_len, _ = data.shape
            self.batch_size = batch_size
            self.traj_balance_loss_weight = torch.tensor(cfg.traj_balance_loss_weight)
            self.voronoi_potential_field_loss_weight = torch.tensor(cfg.voronoi_potential_field_loss_weight)
            self.obstacle_collision_field_loss_weight = torch.tensor(cfg.obstacle_collision_field_loss_weight)
            self.curvature_constrain_loss_weight = torch.tensor(cfg.curvature_constrain_loss_weight)
            self.steering_loss_weight = torch.tensor(cfg.steering_loss_weight)
            self.normalized_ratio = torch.tensor([1/H, 1/W])

            # logger.info(f'The init trajectory is: {data}')

            self.register_buffer(f'history_traj', torch.zeros(10000, self.seq_len, 2))

        def forward(self, dps):
            put_ops_id = (self.history_traj == 0).all(dim=1).nonzero()[0, 0].item()
            self.history_traj[put_ops_id] = self.traj.detach()
            overall_losses = []

            N, _ = self.traj.shape
            normalized_point = (self.traj * self.normalized_ratio.to(self.traj.device)).float()
            # torch.set_printoptions(threshold=torch.inf)
            # print(f'the normalized point is {normalized_point}')
            # assert False
            for bid in range(self.batch_size):
                losses = 0
                # print(f'the self.traj is {self.traj[0:100]}')
                for idx in range(N):

                    traj_balance_loss = self.plan_planner.traj_balance_loss(self.traj, index = idx)
                    curvature_constrain_loss = self.plan_planner.curvature_constrain_loss(self.traj, index = idx, curvature_max=torch.tensor(0.04))
                    assert normalized_point[idx][0] <= 1 and normalized_point[idx][1] <= 1 and normalized_point[idx][0] >= 0 and normalized_point[idx][1] >= 0, f'the normalized_point is {idx} and the {normalized_point[idx]} and the normalized_point is {normalized_point}'
                    _, _, _, voronoi_potential_field_loss = self.plan_planner.voronoi_potential_field_loss(normalized_point[idx], torch.tensor(0))
                    obstacle_collision_field_loss = self.plan_planner.obstacle_collision_field_loss(normalized_point[idx])
                    steering_loss = self.plan_planner.steering_loss(self.traj, idx, 0.04)

                    loss = self.traj_balance_loss_weight * traj_balance_loss + \
                        self.voronoi_potential_field_loss_weight * voronoi_potential_field_loss + \
                        self.obstacle_collision_field_loss_weight *obstacle_collision_field_loss * 1000 + \
                        self.curvature_constrain_loss_weight * curvature_constrain_loss + \
                        self.steering_loss_weight * steering_loss      # Four loss composition
                    
                    # print(f'The traj_balance_loss: {traj_balance_loss} \n \
                    #     and the voronoi_potential_field_loss:{voronoi_potential_field_loss} \n \
                    #     and the obstacle_collision_field_loss:{obstacle_collision_field_loss} \n \
                    #     and the curvature_constrain_loss:{curvature_constrain_loss} \n \
                    #     and the steering loss:{steering_loss}')

                    losses += loss

                    # print(f'The grad_fn for traj_balance_loss: {traj_balance_loss.grad_fn} grad:{traj_balance_loss} \n \
                    #     and the grad_fn for voronoi_potential_field_loss:{voronoi_potential_field_loss.grad_fn} grad:{voronoi_potential_field_loss} \n \
                    #     and the grad_fn for obstacle_collision_field_loss:{obstacle_collision_field_loss.grad_fn} grad:{obstacle_collision_field_loss} \n \
                    #     and the grad_fn for curvature_constrain_loss:{curvature_constrain_loss.grad_fn} grad:{curvature_constrain_loss}')
                overall_losses.append(losses)

            loss = torch.stack(overall_losses).mean()  
            output = {"traj_balance_loss": traj_balance_loss,
                    "curvature_constrain_loss": curvature_constrain_loss,
                    "voronoi_potential_field_loss": voronoi_potential_field_loss,
                    "obstacle_collision_field_loss": obstacle_collision_field_loss,
                    "steering_loss":steering_loss
                    }

            loss_dict = {"loss": loss}
            return output, loss_dict

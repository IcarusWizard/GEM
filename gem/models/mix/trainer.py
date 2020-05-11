import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tabulate import tabulate
from itertools import chain

from gem.utils import select_gpus

class MixTrainer:
    def __init__(self, sensor, predictor, train_loader, val_loader=None, test_loader=None, config={}):
        self.sensor = sensor
        self.predictor = predictor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        # config gpus
        select_gpus(self.config['gpu']) 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.predictor = self.predictor.to(self.device)
        self.sensor = self.sensor.to(self.device)

        # create log writer
        self.writer = SummaryWriter(config['log_name'])

        self.train_iter = iter(self.train_loader) # used in training

        # config optimizer
        self.optim = torch.optim.Adam(chain(self.sensor.parameters(), self.predictor.parameters()), lr=self.config['m_lr'], 
                                      betas=(self.config['m_beta1'], self.config['m_beta2']))

    def train(self):
        for step in tqdm(range(self.config['steps'])):
            self.train_step()

            if step % self.config['log_step'] == 0:
                self.log_step(step)
        self.log_step(self.config['steps'])

    def get_loss_info(self, batch):
        obs, action, reward = self.parse_batch(batch)

        T, B = obs.shape[:2]
        obs = obs.view(T * B, *obs.shape[2:])
        emb = self.sensor.encode(obs, output_dist=True).mode().view(T, B, -1)

        predictor_loss, prediction, info = self.predictor(emb, action, reward, use_emb_loss=False)

        pre_emb = prediction['emb'].view(T * B, -1)
        pre_obs_dist = self.sensor.decode(pre_emb, output_dist=True)
        reconstruction_loss = - pre_obs_dist.log_prob(obs)
        reconstruction_loss = torch.mean(torch.sum(reconstruction_loss, dim=(1, 2, 3)))

        loss = reconstruction_loss + predictor_loss
        info.update({
            "loss" : loss.item(),
            "renconstruction_loss" : reconstruction_loss.item(),
        })

        return loss, info
    
    def train_step(self):
        batch = next(self.train_iter)

        loss, info = self.get_loss_info(batch)

        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(chain(self.sensor.parameters(), self.predictor.parameters()), self.config['m_grad_clip'])
        self.optim.step()
        info.update({"model_grad_norm" : grad_norm})

        self.last_train_info = info

    def log_step(self, step):
        val_loss, val_info = self.test_whole(self.val_loader)

        print('In Step {}'.format(step))
        print('-' * 15)
        print('In training set:')
        print(tabulate(self.last_train_info.items(), numalign="right"))
        print('In validation set:')
        print(tabulate(val_info.items(), numalign="right"))

        for k in val_info:
            self.writer.add_scalars('world_model/' + k, {'train' : self.last_train_info[k], 
                                    'val' : val_info[k]}, global_step=step)
        self.writer.add_scalar('world_model/grad_norm', self.last_train_info['model_grad_norm'], global_step=step)

        with torch.no_grad():
            batch = next(self.train_iter)
            obs, action, reward = self.parse_batch(batch)

            obs = obs[:, :8].permute(1, 0, 2, 3, 4).contiguous() # [B, T]
            action = action[:, :8].contiguous() # [T, B]
            if self.config['predict_reward']:
                reward = reward[:, :8].contiguous() # [T, B]

            B, T = obs.shape[:2]
            _obs = obs.view(B * T, *obs.shape[2:])
            emb = self.sensor.encode(_obs).view(B, T, -1) # [B, T]
            emb = emb.permute(1, 0, 2).contiguous() # [T, B]

            # log compressed video
            compressed_video = torch.clamp(self.sensor.decode(emb.view(-1, emb.shape[-1])), -0.5, 0.5)
            compressed_video = compressed_video.view(*emb.shape[:2], *compressed_video.shape[1:]).permute(1, 0, 2, 3, 4)
            self.writer.add_video('compressed_video', 
                torch.cat([obs + 0.5, compressed_video + 0.5, (compressed_video - obs + 1) / 2], dim=0), 
                global_step=step, fps=self.config['fps'])

            # log predicted video
            _, prediction, _ = self.predictor(emb, action, reward)
            predicted_emb = prediction['emb']
            predicted_video = torch.clamp(self.sensor.decode(predicted_emb.view(-1, emb.shape[-1])), -0.5, 0.5)
            predicted_video = predicted_video.view(*emb.shape[:2], *predicted_video.shape[1:]).permute(1, 0, 2, 3, 4)
            self.writer.add_video('predicted_video', 
                torch.cat([obs + 0.5, predicted_video + 0.5, (predicted_video - obs + 1) / 2], dim=0),
                global_step=step, fps=self.config['fps'])

            # log generated video
            generation = self.predictor.generate(emb[0], emb.shape[0], action)
            generated_emb = generation['emb']
            generated_video = torch.clamp(self.sensor.decode(generated_emb.view(-1, emb.shape[-1])), -0.5, 0.5)
            generated_video = generated_video.view(*emb.shape[:2], *generated_video.shape[1:]).permute(1, 0, 2, 3, 4)
            self.writer.add_video('generated_video_true_action', 
                torch.cat([obs + 0.5, generated_video + 0.5, (generated_video - obs + 1) / 2], dim=0), 
                global_step=step, fps=self.config['fps'])

            # log generated video (action also generated)
            if self.config['action_mimic']:
                generation = self.predictor.generate(emb[0], emb.shape[0])
                generated_emb = generation['emb']
                generated_video = torch.clamp(self.sensor.decode(generated_emb.view(-1, emb.shape[-1])), -0.5, 0.5)
                generated_video = generated_video.view(*emb.shape[:2], *generated_video.shape[1:]).permute(1, 0, 2, 3, 4)
                self.writer.add_video('generated_video_fake_action', 
                    torch.cat([obs + 0.5, generated_video + 0.5, (generated_video - obs + 1) / 2], dim=0), 
                    global_step=step, fps=self.config['fps'])

        self.writer.flush()

    def parse_batch(self, batch):
        obs = batch['image'].permute(1, 0, 2, 3, 4).to(self.device).contiguous()
        action = batch['action'].permute(1, 0, 2).to(self.device).contiguous()
        reward = batch['reward'].permute(1, 0).unsqueeze(dim=-1).to(self.device).contiguous() if self.config['predict_reward'] else None  
        return obs, action, reward    

    def test_whole(self, loader):
        with torch.no_grad():
            num = 0
            info = {}
            loss = 0
            for batch in iter(loader):
                num += 1

                _loss,  _info = self.get_loss_info(batch)

                loss += _loss
                for k in _info.keys():
                    info[k] = info.get(k, 0) + _info[k]

            loss = loss / num
            for k in info.keys():
                info[k] = info[k] / num

        return loss.item(), info

    def save(self, filename):
        test_loss, test_info = self.test_whole(self.test_loader)
        torch.save({
            "sensor_state_dict" : self.sensor.state_dict(),
            "predictor_state_dict" : self.predictor.state_dict(),
            "optimizer_state_dict" : self.optim.state_dict(),
            "test_info" : test_info,
            "config" : self.config,
            "sensor_parameters" : self.config['sensor_param'],
            "predictor_parameters" : self.config['predictor_param'],
            "seed" : self.config['seed'],
        }, filename)

    def restore(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.sensor.load_state_dict(checkpoint['sensor_state_dict'])
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.sensor = self.sensor.to(self.device) # make sure model on right device
        self.predictor = self.predictor.to(self.device)
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
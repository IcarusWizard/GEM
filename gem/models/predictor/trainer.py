import torch

from degmo.base import Trainer
from degmo.utils import nats2bits

class PredictorTrainer(Trainer):
    def __init__(self, model, coder, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__(model, train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

        self.coder = coder.to(self.device)

        # config optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], 
                                      betas=(self.config['beta1'], self.config['beta2']))
    
    def train_step(self):
        batch = next(self.train_iter)

        obs, action, reward = self.parse_batch(batch)

        loss, prediction, info = self.model(obs, action, reward)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.last_train_loss = loss.item()
        self.last_train_perdiction = prediction
        self.last_train_info = info

    def log_step(self, step):
        val_loss, val_info = self.test_whole(self.val_loader)

        print('In Step {}'.format(step))
        print('-' * 15)
        print('In training set:')
        for k in self.last_train_info.keys():
            print('{0} is {1:{2}} bits'.format(k, nats2bits(self.last_train_info[k]), '.5f'))
        print('In validation set:')
        for k in val_info.keys():
            print('{0} is {1:{2}} bits'.format(k, nats2bits(val_info[k]), '.5f'))

        for k in self.last_train_info.keys():
            self.writer.add_scalars(k, {'train' : nats2bits(self.last_train_info[k]), 
                                    'val' : nats2bits(val_info[k])}, global_step=step)
        
        with torch.no_grad():

            batch = next(self.train_iter)
            obs, action, reward = self.parse_batch(batch)

            images = batch['image'].to(self.device)[:8].contiguous()
            obs = obs[:, :8].contiguous()
            action = action[:, :8].contiguous()
            if self.config['predict_reward']:
                reward = reward[:, :8].contiguous()

            # log compressed video
            compressed_video = self.coder.decode(obs.view(-1, obs.shape[-1]))
            compressed_video = compressed_video.view(*obs.shape[:2], *compressed_video.shape[1:]).permute(1, 0, 2, 3, 4)
            self.writer.add_video('compressed_video', torch.cat([images, compressed_video, (compressed_video - images + 1) / 2], dim=0), global_step=step)

            # log predicted video
            _, prediction, _ = self.model(obs, action, reward)
            predicted_obs = prediction['obs']
            predicted_video = self.coder.decode(predicted_obs.view(-1, obs.shape[-1]))
            predicted_video = predicted_video.view(*obs.shape[:2], *predicted_video.shape[1:]).permute(1, 0, 2, 3, 4)
            self.writer.add_video('predicted_video', torch.cat([images, predicted_video, (predicted_video - images + 1) / 2], dim=0), global_step=step)

            # log generated video
            generation = self.model.generate(obs[0], obs.shape[0], action)
            generated_obs = generation['obs']
            generated_video = self.coder.decode(generated_obs.view(-1, obs.shape[-1]))
            generated_video = generated_video.view(*obs.shape[:2], *generated_video.shape[1:]).permute(1, 0, 2, 3, 4)
            self.writer.add_video('generated_video_true_action', torch.cat([images, generated_video, (generated_video - images + 1) / 2], dim=0), global_step=step)

            # log generated video (action also generated)
            if self.config['action_mimic']:
                generation = self.model.generate(obs[0], obs.shape[0])
                generated_obs = generation['obs']
                generated_video = self.coder.decode(generated_obs.view(-1, obs.shape[-1]))
                generated_video = generated_video.view(*obs.shape[:2], *generated_video.shape[1:]).permute(1, 0, 2, 3, 4)
                self.writer.add_video('generated_video_fake_action', torch.cat([images, generated_video, (generated_video - images + 1) / 2], dim=0), global_step=step)

                # generate with coder prior
                obs0 = self.coder.prior.sample((8, self.coder.latent_dim)) if isinstance(self.coder.prior, torch.distributions.Normal) else self.coder.prior.sample(8) 
                generation = self.model.generate(obs0, obs.shape[0])
                generated_obs = generation['obs']
                generated_video = self.coder.decode(generated_obs.view(-1, obs.shape[-1]))
                generated_video = generated_video.view(*obs.shape[:2], *generated_video.shape[1:]).permute(1, 0, 2, 3, 4)
                self.writer.add_video('prior_video_fake_action', generated_video, global_step=step)




    
    def parse_batch(self, batch):
        if 'emb' not in batch.keys():
            images = batch['image'].to(self.device)
            B, T = images.shape[:2]
            images = images.view(B * T, *images.shape[2:])
            obs = self.coder.encode(images)
            obs = obs.view(B, T, obs.shape[1]).permute(1, 0, 2)
        else:
            obs = batch['emb'].permute(1, 0, 2).to(self.device)
        action = batch['action'].permute(1, 0, 2).to(self.device)
        reward = batch['reward'].permute(1, 0, 2).to(self.device) if self.config['predict_reward'] else None    
        return obs, action, reward    

    def test_whole(self, loader):
        with torch.no_grad():
            num = 0
            info = {}
            loss = 0
            for batch in iter(loader):
                num += 1

                obs, action, reward = self.parse_batch(batch)

                _loss, _,  _info = self.model(obs, action, reward)

                loss += _loss
                for k in _info.keys():
                    info[k] = info.get(k, 0) + _info[k]
            loss = loss / num
            for k in info.keys():
                info[k] = info[k] / num

        return loss.item(), info

    def save(self, filename):
        test_loss, _ = self.test_whole(self.test_loader)
        torch.save({
            "model_state_dict" : self.model.state_dict(),
            "optimizer_state_dict" : self.optim.state_dict(),
            "test_loss" : nats2bits(test_loss),
            "config" : self.config,
            "model_parameters" : self.config['model_param'],
            "seed" : self.config['seed'],
        }, filename)

    def restore(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device) # make sure model on right device
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
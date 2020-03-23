import torch

from degmo.base import Trainer
from degmo.utils import nats2bits

class PredictorTrainer(Trainer):
    def __init__(self, model, coder, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__(model, train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

        self.coder = coder

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
            self.coder = self.coder.to(self.device)

            batch = next(self.train_iter)
            obs, action, reward = self.parse_batch(batch)

            # log true video
            decode_video = self.coder.decode(obs.view(-1, obs.shape[-1]))
            decode_video = decode_video.view(*obs.shape[:2], *decode_video.shape[1:])
            self.writer.add_video('input_video', decode_video.permute(1, 0, 2, 3, 4), global_step=step)

            # log predicted video
            _, prediction, _ = self.model(obs, action, reward)
            predicted_obs = prediction['obs']
            predicted_video = self.coder.decode(predicted_obs.view(-1, obs.shape[-1]))
            predicted_video = predicted_video.view(*obs.shape[:2], *predicted_video.shape[1:])
            self.writer.add_video('predicted_video', predicted_video.permute(1, 0, 2, 3, 4), global_step=step)

            # log generated video
            generation = self.model.generate(obs[0], obs.shape[0], action)
            generated_obs = generation['obs']
            generated_video = self.coder.decode(generated_obs.view(-1, obs.shape[-1]))
            generated_video = generated_video.view(*obs.shape[:2], *generated_video.shape[1:])
            self.writer.add_video('generated_video_true_action', predicted_video.permute(1, 0, 2, 3, 4), global_step=step)

            # log generated video (action also generated)
            if self.config['action_mimic']:
                generation = self.model.generate(obs[0], obs.shape[0], action)
                generated_obs = generation['obs']
                generated_video = self.coder.decode(generated_obs.view(-1, obs.shape[-1]))
                generated_video = generated_video.view(*obs.shape[:2], *generated_video.shape[1:])
                self.writer.add_video('generated_video_fake_action', predicted_video.permute(1, 0, 2, 3, 4), global_step=step)

            self.coder = self.coder.cpu()
    
    def parse_batch(self, batch):
        obs = batch['obs'].permute(1, 0, 2).to(self.device)
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
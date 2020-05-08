import torch

from gem.models.base import Trainer
from gem.utils import nats2bits

class PredictorTrainer(Trainer):
    def __init__(self, predictor, sensor, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__(train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

        self.predictor = predictor.to(self.device)
        self.sensor = sensor.to(self.device)

        # config optimizer
        self.optim = torch.optim.Adam(self.predictor.parameters(), lr=self.config['m_lr'], 
                                      betas=(self.config['m_beta1'], self.config['m_beta2']))
    
    def train_step(self):
        batch = next(self.train_iter)

        emb, action, reward = self.parse_batch(batch)

        loss, prediction, info = self.predictor(emb, action, reward)

        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.config['grad_clip'])
        self.optim.step()
        info.update({"predictor_grad_norm" : grad_norm})

        self.last_train_info = info

    def log_step(self, step):
        val_loss, val_info = self.test_whole(self.val_loader)

        print('In Step {}'.format(step))
        print('-' * 15)
        print('In training set:')
        for k in self.last_train_info.keys():
            print('{0} is {1:{2}} bits'.format(k, nats2bits(self.last_train_info[k]), '.2f'))
        print('In validation set:')
        for k in val_info.keys():
            print('{0} is {1:{2}} bits'.format(k, nats2bits(val_info[k]), '.2f'))

        for k in val_info.keys():
            self.writer.add_scalars('predictor/' + k, {'train' : nats2bits(self.last_train_info[k]), 
                                    'val' : nats2bits(val_info[k])}, global_step=step)
        self.writer.add_scalar('predictor/grad_norm', self.last_train_info['predictor_grad_norm'], global_step=step)
        
        with torch.no_grad():

            batch = next(self.train_iter)
            emb, action, reward = self.parse_batch(batch)

            images = batch['image'].to(self.device)[:8].contiguous()
            emb = emb[:, :8].contiguous()
            action = action[:, :8].contiguous()
            if self.config['predict_reward']:
                reward = reward[:, :8].contiguous()

            # log compressed video
            compressed_video = self.sensor.decode(emb.view(-1, emb.shape[-1]))
            compressed_video = compressed_video.view(*emb.shape[:2], *compressed_video.shape[1:]).permute(1, 0, 2, 3, 4)
            self.writer.add_video('compressed_video', torch.cat([images, compressed_video, (compressed_video - images + 1) / 2], dim=0), global_step=step, fps=self.config['fps'])

            # log predicted video
            _, prediction, _ = self.predictor(emb, action, reward)
            predicted_emb = prediction['emb']
            predicted_video = self.sensor.decode(predicted_emb.view(-1, emb.shape[-1]))
            predicted_video = predicted_video.view(*emb.shape[:2], *predicted_video.shape[1:]).permute(1, 0, 2, 3, 4)
            self.writer.add_video('predicted_video', torch.cat([images, predicted_video, (predicted_video - images + 1) / 2], dim=0), global_step=step, fps=self.config['fps'])

            # log generated video
            generation = self.predictor.generate(emb[0], emb.shape[0], action)
            generated_emb = generation['emb']
            generated_video = self.sensor.decode(generated_emb.view(-1, emb.shape[-1]))
            generated_video = generated_video.view(*emb.shape[:2], *generated_video.shape[1:]).permute(1, 0, 2, 3, 4)
            self.writer.add_video('generated_video_true_action', torch.cat([images, generated_video, (generated_video - images + 1) / 2], dim=0), global_step=step, fps=self.config['fps'])

            # log generated video (action also generated)
            if self.config['action_mimic']:
                generation = self.predictor.generate(emb[0], emb.shape[0])
                generated_emb = generation['emb']
                generated_video = self.sensor.decode(generated_emb.view(-1, emb.shape[-1]))
                generated_video = generated_video.view(*emb.shape[:2], *generated_video.shape[1:]).permute(1, 0, 2, 3, 4)
                self.writer.add_video('generated_video_fake_action', torch.cat([images, generated_video, (generated_video - images + 1) / 2], dim=0), global_step=step, fps=self.config['fps'])

                # generate with coder prior
                emb0 = self.sensor.sample_prior(8)
                generation = self.predictor.generate(emb0, emb.shape[0])
                generated_emb = generation['emb']
                generated_video = self.sensor.decode(generated_emb.view(-1, emb.shape[-1]))
                generated_video = generated_video.view(emb.shape[0], -1, *generated_video.shape[1:]).permute(1, 0, 2, 3, 4)
                self.writer.add_video('prior_video_fake_action', generated_video, global_step=step, fps=self.config['fps'])

        self.writer.flush()

    def parse_batch(self, batch):
        if 'emb' not in batch.keys():
            images = batch['image'].to(self.device)
            B, T = images.shape[:2]
            images = images.view(B * T, *images.shape[2:])
            emb = self.sensor.encode(images)
            emb = emb.view(B, T, emb.shape[1]).permute(1, 0, 2).contiguous()
        else:
            emb = batch['emb'].permute(1, 0, 2).to(self.device).contiguous()
        action = batch['action'].permute(1, 0, 2).to(self.device).contiguous()
        reward = batch['reward'].permute(1, 0).unsqueeze(dim=-1).to(self.device).contiguous() if self.config['predict_reward'] else None  
        return emb, action, reward    

    def test_whole(self, loader):
        with torch.no_grad():
            num = 0
            info = {}
            loss = 0
            for batch in iter(loader):
                num += 1

                emb, action, reward = self.parse_batch(batch)

                _loss, _,  _info = self.predictor(emb, action, reward)

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
            "predictor_state_dict" : self.predictor.state_dict(),
            "optimizer_state_dict" : self.optim.state_dict(),
            "test_info" : test_info,
            "config" : self.config,
            "predictor_parameters" : self.config['predictor_param'],
            "seed" : self.config['seed'],
        }, filename)

    def restore(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.predictor = self.predictor.to(self.device) # make sure model on right device
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
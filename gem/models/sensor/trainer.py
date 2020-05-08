import torch
from gem.models.base import Trainer
from itertools import chain

from gem.utils import nats2bits

class VAETrainer(Trainer):
    def __init__(self, sensor, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__(train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

        self.sensor = sensor
        self.sensor = self.sensor.to(self.device)

        # config optimizer
        self.optim = torch.optim.Adam(self.sensor.parameters(), lr=self.config['m_lr'], 
                                      betas=(self.config['m_beta1'], self.config['m_beta2']))

    def parse_batch(self, batch):
        batch = batch[0].to(self.device)
        if len(batch.shape) == 5:
            batch = batch.view(-1, *batch.shape[2:])    
        return batch    

    def train_step(self):
        batch = self.parse_batch(next(self.train_iter))

        loss, info = self.sensor(batch)

        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.sensor.parameters(), self.config['grad_clip'])
        self.optim.step()
        info.update({"sensor_grad_norm" : grad_norm})

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
            self.writer.add_scalars('sensor/' + k, {'train' : nats2bits(self.last_train_info[k]), 
                                    'val' : nats2bits(val_info[k])}, global_step=step)
        self.writer.add_scalar('sensor/grad_norm', self.last_train_info['sensor_grad_norm'], global_step=step)
        
        with torch.no_grad():
            imgs = torch.clamp(self.sensor.sample(64) + 0.5, 0, 1)
            self.writer.add_images('samples', imgs, global_step=step)
            batch = self.parse_batch(next(self.train_iter))
            input_imgs = batch[:32]
            reconstructions = torch.clamp(self.sensor.decode(self.sensor.encode(input_imgs)) + 0.5, 0, 1)
            inputs_and_reconstructions = torch.stack([input_imgs, reconstructions], dim=1).view(input_imgs.shape[0] * 2, *input_imgs.shape[1:])
            self.writer.add_images('inputs_and_reconstructions', inputs_and_reconstructions, global_step=step)

        self.writer.flush()
        
    def test_whole(self, loader):
        with torch.no_grad():
            num = 0
            info = {}
            loss = 0
            for batch in iter(loader):
                num += 1
                batch = self.parse_batch(batch)
                _loss, _info = self.sensor(batch)
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
            "optimizer_state_dict" : self.optim.state_dict(),
            "test_info" : test_info,
            "config" : self.config,
            "sensor_parameters" : self.config['sensor_param'],
            "seed" : self.config['seed'],
        }, filename)

    def restore(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.sensor.load_state_dict(checkpoint['sensor_state_dict'])
        self.sensor = self.sensor.to(self.device) # make sure model on right device
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

class AVAETrainer(Trainer): # NOTE: this is broken now!
    def __init__(self, model, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__(model, train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

        # config optimizor
        self.discriminator_optim = torch.optim.Adam(model.discriminator.parameters(), 
            lr=config['lr'], betas=(config['beta1'], config['beta2']))
        self.coder_optim = torch.optim.Adam(chain(model.encoder.parameters(), model.decoder.parameters()), 
            lr=config['lr'], betas=(config['beta1'], config['beta2']))

    def parse_batch(self, batch):
        batch = batch[0].to(self.device)
        if len(batch.shape) == 5:
            batch = batch.view(-1, *batch.shape[2:])    
        return batch    

    def train_step(self):
        for i in range(self.config['n_critic']):
            real = self.parse_batch(next(self.train_iter))

            loss, mask, info = self.model(real)
            discriminator_loss = -loss

            self.discriminator_optim.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optim.step()

        # train generator only once
        real = self.parse_batch(next(self.train_iter))
        generator_loss, mask, info = self.model(real)

        self.coder_optim.zero_grad()
        generator_loss.backward()
        self.coder_optim.step()

        self.last_discriminator_loss = discriminator_loss.item()
        self.last_train_loss = generator_loss.item()
        self.last_train_info = info
        self.last_mask = mask.detach()

    def log_step(self, step):
        val_loss, val_info = self.test_whole(self.val_loader)
        print('In Step {}'.format(step))
        print('-' * 15)
        print('In training set:')
        print('NELBO is {0:{1}} bits'.format(nats2bits(self.last_train_loss), '.2f'))
        print('D loss is {0:{1}} bits'.format(nats2bits(self.last_discriminator_loss), '.2f'))
        for k in self.last_train_info.keys():
            print('{0} is {1:{2}} bits'.format(k, nats2bits(self.last_train_info[k]), '.2f'))
        print('In validation set:')
        print('NELBO is {0:{1}} bits'.format(nats2bits(val_loss), '.2f'))
        for k in val_info.keys():
            print('{0} is {1:{2}} bits'.format(k, nats2bits(val_info[k]), '.2f'))

        self.writer.add_scalars('NELBO', {'train' : nats2bits(self.last_train_loss), 
                                        'val' : nats2bits(val_loss)}, global_step=step)
        for k in self.last_train_info.keys():
            self.writer.add_scalars(k, {'train' : nats2bits(self.last_train_info[k]), 
                                    'val' : nats2bits(val_info[k])}, global_step=step)
        
        with torch.no_grad():
            imgs = torch.clamp(self.model.sample(64), 0, 1)
            self.writer.add_images('samples', imgs, global_step=step)
            batch = self.parse_batch(next(self.train_iter))
            input_imgs = batch[:32]
            reconstructions = torch.clamp(self.model.decode(self.model.encode(input_imgs)), 0, 1)
            inputs_and_reconstructions = torch.stack([input_imgs, reconstructions], dim=1).view(input_imgs.shape[0] * 2, *input_imgs.shape[1:])
            self.writer.add_images('inputs_and_reconstructions', inputs_and_reconstructions, global_step=step)
            mask_max = torch.max(torch.max(self.last_mask, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            self.writer.add_images('mask', self.last_mask / mask_max, global_step=step)

    def test_whole(self, loader):
        with torch.no_grad():
            num = 0
            info = {}
            loss = 0
            for batch in iter(loader):
                num += 1
                batch = self.parse_batch(next(self.train_iter))
                _loss, _mask, _info = self.model(batch)
                loss += _loss
                for k in _info.keys():
                    info[k] = info.get(k, 0) + _info[k]
            loss = loss / num
            for k in info.keys():
                info[k] = info[k] / num

        return loss.item(), info

    def save(self, filename):
        torch.save({
            "model_state_dict" : self.model.state_dict(),
            "discriminator_optimizer_state_dict" : self.discriminator_optim.state_dict(),
            "coder_optimizer_state_dict" : self.coder_optim.state_dict(),
            "config" : self.config,
            "model_parameters" : self.config['model_param'],
            "seed" : self.config['seed'],
        }, filename)

    def restore(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device) # make sure model on right device
        self.discriminator_optim.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.coder_optim.load_state_dict(checkpoint['coder_optimizer_state_dict'])


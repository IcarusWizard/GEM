import torch
from degmo.base import Trainer
from itertools import chain

from degmo.utils import nats2bits

class AVAETrainer(Trainer):
    def __init__(self, model, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__(model, train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

        # config optimizor
        self.discriminator_optim = torch.optim.Adam(model.discriminator.parameters(), 
            lr=config['lr'], betas=(config['beta1'], config['beta2']))
        self.coder_optim = torch.optim.Adam(chain(model.encoder.parameters(), model.decoder.parameters()), 
            lr=config['lr'], betas=(config['beta1'], config['beta2']))
    
    def train_step(self):
        for i in range(self.config['n_critic']):
            real = next(self.train_iter)[0].to(self.device)

            discriminator_loss = self.model.get_discriminator_loss(real)

            self.discriminator_optim.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optim.step()

        # train generator only once
        real = next(self.train_iter)[0].to(self.device)
        generator_loss, info = self.model(real)

        self.coder_optim.zero_grad()
        generator_loss.backward()
        self.coder_optim.step()

        self.last_discriminator_loss = discriminator_loss.item()
        self.last_train_loss = generator_loss.item()
        self.last_train_info = info

    def log_step(self, step):
        val_loss, val_info = self.test_whole(self.val_loader)
        print('In Step {}'.format(step))
        print('-' * 15)
        print('In training set:')
        print('NELBO is {0:{1}} bits'.format(nats2bits(self.last_train_loss), '.5f'))
        print('D loss is {0:{1}} bits'.format(nats2bits(self.last_discriminator_loss), '.5f'))
        for k in self.last_train_info.keys():
            print('{0} is {1:{2}} bits'.format(k, nats2bits(self.last_train_info[k]), '.5f'))
        print('In validation set:')
        print('NELBO is {0:{1}} bits'.format(nats2bits(val_loss), '.5f'))
        for k in val_info.keys():
            print('{0} is {1:{2}} bits'.format(k, nats2bits(val_info[k]), '.5f'))

        self.writer.add_scalars('NELBO', {'train' : nats2bits(self.last_train_loss), 
                                        'val' : nats2bits(val_loss)}, global_step=step)
        for k in self.last_train_info.keys():
            self.writer.add_scalars(k, {'train' : nats2bits(self.last_train_info[k]), 
                                    'val' : nats2bits(val_info[k])}, global_step=step)
        
        with torch.no_grad():
            imgs = torch.clamp(self.model.sample(64, deterministic=True), 0, 1)
            self.writer.add_images('samples', imgs, global_step=step)
            input_imgs = batch = next(self.train_iter)[0].to(self.device)[:32]
            reconstructions = torch.clamp(self.model.decode(self.model.encode(input_imgs)), 0, 1)
            inputs_and_reconstructions = torch.stack([input_imgs, reconstructions], dim=1).view(input_imgs.shape[0] * 2, *input_imgs.shape[1:])
            self.writer.add_images('inputs_and_reconstructions', inputs_and_reconstructions, global_step=step)

    def test_whole(self, loader):
        with torch.no_grad():
            num = 0
            info = {}
            loss = 0
            for batch in iter(loader):
                num += 1
                batch = batch[0].to(self.device)
                _loss, _info = self.model(batch)
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

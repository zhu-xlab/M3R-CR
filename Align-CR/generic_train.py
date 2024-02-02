import os
import time
import torch
import numpy as np

class Generic_Train():
	def __init__(self, model, opts, train_dataloader, val_dataloader):
		self.model=model
		self.opts=opts
		self.train_dataloader=train_dataloader
		self.val_dataloader=val_dataloader

	def train(self):
		
		total_steps = 0
		log_loss = 0
		best_score = 0

		for epoch in range(self.opts.max_epochs):
			for data in self.train_dataloader:
				total_steps+=1

				self.model.set_input(data)
				batch_loss = self.model.optimize_parameters()
				log_loss = log_loss + batch_loss

				if total_steps % self.opts.log_iter == 0:
					avg_log_loss = log_loss/self.opts.log_iter
					print('epoch', epoch, 'steps', total_steps, 'loss', avg_log_loss)
					log_loss = 0
					
			if (epoch+1) % self.opts.val_freq == 0:
				print("validation...")
				self.model.net_G.eval()
				with torch.no_grad():
					_iter = 0
					score = 0
					for data in self.val_dataloader:
						self.model.set_input(data)
						score += self.model.val_scores()['PSNR']
						_iter += 1
					score = score/_iter
				print(f'PSNR: {score}')
				if score > best_score:  # save best model
					best_score = score
					self.model.save_checkpoint('best')
				self.model.net_G.train()
			
			if epoch >= self.opts.lr_start_epoch_decay - self.opts.lr_step:
				self.model.update_lr()
			
			if epoch % self.opts.save_freq == 0:
				self.model.save_checkpoint(epoch)




import torch
import tensorflow as tf
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)
  
  class Reshape(torch.nn.Module):
      def __init__(self, C, H, W):
          super(Reshape, self).__init__()
          self.C = C
          self.H = H
          self.W = W
      def forward(self, x):
          return x.reshape((x.shape[0], self.C, self.H, self.W))
  class Flatten(torch.nn.Module):
      def __init__(self):
          super(Flatten, self).__init__()
      def forward(self, x):
          return x.reshape((x.shape[0], -1))
      
  class Merge(torch.nn.Module):
      def __init__(self, net):
          super(Merge, self).__init__()
          self.net = net
      def forward(self, x, t):
          x = Reshape(3, 32, 32)(x)
          out = self.net(x, t)
          out = Flatten()(out)
          return out
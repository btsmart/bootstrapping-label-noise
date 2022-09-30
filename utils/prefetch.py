# coding=utf-8
"""
Ref:
    https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256
"""
import torch


class data_prefetcher():
    def __init__(self, loader, device):
        self.device = device
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(self.device)
        self.preload()

    def preload(self):
        try:
            self.next_idxs, self.next_input, self.next_target, self.next_l_labels, self.next_n_labels = next(self.loader)
            
        except StopIteration:
            self.next_idxs = None
            self.next_input = None
            self.next_target = None
            self.next_l_labels = None
            self.next_n_labels = None
            return
        with torch.cuda.stream(self.stream):
            self.next_idxs = self.next_idxs.to(self.device, non_blocking=True)
            if isinstance(self.next_input, list):
                for i in range(len(self.next_input)):
                    self.next_input[i] = self.next_input[i].to(self.device, non_blocking=True)
            else:
                self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)
            self.next_l_labels = self.next_l_labels.to(self.device, non_blocking=True)
            self.next_n_labels = self.next_n_labels.to(self.device, non_blocking=True)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_idxs = self.next_idxs
        input = self.next_input
        target = self.next_target
        l_labels = self.next_l_labels
        n_labels = self.next_n_labels
        self.preload()
        return next_idxs, input, target, l_labels, n_labels

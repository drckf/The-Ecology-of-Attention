import torch
import torch.optim as optim


class ExponentiatedSGD(optim.Optimizer):
    """
    Implements Exponentiated Gradient Descent.
    
    This optimizer updates weights using the rule:
    w_l' = w_l * exp(-lr * grad_l)
    
    where:
    - w_l is the weight
    - lr is the learning rate
    - grad_l is the gradient of the cost function with respect to w_l
    """
    
    def __init__(self, params, lr=0.01, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ExponentiatedSGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Exponentiated update: w_l' = w_l * exp(-lr * grad_l)
                p.data.mul_(torch.exp(-group['lr'] * grad))
                
        return loss
    
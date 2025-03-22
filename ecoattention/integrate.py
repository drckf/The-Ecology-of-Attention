import torch


def integrate_linear(
        w_0: torch.Tensor,
        s: torch.Tensor,
        A: torch.Tensor,
        t_max: float,
        dt: float = 0.01,
        callback: callable = None,
    ) -> torch.Tensor:
    """
    Integrate a linear system dw/dt = s - A @ w

    Args:
        w_0 (torch.Tensor ~ (L,)): initial state
        s (torch.Tensor ~ (L,)): source term
        A (torch.Tensor ~ (L,L)): linear operator
        t_max (float): maximum time
        dt (float): time step
        callback (callable, optional): function to call after each step

    Returns:
        x (torch.Tensor ~ (L,)): final state
    """
    w = w_0
    t = 0
    while t < t_max:
        dw = s - A @ w
        w = w + dt * dw
        t += dt
        if callback is not None:
            callback(w, t)
    return w


def integrate_lotka_volterra(
        w_0: torch.Tensor,
        s: torch.Tensor,
        A: torch.Tensor,
        t_max: float,
        dt: float = 0.01,
        callback: callable = None,
    ) -> torch.Tensor:
    """
    Integrate a Lotka-Volterra system dw/dt = (s - A @ w) * w

    Args:
        w_0 (torch.Tensor ~ (L,)): initial state  
        s (torch.Tensor ~ (L,)): source term
        A (torch.Tensor ~ (L,L)): linear operator
        t_max (float): maximum time
        dt (float): time step
        callback (callable, optional): function to call after each step

    Returns:
        w (torch.Tensor ~ (L,)): final state
    """
    w = w_0
    t = 0
    while t < t_max:
        dw = (s - A @ w) * w
        w = (w + dt * dw).clamp(min=0)
        t += dt
        if callback is not None:
            callback(w, t)
    return w


def integrate_replicator_equation(
        w_0: torch.Tensor,
        s: torch.Tensor,
        A: torch.Tensor,
        t_max: float,
        dt: float = 0.01,
        callback: callable = None,
    ) -> torch.Tensor:
    """
    Integrate a replicator equation dw/dt = (f(w) - f(w) @ w) * w
    where the fitness function is f(w) = s - A @ w

    Args:
        w_0 (torch.Tensor ~ (L,)): initial state
        s (torch.Tensor ~ (L,)): source term
        A (torch.Tensor ~ (L,L)): linear operator
        t_max (float): maximum time
        dt (float): time step
        callback (callable, optional): function to call after each step

    Returns:
        w (torch.Tensor ~ (L,)): final state  
    """
    w = w_0
    t = 0
    while t < t_max:
        f = s - A @ w
        f_bar = f @ w
        dw = (f - f_bar) * w
        w = (w + dt * dw).clamp(min=0)
        w = w / w.sum(dim=-1, keepdim=True)
        t += dt
        if callback is not None:
            callback(w, t)
    return w

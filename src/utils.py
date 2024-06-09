import numpy as np
import threading
import asyncio

def set_interval(func, sec):
    def func_wrapper(stop_event):
        if not stop_event.is_set():
            func()
            threading.Timer(sec, func_wrapper, args=[stop_event]).start()
    stop_event = threading.Event()
    t = threading.Timer(sec, func_wrapper, args=[stop_event])
    t.start()
    t.cancel = stop_event.set
    return t

def set_timeout(func, sec):
    t = threading.Timer(sec, func)
    t.start()
    return t

async def wait_until_true(async_func, interval=0.1):
    while True:
        if await async_func():
            return
        await asyncio.sleep(interval)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    sum_exp_x = np.sum(exp_x)
    return exp_x / (sum_exp_x + 1e-9)

def sample_from_prob(prob):
    prob = prob.squeeze()
    idx = np.random.choice(len(prob), p=prob)
    one_hot = [0 if i != idx else 1 for i in range(len(prob))]
    return one_hot

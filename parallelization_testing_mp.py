import torch

global_lock = 1


class Net1(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = torch.nn.Linear(10,5, device='cuda:1')
        self.w = torch.rand((10), device='cuda:1')
        # self.w2 = torch.rand((1), device='cuda:1')

    def forward(self,x):
        # self.w2 = self.l(x)
        # print(self.w2.is_leaf)
        return self.l(x)
        # return self.l(x)
    
    def method(self, queue):
        z = (queue.get())
        print(z)
        return z
    

class Net2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = torch.nn.Linear(10,1, device='cuda:1')
        self.w = torch.rand((10), device='cuda:1')
        # self.w2 = torch.rand((1), device='cuda:1')

    def forward(self,x):
        self.i = x
        return self.l(self.i)
    
    def method(self, queue):
        z = (queue.get())
        print(z)
        return z

    

def _process1(net, queue, x):
    # global global_lock
    # x = queue.get()
    y = net(x)
    print('y_1: ', y)
    queue.put(y.clone().detach())
    # global_lock = 0
    # net.method(queue)
    # print('get')
    # queue.put(0)
    # print('put')

    # start backward pass! 
    grad_in = queue.get()
    # loss = torch.sum((y - ))
    # loss.backward()
    # print('p1 back done')

def _process2(net, queue):
    # print('global_locki ', global_lock)
    # while global_lock != 0:
    #     continue
    # print('global_lock ', global_lock)
    x = queue.get()
    y = net(x)
    # y.backward()
    print('y_2: ', y)

    loss = torch.sum((y - torch.ones(y.shape, device='cuda:1')))
    loss.backward()
    # TODO get the gradient of the top of this network?
    grad_in = something
    queue.put(something)


if __name__ == '__main__':
    # l = torch.nn.Linear(10,1, device='cuda:1')
    net1 = Net1().to('cuda:1')
    net2 = Net2().to('cuda:1')
    torch.multiprocessing.set_start_method('spawn')
    input_ = torch.ones((1,10)).to('cuda:1')
    queue = torch.multiprocessing.Queue()
    # out = l(input_)
    # out.backward()
    # out = net(input_)
    # out.backward()

    queue.put(input_)
    process1 = torch.multiprocessing.Process(target=_process1, args=(net1, queue, input_))
    process2 = torch.multiprocessing.Process(target=_process2, args=(net2, queue,))
    process1.start()
    process2.start()
    # queue.put(input_)
    process1.join()
    process2.join()

    # result = queue.get()
    # print('end')
    # print(result)
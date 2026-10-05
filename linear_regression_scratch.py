import torch 
from d2l import torch as d2l 

class LinearRegressionScratch(d2l.Module):
    def __init__(self , num_inputs , lr , sigma=0.01): #luu lai learning rate
        super().__init__()  
        self.save_hyperparameters() 
        self.w = torch.normal(0 , sigma , (num_inputs , 1) , requires_grad=True) #khoi tao w ngau nhien theo phan phoi chuan , bat tinh gradient
        self.b = torch.zeros(1 , requires_grad=True) #khoi tao 1 phan tu cua b , tinh gradient 


@d2l.add_to_class(LinearRegressionScratch)
def forward(self , X): 
    return torch.matmul(X , self.w) + self.b #khoi tao y = X * w + b

@d2l.add_to_class(LinearRegressionScratch)
def loss(self , y_hat , y) : 
    l = (1/2)*(y_hat - y) ** 2  
    return l.mean() #loss = trung binh cua 1/2 * (y_hat - y) ** 2 tren ca batch

class SGD(d2l.HyperParameters):
    def __init__(self , params , lr): 
        self.save_hyperparameters() 
    def step(self): 
        #cap nhap tung tham so cho self params 
        for p in self.params:
            p -= self.lr * p.grad #cap nhat tai cho (in-place): p = p - lr * grad
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()  

@d2l.add_to_class(LinearRegressionScratch)
def configure_optimizers(self):
    return SGD([self.w , self.b] , self.lr)

@d2l.add_to_class(d2l.Trainer)
def prepare_batch(self , batch):  #tra ve batch nguyen ven (X, y) de dua vao model
    return batch  

@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    self.model.train()                                     # chuyen model sang che do huan luyen
    for batch in self.train_dataloader:                    # duyet tung mini-batch trong tap train
        loss = self.model.training_step(self.prepare_batch(batch))  # forward + tinh loss cho batch
        self.optim.zero_grad()                             # xoa gradient cu (tranh bi cong don)
        with torch.no_grad():                              # khong ghi lai do thi tinh toan khi cap nhat
            loss.backward()                                # lan truyen nguoc, tinh gradient cho w, b
            if self.gradient_clip_val > 0:                 # neu co bat gradient clipping
                self.clip_gradients(self.gradient_clip_val, self.model)  # cat gradient tranh bung no
            self.optim.step()                              # cap nhat tham so: p = p - lr * grad
        self.train_batch_idx += 1                          # tang bo dem so batch da train
    if self.val_dataloader is None:                        # neu khong co tap validation
        return                                             # thi ket thuc epoch luon
    self.model.eval()                                      # chuyen model sang che do danh gia
    for batch in self.val_dataloader:                      # duyet tung mini-batch trong tap validation
        with torch.no_grad():                              # khong tinh gradient khi danh gia
            self.model.validation_step(self.prepare_batch(batch))  # tinh va ve loss tren tap val
        self.val_batch_idx += 1                            # tang bo dem so batch da validate


model = LinearRegressionScratch(num_inputs=2, lr=0.03) #so input = so dac trung cua du lieu (len(w) = 2)

data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)


with torch.no_grad():
    print(f'Sai số khi ước lượng w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'Sai số khi ước lượng b: {data.b - model.b}')

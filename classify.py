import os
import torch
import torchvision
import time
from torch import optim, nn
from config import params
from data_process import SkeletonFeeder
from torch.utils.data import DataLoader
from model import Classification
from tensorboardX import SummaryWriter
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, way, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
  
  
def train(model, train_loader, optimizer, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    model.train()
    for step, (input, label) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)
        input = input.cuda(params['gpu'][0]) #to do--->
        label = label.cuda(params['gpu'][0]) #to do---> 
        output_classify = model(input)

        prec1, prec5 = accuracy(output_classify.data, label, 1, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
              
        loss = criterion(output_classify, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item())
    return losses.avg, top1.avg, top5.avg, batch_time, data_time
  
  
def valid(model, train_loader, optimizer, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, label) in enumerate(tqdm(train_loader)):
            input = input.cuda(params['gpu'][0])
            label = label.cuda(params['gpu'][0])

            output_classify = model(input)
            prec1, prec5 = accuracy(output_classify.data, label, 1, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            loss = criterion(output_classify, label)
            losses.update(loss.item())
    return losses.avg, top1.avg, top5.avg
  
def main():
    #define the dataloader
    train_loader = DataLoader(SkeletonFeeder(mode='train', debug=False), batch_size=params['batchsize'], shuffle=True, num_workers=params['numworkers'])
    val_loader = DataLoader(SkeletonFeeder(mode='valid', debug=False), batch_size=params['batchsize'], shuffle=False, num_workers=params['numworkers'])
    n_class = params['n_class']
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    #use the model and transfer to gpu
    model = Classification(n_class=n_class) #add model cfg 
    model = model.cuda(params['gpu'][0])#to do--->
    model = nn.DataParallel(model, device_ids=params['gpu'])

    
    if params['retrain']:
        trained_dict = torch.load(params['retrain'],map_location='cpu')
        model_dict = model.state_dict()
        trained_dict = {k:v for k,v in trained_dict.items() if k in model_dict}      
        model_dict.update(trained_dict)
        model.load_state_dict(model_dict)
        print('load trained model finish')
        

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    
    
    optimizer = optim.Adam(model_params, lr=params['lr'], weight_decay=params['weight_decay'])
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.333, patience=2, verbose=True)
    writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()
    min_loss = 1000
    
    print('-------------------start training----------------------')
    print('lr:', optimizer.param_groups[0]['lr'])
    for i in range(params['epoch']):
        train_loss, train_top1, train_top5, batch_time, data_time = train(model, train_loader, optimizer,criterion)
        valid_loss, val_top1, val_top5 = valid(model, val_loader,optimizer,criterion)
        schedule.step(valid_loss)

        f = open(params['log']+'bert_classifylog_'+cur_time+'.txt', 'a')
        print('epoch:', str(i + 1) + "/" + str(params['epoch']))
        print('data time:%0.3f'%data_time.avg, 'batch time:%0.3f'%batch_time.avg, 'epoch time:%0.3f'%(batch_time.sum))
        print('train loss:%0.8f'%train_loss, 'top1:%0.2f'%train_top1, '%', 'top5:%0.2f'%train_top5, '%', 'lr:', optimizer.param_groups[0]['lr'])
        print('valid loss:%0.8f'%valid_loss, 'top1:%0.2f'%val_top1, '%', 'top5:%0.2f'%val_top5, '%')
        f.write('epoch:'+str(i+1)+"/"+str(params['epoch'])+'\n')
        f.write('data time:%0.3f'%data_time.avg+'batch time:%0.3f'%batch_time.avg+'epoch time:%0.3f'%(batch_time.sum)+'\n')
        f.write('train loss:%0.8f'%train_loss+'top1:%0.2f'%train_top1+'%'+'top5:%0.2f'%train_top5+'%'+'lr:'+str(optimizer.param_groups[0]['lr'])+'\n')
        f.write('valid loss:%0.8f'%valid_loss+'top1:%0.2f'%val_top1+'%'+'top5:%0.2f'%val_top5+'%'+'\n')
        f.write('************************************\n')
        f.close()
        
        writer.add_scalar('train loss', train_loss, i)
        writer.add_scalar('valid loss', valid_loss, i) 
        writer.add_scalar('train top1', train_top1, i)
        writer.add_scalar('valid top1', val_top1, i)
        writer.add_scalar('train top5', train_top5, i)
        writer.add_scalar('valid top5', val_top5, i)
        
        if valid_loss < min_loss:
            torch.save(model.state_dict(), params['save_path']+'bert_classifymodel_'+cur_time+'.pth')
            print('saving model successful to --->',params['save_path'])
            min_loss = valid_loss

    writer.close()
    
if __name__ == '__main__':
    main()
    
    

params = {}

params['root'] = '/data/nturgbd-skeleton/xsub' # to change according to your own data path
params['batchsize'] = 32
params['numworkers'] = 8
params['gpu'] =[0,1] # to change according to your own gpu settings
params['retrain'] = False #'./checkpoint/bert_classifymodel_xxxx.pth'

params['lr'] = 1e-3 
params['weight_decay'] = 1e-6
params['epoch'] = 100
params['save_path'] = './checkpoint/'
params['log'] = './log/'
params['n_class'] = 60

#To load pretrained GCN model
params['load_stgcn'] = False

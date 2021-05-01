params = {}

params['root'] = '/data/nturgbd-skeleton/xsub' # to change according to your own data repo
params['batchsize'] = 32
params['numworkers'] = 8
params['gpu'] =[0,1] # to change according to your own gpu settings
params['pretrain'] = False #'./checkpoint/bert_model_2020-06-18-09-09-59.pth' 
params['retrain'] = False #'./checkpoint/bert_classifymodel_2020-06-24-22-54-42.pth'#'./checkpoint/bert_classifymodel_2020-06-17-14-44-48.pth'#'./checkpoint/bert_model_2020-06-06-17-05-13.pth'

params['lr'] = 1e-3 
params['weight_decay'] = 1e-6
params['epoch'] = 100
params['save_path'] = './checkpoint/'
params['log'] = './log/'
params['n_class'] = 60

#To load pretrained GCN model
params['load_stgcn'] = False

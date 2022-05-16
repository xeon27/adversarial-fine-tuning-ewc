from torch import optim
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
#from visdom import Visdom
import utils
from utils import adv_accuracy
from adversarial import AttackBase, FGSM, LinfPGD
from optimizer import Weight_Regularized_AdamW


def train(model, train_datasets, test_datasets, epochs_per_task=10,
          batch_size=64, test_size=1024, consolidate=True,
          fisher_estimation_sample_size=1024,
          lr=1e-3, weight_decay=1e-5,
          loss_log_interval=30,
          eval_log_interval=50,
          cuda=False, adv_train = None, args = None):
    
    # prepare the loss criteriton and the optimizer.
    criteriton = nn.CrossEntropyLoss()
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr,
                            weight_decay=weight_decay)

        if args.with_scheduling: 
            lambda1 = lambda epoch: epoch if epoch < 6 else 5*(0.5**(epoch-5))        
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
         
    elif args.optim == "adam":
        print("Using Adam")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        if args.with_scheduling:
            lambda1 = lambda epoch: epoch if epoch < 6 else 5*(0.5**(epoch-5))        
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    elif args.optim == "adamw":
        print("Using AdamW")
        optimizer = Weight_Regularized_AdamW( model.parameters(), lr=lr)
         
        if args.with_scheduling:
            lambda1 = lambda epoch: epoch if epoch < 6 else 5*(0.5**(epoch-5))        
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
   
        optimizer.set_model(model)

 

    # set the model's mode to training mode and attack type
    adv, stats = adv_train_module("pgd", model, "cifar", args.iters, cuda, args.alpha, args.repeat)
    model.train()

    for task, train_dataset in enumerate(train_datasets, 1):
       
        fisher_estimte = True
        if consolidate and fisher_estimte:
            fisher_estimte = False

            # estimate the fisher information of the parameters and consolidate
            # them in the network.
            print(
                '=> Estimating diagonals of the fisher information matrix...',
            flush=True, end='',
            )
            model.consolidate(model.estimate_fisher(
                train_dataset, 16, 2
            ))
            print(' Done!')
            
            if args.optim == "adamw":
                model.load_reg_params(args.lamda)

        for epoch in range(1, epochs_per_task+1):
            # prepare the data loaders.
            data_loader = utils.get_data_loader(
                train_dataset, batch_size=batch_size,
                cuda=cuda
            )

            if epoch == 1:
                print("Test Adv Accuracy : ", adv_accuracy(model, test_datasets[0], cuda = cuda , batch_size = args.adv_test_size))
            
            model.train()
            if args.with_scheduling:
                scheduler.step()
                print('learning rate : ', optimizer.param_groups[0]['lr'])
            print("Epoch : ", epoch)
            

            data_stream = tqdm(enumerate(data_loader, 1))
            for batch_index, (x, y) in data_stream:
                data_size = len(x)
                dataset_size = len(data_loader.dataset)
                dataset_batches = len(data_loader)
                previous_task_iteration = sum([
                    epochs_per_task * len(d) // batch_size for d in
                    train_datasets[:task-1]
                ])
                current_task_iteration = (
                    (epoch-1)*dataset_batches + batch_index
                )
                iteration = (
                    previous_task_iteration +
                    current_task_iteration
                )

                x = Variable(x).cuda() if cuda else Variable(x)         
                y = Variable(y).cuda() if cuda else Variable(y)
                x = adv.perturb(x,y)

                # run the model and backpropagate the errors.
                optimizer.zero_grad()
                scores = model(x)
                ce_loss = criteriton(scores, y)
                if consolidate and args.optim != "adamw":
                    ewc_loss = model.ewc_loss(lamda=args.lamda, cuda=cuda)
                else:
                    ewc_loss = 0
                loss = ce_loss + ewc_loss
                loss.backward()

                if args.optim == "adamw":
                    optimizer.step(lamda = args.lamda)
                else:
                    optimizer.step()

                # calculate the training precision.
                _, predicted = scores.max(1)
                precision = (predicted == y).sum().float() / len(x)

                data_stream.set_description((
                    '=> '
                    'task: {task}/{tasks} | '
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'prec: {prec:.4} | '
                    'loss => '
                    'ce: {ce_loss:.4} / '
                    'ewc: {ewc_loss:.4} / '
                    'total: {loss:.4}'
                ).format(
                    task=task,
                    tasks=len(train_datasets),
                    epoch=epoch,
                    epochs=epochs_per_task,
                    trained=batch_index*batch_size,
                    total=dataset_size,
                    progress=(100.*batch_index/dataset_batches),
                    prec=float(precision),
                    ce_loss=float(ce_loss),
                    ewc_loss=float(ewc_loss),
                    loss=float(loss),
                ))

            precs = [
                utils.validate(
                    model, test_datasets[i], test_size=-1,
                    cuda=cuda, verbose=True,
                ) if i+1 <= task else 0 for i in range(len(train_datasets))
            ]

            print("Test Adv Accuracy : ", adv_accuracy(model, test_datasets[0], cuda = cuda , batch_size = args.adv_test_size))

                

          
def adv_train_module(attack, model, data_type, iters, device, alpha=None, repeat=None) :
    norm = True if data_type != "mnist" else False
    bound = 0.3 if data_type == "mnist" else 8/255
    step = alpha or 2/255
    print("Alpha here :", step)
    random_start=False if attack != "pgd" else True
    if attack == "pgd" :
        random_start = True
    else :
        random_start = False
    repeat_num = repeat or 5
    stats = (repeat_num,)

    if attack == None :
        adv = AttackBase(norm=norm, device=device)
    elif attack == "pgd" :
        adv = LinfPGD(model, bound=bound, step=step, iters=iters, norm=norm, random_start=random_start, device=device)
    elif attack == "fgsm" :
        adv = FGSM(model, bound=bound, norm=norm, random_start=random_start, device=device)
    return adv, stats





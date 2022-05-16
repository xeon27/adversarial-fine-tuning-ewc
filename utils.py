import os
import os.path
import shutil
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn import init
import torchattacks
from torchattacks import PGD, FGSM



def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None):

    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=(collate_fn or default_collate),
        **({'num_workers': 2, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir, epoch, precision, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'state': model.state_dict(),
        'epoch': epoch,
        'precision': precision,
    }, path)

    # override the best model if it's the best.
    if best:
        shutil.copy(path, path_best)
        print('=> updated the best model of {name} at {path}'.format(
            name=model.name, path=path_best
        ))

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # load the checkpoint.
    checkpoint = torch.load(path_best if best else path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path_best if best else path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    precision = checkpoint['precision']
    return epoch, precision

def adv_accuracy(model, dataset_loader, batch_size = 128,cuda = False, criterion=None) :

    model.train()
    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
    total = 0
    correct = 0
    criterion = criterion or torch.nn.CrossEntropyLoss().cuda()
    data_loader = get_data_loader(dataset_loader, batch_size, cuda=cuda)
    for i, (x,y) in enumerate(data_loader) :
            x = atk(x, y)
            y = y.cuda()
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)

            total += y.size(0)
            correct += (predicted == y.cuda()).sum()

    accuracy =100 * float(correct) / total
    return accuracy


def validate(model, dataset, test_size=256, batch_size=32,
             cuda=False, verbose=True):
    mode = model.training
    model.train(mode=False)
    data_loader = get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = 0
    total_correct = 0
    for x, y in data_loader:
        # break on test size.
        if (test_size > 0) and (total_tested >= test_size):
            break
        # test the model.
        x = Variable(x).cuda() if cuda else Variable(x)
        y = Variable(y).cuda() if cuda else Variable(y)
        scores = model(x)
        _, predicted = scores.max(1)
        # update statistics.
        total_correct += int((predicted == y).sum())
        total_tested += len(x)
    print(f"Total test size: {total_tested}")
    print(len(data_loader.dataset))
    model.train(mode=mode)
    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal(p)


def gaussian_intiailize(model, std=.1):
    for p in model.parameters():
        init.normal(p, std=std)

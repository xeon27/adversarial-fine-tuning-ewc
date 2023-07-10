## Adversarial fine-tuning using Elastic Weight Consolidation (EWC)

While the success in Deep Learning has led to many breakthroughs in a variety of domains, Neural Networks have also been shown to be vulnerable to small perturbations to the input data. These perturbations also known as adversarial attacks and create a security threat in critical applications.  

Fine-tuning allows us to improve adversarial of any pre-trained model by saving significant computational time as compared to training the model from scratch with adversarial examples. However, vanilla fine-tuning with a constant learning rate still suffers from forgetting issues as the new learning updates the parameters of the model, leading to drop in performance on the test set. In this study, we attempt to address the overfitting issue in adversarial fine-tuning through the lens of elastic weight consolidation.

 
## Installation
```
$ pip install -r requirements.txt
```


## CLI
Implementation CLI is provided by `main.py`

#### Train
```
$ python -m visdom.server &
$ ./main.py               # Train the network without consolidation.
$ ./main.py --consolidate # Train the network with consolidation.
```

## References

In this project, we refer the code repositories of existing work in ewc and pgd training. We would like to thank the authors of the following code repositories forsharing their code online for public use.

PGD Code : https://github.com/matbambbang/pgd_adversarial_training
EWC Code : https://github.com/kuc2477/pytorch-ewc
Pre-trained CIFAR 10/100 Models : https://github.com/meliketoy/wide-resnet.pytorch
ADAMW Optimizer : https://github.com/INK-USC/CLIF



## Author
Diljot Singh, 
Omkar Dige, 
Pranav Gupta 

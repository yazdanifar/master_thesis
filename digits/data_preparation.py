from torchvision import transforms
from torch.utils.data import DataLoader
from data_load import mnist, svhn, usps


def digit_load(args):
    train_bs = args.batch_size
    train_datasets = {'s': svhn.SVHN_idx('./data/svhn/', split='train', download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(32),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])),
                      'u': usps.USPS_idx('./data/usps/', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(32),
                                             transforms.Lambda(lambda x: x.convert("RGB")),
                                             transforms.RandomCrop(32),
                                             transforms.RandomRotation(10),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])),
                      'm': mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(32),
                                           transforms.Lambda(lambda x: x.convert("RGB")),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))
                      }
    shuffled_train_loaders = {k: DataLoader(v, batch_size=train_bs, shuffle=True,
                                            num_workers=args.worker, drop_last=False) for k, v in
                              train_datasets.items()}
    un_shuffled_train_loaders = {k: DataLoader(v, batch_size=train_bs, shuffle=False,
                                               num_workers=args.worker, drop_last=False) for k, v in
                                 train_datasets.items()}
    test_loaders = {'s': DataLoader(svhn.SVHN('./data/svhn/', split='test', download=True,
                                              transform=transforms.Compose([
                                                  transforms.Resize(32),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ])), batch_size=train_bs * 2, shuffle=False,
                                    num_workers=args.worker, drop_last=False),
                    'u': DataLoader(usps.USPS('./data/usps/', train=False, download=True,
                                              transform=transforms.Compose([
                                                  transforms.Resize(32),
                                                  transforms.Lambda(lambda x: x.convert("RGB")),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ])), batch_size=train_bs * 2, shuffle=False,
                                    num_workers=args.worker, drop_last=False),
                    'm': DataLoader(mnist.MNIST('./data/mnist/', train=False, download=True,
                                                transform=transforms.Compose([
                                                    transforms.Resize(32),
                                                    transforms.Lambda(lambda x: x.convert("RGB")),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])), batch_size=train_bs * 2, shuffle=False,
                                    num_workers=args.worker, drop_last=False)}

    return shuffled_train_loaders, un_shuffled_train_loaders, test_loaders

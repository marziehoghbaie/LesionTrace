from torch.utils.data import DataLoader
import torchvision, torch
from torchvision.transforms import v2

def OLIVES_Bscans_loader(dataset_info, model_config, train_config, logger, where):
        train_loader = None
        val_loader = None
        test_loader = None
        transformation_train = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transformation_test = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # load test set
        if not train_config['train']:
            logger.info('[INFO] load test set from {}...'.format(dataset_info['data_path']))
            test_set = torchvision.datasets.ImageFolder(dataset_info['data_path']+'/test', transform=transformation_test)
            test_loader = DataLoader(test_set, batch_size=train_config['batch_size'],
                                     shuffle=dataset_info['shuffle'],
                                     num_workers=dataset_info['num_workers'])
        else:
            logger.info('[INFO] load train set from {}...'.format(dataset_info['data_path']))
            train_set = torchvision.datasets.ImageFolder(dataset_info['data_path'] + '/train',
                                                         transform=transformation_train)
            val_set = torchvision.datasets.ImageFolder(dataset_info['data_path'] + '/val',
                                                       transform=transformation_test)

            train_loader = DataLoader(train_set,
                                      batch_size=train_config['batch_size'],
                                      shuffle=dataset_info['shuffle'],
                                      num_workers=dataset_info['num_workers'], drop_last=True)
            val_loader = DataLoader(val_set,
                                    batch_size=train_config['batch_size'],
                                    shuffle=dataset_info['shuffle'],
                                    num_workers=dataset_info['num_workers'], drop_last=True)

        return train_loader, val_loader, test_loader

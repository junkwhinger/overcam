import os, time, copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as vmodels
import torchvision.datasets as vdatasets

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 64

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(197),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize((197, 197)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


data_dir = "/input"
image_datasets = {x: vdatasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'validation']}

dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
               'validation': torch.utils.data.DataLoader(image_datasets['validation'], BATCH_SIZE, shuffle=False)}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
nb_classes = len(image_datasets['train'].classes)


class Resnet_fc(nn.Module):
    def __init__(self, base_model, nb_classes):
        super(Resnet_fc, self).__init__()

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])

        tmp = OrderedDict()
        tmp['last_conv'] = nn.Conv2d(2048, nb_classes, 1, 1)
        tmp['gap'] = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.classifier_layer = nn.Sequential(tmp)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.classifier_layer(x)
        return F.sigmoid(x)


base_model = vmodels.resnet50(pretrained=True)
net = Resnet_fc(base_model, 5)
net.to(device)

loss_function = nn.MSELoss()
opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)


def train_model(model, loss_function, optimizer, num_epochs=30):
    since = time.time()

    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            counter = 0
            tmp = {}

            for inputs, labels in dataloaders[phase]:

                counter += 1

                current_label_set = labels.numpy().tolist()

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # forward
                    outputs = model(inputs).squeeze(-1).squeeze(-1)
                    label_prob = F.softmax(outputs, dim=1)

                    # pred = first classifier(p) output
                    _, preds = torch.max(outputs, 1)

                    base = torch.zeros((4, 5))
                    base.to(device)
                    label_onehot = base.scatter(1, labels.view(-1, 1), 1)

                    loss = loss_function(outputs, label_onehot)

                    # backward and update optimizer
                    if phase == 'train':

                        if counter % 10 == 0:
                            print("| e-{:03d} | i-{:03d} | loss: {:.4f} |"
                                  .format(epoch, counter, loss.item()))
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # metric for epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))
            print('{{"metric": "{}_loss", "value": {}}}'.format(phase, epoch_loss))
            print('{{"metric": "{}_acc", "value": {}}}'.format(phase, epoch_acc))

            # save the model when val acc is updated
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc

                if epoch > 3:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, "/output/best_model_e{:02d}_val_acc{:.2f}.pth.tar".format(epoch, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))


train_model(net, loss_function, opt, num_epochs=20)
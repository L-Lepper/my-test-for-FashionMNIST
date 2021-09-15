#   Training originally adopted from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F

import brevitas.nn as qnn
import brevitas.quant_tensor
from brevitas.export.onnx.generic.manager import BrevitasONNXManager

from brevitas.loss import WeightBitWidthWeightedBySize
from brevitas.loss import ActivationBitWidthWeightedBySize
from brevitas.loss import QuantLayerOutputBitWidthWeightedByOps


def main():
    # Setting seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    #   Hyperparameters
    learning_rate = 1e-4
    batch_size = 64
    epochs = 35

    #   set up train and test dataset and the dataloaders
    temp = setup_datasets_dataloader(batch_size)
    training_data, test_data, train_dataloader, test_dataloader = temp

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    #   Ask user: load already trained model or start new training?
    temp = check_pretrained_model(device)
    skip_training, trainied_model = temp

    #   start new training if no previous trained network should be used
    if skip_training == False:

        #   set model (using bit-width parameter)
        model = QuantLeNetD2L_DynBitWidth().to(device)
        model.apply(weights_init_normal)

        #   set training mode
        model.train()
        print('\nused model:\n', model, '\n\n')

        #   Set loss function
        loss_fn = nn.CrossEntropyLoss()  # criterion
        #loss_fn = Loss()

        #   Optimizing the Model Parameters
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

        #   Training in epochs with refreshing progress bar
        running_loss = []
        running_test_acc = []
        bit_widths = []
        bit_widths_param = []
        test_acc = 0

        #   3 classes from brevitas.loss
        loss_weight = WeightBitWidthWeightedBySize(model)
        loss_act = ActivationBitWidthWeightedBySize(model)
        loss_out = QuantLayerOutputBitWidthWeightedByOps(model)

        #loss_fn = Loss(loss_weight)

        #   Progressbar shows training progress
        with tqdm(total=epochs * len(training_data), desc="Training loss") as t:
            for epoch in range(0, epochs):
                progress_bar_data = [t, epoch + 1, epochs, test_acc]
                regularization_terms = [loss_weight, loss_act]

                #   train
                loss_epoch = train(progress_bar_data, train_dataloader, model, device, loss_fn, optimizer, regularization_terms)
                running_loss.append(loss_epoch)

                #   test
                test_acc = test(progress_bar_data, test_dataloader, model, device, loss_fn)
                running_test_acc.append(test_acc)

                #   get bit-width from Layer with BitWidthParameter
                bit_width_weigth_param = model.net[0].quant_weight_bit_width()
                bit_widths_param.append(bit_width_weigth_param)

                #   get bit-width from the first class from brevitas.loss (WeightBitWidthWeightedBySize)
                #bit_width_weigth = loss_weight.retrieve(as_average=True)
                #bit_widths.append(bit_width_weigth)

                #bit_width_act = loss_act.retrieve()
                # bit_width_layer = loss_layer.retrieve()


            print('Done!')

        #   plot graph for the loss
        loss_per_epoch = np.array([np.mean(loss_per_epoch) for loss_per_epoch in running_loss])
        display_loss_plot(loss_per_epoch)

        #   plot graph for the bit-width as parameter
        bit_width_per_epoch_param = np.array([bit_width_per_epoch2.detach().to("cpu") for bit_width_per_epoch2 in bit_widths_param])
        display_loss_plot(bit_width_per_epoch_param,
                          title="bit-width as parameter during Training",
                          xlabel="Iterations",
                          ylabel="Bit-Width Parameter")

        #   plot graph for the bit-width from class WeightBitWidthWeightedBySize
        #bit_width_per_epoch = np.array([bit_width_per_epoch.detach().to("cpu") for bit_width_per_epoch in bit_widths])
        #display_loss_plot(bit_width_per_epoch,
        #                  title="bit-width during Training",
        #                  xlabel="Iterations",
        #                  ylabel="Bit-Width")

        #   plot graph for the accuracy
        acc_per_epoch = [np.mean(acc_per_epoch) for acc_per_epoch in running_test_acc]
        display_loss_plot(acc_per_epoch, title="Test accuracy", ylabel="Accuracy [%]")

        print('\nTesting model accuracy: ', test(progress_bar_data, test_dataloader, model, device, loss_fn))

        #   Save model
        torch.save(model.state_dict(), 'FashionMNIST_LeNet5_quant_model_state_dict.pth')

        #   export model
        BrevitasONNXManager.export(model.to("cpu"),
                                   input_shape=(64, 1, 28, 28),
                                   export_path='FashionMNIST_LeNet5_quant_model.onnx'
                                   )

    trainied_model.eval()

    print('\nTesting model accuracy: ')
    print_model_accuracy(test_dataloader, device, trainied_model)

    print('finished')


class QuantLeNetD2L_DynBitWidth(nn.Module):
    """
    Model adapted from http://d2l.ai/chapter_convolutional-neural-networks/lenet.html [29.07.2021],
    but changed layers to quantized ones.
    """

    def __init__(self):
        super(QuantLeNetD2L_DynBitWidth, self).__init__()
        self.net = nn.Sequential(
            # qnn.QuantIdentity(bit_width=8, return_quant_tensor=True),      # BitWidthParameter(16, 4)
            qnn.QuantConv2d(in_channels=1,
                            out_channels=6,
                            kernel_size=5,
                            padding=2,
                            weight_bit_width_impl_type='parameter',
                            weight_bit_width=16,
                            return_quant_tensor=False),
            # input_quant=Int8ActPerTensorFloat),

            qnn.QuantReLU(bit_width_impl_type='parameter',
                          bit_width=16,
                          return_quant_tensor=False),

            nn.AvgPool2d(kernel_size=2,
                         stride=2),
            # qnn.QuantAvgPool2d(kernel_size=2, stride=2),

            qnn.QuantConv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            weight_bit_width_impl_type='parameter',
                            weight_bit_width=16,
                            return_quant_tensor=False),
            # input_quant=Int8ActPerTensorFloat),

            qnn.QuantReLU(bit_width_impl_type='parameter',
                          bit_width=16,
                          return_quant_tensor=False),

            nn.AvgPool2d(kernel_size=2,
                         stride=2),
            # qnn.QuantAvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            qnn.QuantLinear(in_features=16 * 5 * 5,
                            out_features=120,
                            bias=True,
                            weight_bit_width_impl_type='parameter',
                            weight_bit_width=16,
                            return_quant_tensor=False),
            # input_quant=Int8ActPerTensorFloat),

            qnn.QuantReLU(bit_width_impl_type='parameter',
                          bit_width=16,
                          return_quant_tensor=False),

            qnn.QuantLinear(in_features=120,
                            out_features=84,
                            bias=True,
                            weight_bit_width_impl_type='parameter',
                            weight_bit_width=16,
                            return_quant_tensor=False),
            # input_quant=Int8ActPerTensorFloat),

            qnn.QuantReLU(bit_width_impl_type='parameter',
                          bit_width=16,
                          return_quant_tensor=False),

            qnn.QuantLinear(in_features=84,
                            out_features=10,
                            bias=True,
                            weight_bit_width_impl_type='parameter',
                            weight_bit_width=16,
                            return_quant_tensor=False),
            # input_quant=Int8ActPerTensorFloat),
            # TensorNorm()
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


#   adapted and edited from https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch#comment89063424_49433937
def weights_init_normal(m):
    """
    Weight initialization for the model.
    Takes in a module and initializes all linear layers with weight
    values taken from a normal distribution.
    """

    # for every Linear layer in a model
    if isinstance(m, qnn.QuantLinear):
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0)
    elif isinstance(m, qnn.QuantConv2d):  # adapted from https://androidkt.com/initialize-weight-bias-pytorch/
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def train(progress_bar_data, dataloader, model, device, loss_fn, optimizer, regularization_terms):
    """
    Function is called in every epoch of training.
    :param progress_bar_data:
    :param dataloader:
    :param model:
    :param device:
    :param loss_fn:
    :param optimizer:
    :return:
    """

    size = len(dataloader.dataset)
    losses = []
    t, actual_epoch, total_epochs, acc_per_epoch = progress_bar_data
    weight_reg_loss, act_reg_loss = regularization_terms

    # ensure model is in training mode
    model.train()

    loss, current = [0, 0]
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y.argmax(-1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #   Outout current information in the progressbar (in the for loop for training)
        loss = loss.to('cpu')
        losses.append(loss.data.numpy())
        if batch % 100 == 0:  # calculate output for every 100 batches
            #   current: number of already processed images
            #   batch: number of the actual batch (dataset is divided in to parts of batch_size (e.g. 64))
            #   len(x): number ob images in the actual batch = batch_size
            loss, current = loss.item(), batch * len(X)
            if current != 0:
                t.update(len(X) * 100)
            t.set_description(
                f"Training loss: {loss:>7f}, image  [{current:>5d}/{size:>5d}] in epoch "
                f"[{actual_epoch:>3d}/{total_epochs:>3d}] (Accuracy: {acc_per_epoch * 100:>0.1f}%)")
            t.refresh()  # to show immediately the update

    #   show the last result
    t.update(actual_epoch * size - t.n)
    t.set_description(
        f"Training loss: {loss:>7f}, image  [{actual_epoch * size - t.n:>5d}/{size:>5d}] in epoch "
        f"[{actual_epoch:>3d}/{total_epochs:>3d}] (Accuracy: {acc_per_epoch * 100:>0.1f}%)")
    t.refresh()  # to show immediately the update
    return losses


def test(progress_bar_data, dataloader, model, device, loss_fn):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            output = torch.argmax(pred, dim=1)
            target = y
            y_true.extend(target.tolist())
            y_pred.extend(output.tolist())

    return accuracy_score(y_true, y_pred)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, inputs, targets, weight_reg_loss):

        # first compute binary cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='mean')
        regularization_loss = weight_reg_loss.retrieve(as_average=True)

        return ce_loss + regularization_loss


def setup_datasets_dataloader(batch_size):
    # Download training data from open datasets.
    transformation1 = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=transformation1
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return [training_data, test_data, train_dataloader, test_dataloader]


def check_pretrained_model(device):
    # use pretrained model, if available
    skip_training = False
    trainied_model = None
    try:
        #   Create model of the same type to load state_dict in it
        trainied_model = QuantLeNetD2L_DynBitWidth().to(device)
        trained_state_dict = torch.load('FashionMNIST_LeNet5_quant_model_state_dict.pth',
                                        map_location=torch.device(device))
        trainied_model.load_state_dict(trained_state_dict, strict=False)
        #   set eval mode for inferenz, change to train mode before training
        trainied_model.eval()

    except FileNotFoundError:
        print('FashionMNIST_LeNet5_quant_model_state_dict.pth found, '
              'start to train a new one...')
    else:
        temp = 'x'
        while temp != 'y' and temp != 'n':
            temp = input('Found an already trained model.\nSkip training? [y/n]')
        if temp == 'y':
            skip_training = True
        elif temp == 'n':
            skip_training = False
    return [skip_training, trainied_model]


def display_loss_plot(losses, title="Training loss", xlabel="Iterations", ylabel="Loss"):
    x_axis = [i for i in range(len(losses))]
    plt.plot(x_axis, losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def print_model_accuracy(test_dataloader, device, model):
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    # Check Accuracy
    correct = 0
    total = 0
    # Check accuracy of each class
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(outputs, brevitas.quant_tensor.QuantTensor):
                _, predicted = torch.max(outputs.value, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)

            # Check Accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Check accuracy of each class
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Check Accuracy
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    # Check accuracy of each class
    print('\nCheck accuracy of each class')
    for i in range(10):
        print("Accuracy of %5s : %2d %%" % (labels_map[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    main()

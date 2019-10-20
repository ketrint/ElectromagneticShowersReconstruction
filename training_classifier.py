import comet_ml
from comet_ml import Experiment
import click
import numpy as np
import torch
import torch.nn as nn
from nets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, average_precision_score
from torch_geometric.data import DataLoader
from preprocessing import preprocess_dataset
from utils import RunningAverageMeter, plot_aucs
from tqdm import tqdm
import sys


def str_to_class(classname: str):
    """
    Function to get class object by its name signature
    :param classname: str
        name of the class
    :return: class object with the same name signature as classname
    """
    return getattr(sys.modules[__name__], classname)


def predict_one_shower(shower, graph_embedder, edge_classifier):
    # TODO: batch training
    embeddings = graph_embedder(shower)
    edge_labels_true = (shower.y[shower.edge_index[0]] == shower.y[shower.edge_index[1]]).view(-1)
    edge_data = torch.cat([
        embeddings[shower.edge_index[0]],
        embeddings[shower.edge_index[1]]
    ], dim=1)
    edge_labels_predicted = edge_classifier(edge_data).view(-1)

    return edge_labels_true, edge_labels_predicted


@click.command()
@click.option('--datafile', type=str, default='./data/train_.pt')
@click.option('--project_name', type=str, prompt='Enter project name', default='em_showers_network_training')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--epochs', type=int, default=1000)
@click.option('--learning_rate', type=float, default=1e-3)
@click.option('--dim_out', type=int, default=144)
@click.option('--graph_embedder', type=str, default='GraphNN_KNN_v1')
@click.option('--edge_classifier', type=str, default='EdgeClassifier_v1')
def main(
        datafile='./data/train_.pt',
        epochs=1000,
        learning_rate=1e-3,
        dim_out=144,
        device='cuda:3',
        project_name='em_showers_net_training',
        work_space='schattengenie',
        graph_embedder='GraphNN_KNN_v1',
        edge_classifier='EdgeClassifier_v1'
):
    experiment = Experiment(project_name=project_name, workspace=work_space)

    device = torch.device(device)
    showers = preprocess_dataset(datafile)
    showers_train, showers_test = train_test_split(showers, random_state=1337)

    train_loader = DataLoader(showers_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(showers_test, batch_size=1, shuffle=True)

    k = showers[0].x.shape[1]
    print(k)
    graph_embedder = str_to_class(graph_embedder)(dim_out=dim_out, k=k).to(device)
    edge_classifier = str_to_class(edge_classifier)(dim_out=dim_out).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(list(graph_embedder.parameters()) + list(edge_classifier.parameters()),
                                 lr=learning_rate)

    loss_train = RunningAverageMeter()
    loss_test = RunningAverageMeter()
    roc_auc_test = RunningAverageMeter()
    pr_auc_test = RunningAverageMeter()
    acc_test = RunningAverageMeter()
    class_disbalance = RunningAverageMeter()

    for _ in tqdm(range(epochs)):
        for shower in train_loader:
            shower = shower.to(device)
            edge_labels_true, edge_labels_predicted = predict_one_shower(shower,
                                                                         graph_embedder=graph_embedder,
                                                                         edge_classifier=edge_classifier)
            # calculate the batch loss
            loss = criterion(edge_labels_predicted, edge_labels_true.float())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train.update(loss.item())
            class_disbalance.update((edge_labels_true.sum().float() / len(edge_labels_true)).item())

        y_true_list = []
        y_pred_list = []
        for shower in test_loader:
            shower = shower.to(device)
            edge_labels_true, edge_labels_predicted = predict_one_shower(shower,
                                                                         graph_embedder=graph_embedder,
                                                                         edge_classifier=edge_classifier)

            # calculate the batch loss
            loss = criterion(edge_labels_predicted, edge_labels_true.float())
            y_true, y_pred = edge_labels_true.detach().cpu().numpy(), edge_labels_predicted.detach().cpu().numpy()
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            acc = accuracy_score(y_true, y_pred.round())
            roc_auc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)
            loss_test.update(loss.item())
            acc_test.update(acc)
            roc_auc_test.update(roc_auc)
            pr_auc_test.update(pr_auc)
            class_disbalance.update((edge_labels_true.sum().float() / len(edge_labels_true)).item())


        experiment.log_metric('loss_test', loss_test.val)
        experiment.log_metric('acc_test', acc_test.val)
        experiment.log_metric('roc_auc_test', roc_auc_test.val)
        experiment.log_metric('pr_auc_test', pr_auc_test.val)
        experiment.log_metric('class_disbalance', class_disbalance.val)

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        f = plot_aucs(y_true=y_true, y_pred=y_pred)
        experiment.log_figure("Optimization dynamic", f, overwrite=True)
        experiment_key = experiment.get_key()
        # TODO: save best
        torch.save(graph_embedder.state_dict(), "graph_embedder_{}.pt".format(experiment_key))
        torch.save(edge_classifier.state_dict(), "edge_classifier_{}.pt".format(experiment_key))


if __name__ == "__main__":
    main()

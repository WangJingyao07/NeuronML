import csv
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_generator import DataGenerator
from maml import MAML

# example for fast deployment
datasource = 'sinusoid'
num_classes = 5
baseline = None
pretrain_iterations = 0
metatrain_iterations = 15000
meta_batch_size = 25
meta_lr = 0.001
update_batch_size = 5
update_lr = 1e-3
num_updates = 1
norm = 'batch_norm'
num_filters = 64
conv = True
max_pool = False
stop_grad = False
log = True
logdir = './logs'
resume = True
train_flag = True
test_iter = -1
test_set = False
train_update_batch_size = -1
train_update_lr = -1


def train(model, data_generator, exp_string, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 1000 if datasource == 'sinusoid' else 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

    if log:
        writer = SummaryWriter(logdir + '/' + exp_string)

    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    for itr in range(resume_itr, pretrain_iterations + metatrain_iterations):
        batch_x, batch_y, amp, phase = data_generator.generate()

        if baseline == 'oracle':
            batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
            for i in range(meta_batch_size):
                batch_x[i, :, 1] = amp[i]
                batch_x[i, :, 2] = phase[i]

        inputa = torch.tensor(batch_x[:, :num_classes * update_batch_size, :], dtype=torch.float32)
        labela = torch.tensor(batch_y[:, :num_classes * update_batch_size, :], dtype=torch.float32)
        inputb = torch.tensor(batch_x[:, num_classes * update_batch_size:, :], dtype=torch.float32)
        labelb = torch.tensor(batch_y[:, num_classes * update_batch_size:, :], dtype=torch.float32)

        model.train()
        if itr < pretrain_iterations:
            loss = model.meta_train(inputa, inputb, labela, labelb)[0]  # Adjusted to use meta_train
        else:
            loss = model.meta_train(inputa, inputb, labela, labelb)[0]

        prelosses.append(loss.item())

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            if log:
                writer.add_scalar('Loss', np.mean(prelosses), itr)
            postlosses.append(loss.item())

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Pretrain Iteration ' + str(itr) if itr < pretrain_iterations else 'Iteration ' + str(itr - pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr != 0) and itr % SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), f"{logdir}/{exp_string}/model_{itr}.pth")

    torch.save(model.state_dict(), f"{logdir}/{exp_string}/model_{itr}.pth")

NUM_TEST_POINTS = 600

def test(model, data_generator, exp_string):
    np.random.seed(1)
    random.seed(1)
    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):
        batch_x, batch_y, amp, phase = data_generator.generate(train=False)
        inputa = torch.tensor(batch_x[:, :num_classes * update_batch_size, :], dtype=torch.float32)
        labela = torch.tensor(batch_y[:, :num_classes * update_batch_size, :], dtype=torch.float32)
        inputb = torch.tensor(batch_x[:, num_classes * update_batch_size:, :], dtype=torch.float32)
        labelb = torch.tensor(batch_y[:, num_classes * update_batch_size:, :], dtype=torch.float32)

        with torch.no_grad():
            model.eval()
            accuracy = model.meta_train(inputa, inputb, labela, labelb)[0]  # Adjusted to use meta_train
        metaval_accuracies.append(accuracy.item())

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = logdir + '/' + exp_string + '/' + f'test_ubs{update_batch_size}_stepsize{update_lr}.csv'
    out_pkl = logdir + '/' + exp_string + '/' + f'test_ubs{update_batch_size}_stepsize{update_lr}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update' + str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

if __name__ == "__main__":
    data_generator = DataGenerator(update_batch_size * 2, meta_batch_size)
    dim_input = data_generator.dim_input
    dim_output = data_generator.dim_output

    model = MAML(dim_input, dim_output, test_num_updates=num_updates)  # Changed argument to test_num_updates

    exp_string = f'cls_{num_classes}.mbs_{meta_batch_size}.ubs_{train_update_batch_size}.numstep{num_updates}.updatelr{update_lr}'

    if resume:
        checkpoint = torch.load(f"{logdir}/{exp_string}/model_{test_iter}.pth")
        model.load_state_dict(checkpoint)

    if train_flag:
        train(model, data_generator, exp_string)
    else:
        test(model, data_generator, exp_string)

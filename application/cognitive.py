import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pdb

import scipy
from scipy import ndimage
from scipy import linalg

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

import numpy as np
from numpy import fft

from scipy import io as spio

from learner import learner



device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(precision=5)
np.set_printoptions(precision=5)

TESTTASK = 'nand';
TESTTASKNEG = 'and'
LR = .3 * 1e-2
NBGEN = 400
N = 70
BS = 500
assert BS % 2 == 0

TAU = 100
DT = 20

T = 50
STIMTIME = 20
REWARDTIME = 10
RESPONSETIME = 10
STARTRESPONSETIME = 25
ENDRESPONSETIME = STARTRESPONSETIME + RESPONSETIME
STARTREWARDTIME = 36
ENDREWARDTIME = STARTREWARDTIME + REWARDTIME
assert ENDREWARDTIME < T

JINIT = 1.5
TAU_ET = 1000.0
PROBAMODUL = .1
ALPHAMODUL = .5
ETA = .03
MULOSSTRACE = .9
MAXDW = 1e-2
INITALPHA = .5

alltasks = ['01', 'anti01', '10', 'anti10', 'watchstim1', 'watchstim2', 'dms', 'antiwatchstim2', 'antiwatchstim1', 'or',
            'and', 'nor', 'nand', 'dnms']

NBSTIMNEURONS = 2
NBREWARDNEURONS = 2
NBBIASNEURONS = 1
NBINPUTNEURONS = NBSTIMNEURONS + NBREWARDNEURONS + NBBIASNEURONS
NBRESPNEURONS = 2
NBOUTPUTNEURONS = NBRESPNEURONS
NBRESPSIGNALNEURONS = NBRESPNEURONS
STIMNEURONS = np.arange(NBSTIMNEURONS)
INPUTNEURONS = np.arange(NBINPUTNEURONS)
OUTPUTNEURONS = np.arange(N - NBOUTPUTNEURONS, N)
RESPNEURONS = np.arange(N - NBOUTPUTNEURONS, N - NBOUTPUTNEURONS + NBRESPNEURONS)
REWARDNEURONS = np.arange(NBSTIMNEURONS, NBSTIMNEURONS + NBREWARDNEURONS)
BIASNEURONS = np.arange(NBSTIMNEURONS + NBREWARDNEURONS, NBSTIMNEURONS + NBREWARDNEURONS + NBBIASNEURONS)
FIRSTRESPSIGNALNEURON = NBSTIMNEURONS + NBREWARDNEURONS + NBBIASNEURONS
assert FIRSTRESPSIGNALNEURON == NBINPUTNEURONS
RESPSIGNALNEURONS = np.arange(FIRSTRESPSIGNALNEURON, FIRSTRESPSIGNALNEURON + NBRESPSIGNALNEURONS)

BIASVALUE = 1.0

NBTASKSPERGEN = 2
NBTRIALSLOSS = 100
NBTRIALS = 300 + NBTRIALSLOSS
WDECAY = 3e-4
MUTATIONSIZE = .01
REWARDSIZE = 3.0
STIMSIZE = 3.0
RESPSIGNALSIZE = 3.0

totalnbtasks = 0
ticstart = time.time()

with torch.no_grad():
    PRINTING = True
    w = torch.randn(N, N) * JINIT / np.sqrt(N)
    w = w.to(device)

    alpha = INITALPHA * torch.ones_like(w).to(device)
    w.data[:NBINPUTNEURONS, :] = 0
    winit = w.clone()
    optimizer = optim.Adam([w, alpha], lr=LR, weight_decay=WDECAY)

    evolosses = []
    responses0 = []
    binarylosses = []
    wgradnorms = []
    mytaskprev = mytaskprevprev = mytaskprevprevprev = -1

    for numgen in range(NBGEN):
        TESTING = False
        if numgen == 0 or numgen == NBGEN - 1 or numgen % 10 == 0:
            TESTING = True
            if PRINTING:
                print("TESTING")
        tic = time.time()
        responses0thisgen = []

        bw = torch.dstack(BS * [w]).movedim(2, 0).to(device)
        balpha = torch.dstack(BS * [alpha]).movedim(2, 0).to(device)
        mutations_wandalpha = []
        for n, x in enumerate((bw, balpha)):
            mutations = torch.randn_like(x, requires_grad=False).to(device) * MUTATIONSIZE
            mutations[0, :, :] = 0
            mutations[BS // 2:, :, :] = -mutations[:BS // 2, :, :]
            if TESTING:
                mutations *= 0.0
            x += mutations
            mutations_wandalpha.append(mutations)

        bw.data[:, :NBINPUTNEURONS, :] = 0
        bworig = bw.clone()

        lifelosses = torch.zeros(BS, requires_grad=False).to(device)
        lifemselosses = torch.zeros(BS, requires_grad=False).to(device)
        lifeblosses = torch.zeros(BS, requires_grad=False).to(device)
        allresps = []

        for numtask in range(NBTASKSPERGEN):
            print("NUMTASK", numtask)
            if mytaskprev != -1:
                mytaskprevprevprev = mytaskprevprev
                mytaskprevprev = mytaskprev
            else:
                mytaskprevprevprev = mytaskprevprev = mytaskprev = -1

            while True:
                mytask = np.random.choice(alltasks)
                if mytask != mytaskprevprevprev and mytask != mytaskprevprev and mytask != mytaskprev and mytask != mytaskprev:
                    break
            mytaskprevprevprev = mytaskprevprev
            mytaskprevprev = mytaskprev
            mytaskprev = mytask
            print("mytask", mytask)

            s = np.zeros((BS, NBSTIMNEURONS, T), dtype=np.float32)
            s[:, 0, :STIMTIME] = np.random.randn(BS, 1, STIMTIME).astype(np.float32) * STIMSIZE
            s[:, 1, :STIMTIME] = np.random.randn(BS, 1, STIMTIME).astype(np.float32) * STIMSIZE
            s[:, :, STIMTIME:] = 0

            r = np.zeros((BS, NBREWARDNEURONS, T), dtype=np.float32)
            r[:, 0, STARTREWARDTIME:ENDREWARDTIME] = 1.0

            resp = np.zeros((BS, NBRESPSIGNALNEURONS, T), dtype=np.float32)
            resp[:, 0, STARTRESPONSETIME:ENDRESPONSETIME] = 1.0

            # Not used
            output = np.zeros((BS, NBOUTPUTNEURONS, T), dtype=np.float32)
            for t in range(STIMTIME - 1, T - 1):
                for b in range(BS):
                    task = int(mytask[0])
                    taskneurons = OUTPUTNEURONS + task
                    s[b, NBSTIMNEURONS:, t + 1] = alpha[taskneurons[0], taskneurons[1]] * w[
                        taskneurons[0], taskneurons[1]] * r[b, 0, t]
                    s[b, NBSTIMNEURONS:, t + 1] = s[b, NBSTIMNEURONS:, t + 1] + PROBAMODUL * w[
                        taskneurons[0], taskneurons[1]] * r[b, 0, t] * r[b, 1, t]
                    s[b, NBSTIMNEURONS:, t + 1] = s[b, NBSTIMNEURONS:, t + 1] + ALPHAMODUL * w[
                        taskneurons[0], taskneurons[1]] * r[b, 0, t] * resp[b, 0, t]
                    s[b, NBSTIMNEURONS:, t + 1] = s[b, NBSTIMNEURONS:, t + 1] + ETA * w[
                        taskneurons[0], taskneurons[1]] * np.random.randn()

                    s[b, NBSTIMNEURONS:, t + 1] = MULOSSTRACE * s[b, NBSTIMNEURONS:, t]
                    output[b, :, t + 1] = np.dot(w, s[b, :, t + 1])
                    output[b, :, t + 1] = 1.0 / (1.0 + np.exp(-output[b, :, t + 1]))

            # Just for the record...
            sorig = s.copy()
            rolds = r.copy()
            respolds = resp.copy()
            outputolds = output.copy()
            mutations = torch.dstack(BS * [torch.from_numpy(sorig).to(device)]).movedim(2, 0)
            mutations[0, :, :] = 0
            mutations[BS // 2:, :, :] = -mutations[:BS // 2, :, :]
            s = sorig + mutations.cpu().numpy()

            bw.data = w.data
            balpha.data = alpha.data

            for t in range(STIMTIME, T - 1):
                inputt = torch.from_numpy(s[:, :, t]).to(device)
                resp[:, 0, t + 1] = torch.sigmoid(
                    torch.sum(torch.matmul(w, inputt[:, :NBSTIMNEURONS]), axis=2) - BIASVALUE)
                resp[:, 1, t + 1] = 1 - resp[:, 0, t + 1]
                r[:, :, t + 1] = rolds[:, :, t + 1] + MUTATIONSIZE * np.random.randn(BS,
                                                                                     NBREWARDNEURONS) + REWARDSIZE * (
                                             resp[:, :, t + 1] - respolds[:, :, t])
                output[:, :, t + 1] = torch.sigmoid(torch.sum(torch.matmul(w, inputt), axis=2) - BIASVALUE)

            if TESTING:
                plt.figure(figsize=(12, 12))
                plt.plot(s[0, 0, :])
                plt.plot(r[0, 0, :])
                plt.plot(resp[0, 0, :])
                plt.plot(output[0, 0, :])
                plt.title("Task " + mytask + ": s, r, resp, output ")
                plt.show()

            allresps.append(resp)
            loss = np.zeros(BS)

            for b in range(BS):
                loss[b] = np.mean(0.5 * (output[b, :, STIMTIME - RESPONSETIME:STIMTIME] - 1) ** 2)
                loss[b] = loss[b] + 0.5 * np.mean(0.5 * (output[b, :, STIMTIME + RESPONSETIME:] - 0) ** 2)

            bwl2norm = torch.norm(bw[:, :NBSTIMNEURONS, :] * bworig[:, :NBSTIMNEURONS, :], dim=1)
            loss += WDECAY * 0.5 * (bwl2norm ** 2).cpu().numpy()
            loss = torch.from_numpy(loss).to(device)

            lifeloss = 0.0
            lifemseloss = 0.0
            lifebloss = 0.0
            for n in range(BS):
                lifeloss = lifeloss + torch.sum(
                    w[RESPNEURONS[0], RESPNEURONS[1]] * bworig[n, :NBSTIMNEURONS, STIMTIME - RESPONSETIME:STIMTIME])
                lifemseloss = lifemseloss + torch.sum((w[RESPNEURONS[0], RESPNEURONS[1]] * bworig[n, :NBSTIMNEURONS,
                                                                                           STIMTIME - RESPONSETIME:STIMTIME]) ** 2)
                lifebloss = lifebloss + torch.sum(balpha[RESPNEURONS[0], RESPNEURONS[1]] * bworig[n, :NBSTIMNEURONS,
                                                                                           STIMTIME - RESPONSETIME:STIMTIME])

            lifelosses += lifeloss
            lifemselosses += lifemseloss
            lifeblosses += lifebloss

            loss = loss + LIFEW * lifeloss
            loss = loss + LIFEMSEW * lifemseloss
            loss = loss + LIFEBW * lifebloss

            if TESTING:
                print("lifeloss", lifeloss.cpu().numpy())
                print("lifemseloss", lifemseloss.cpu().numpy())
                print("lifebloss", lifebloss.cpu().numpy())

            alllosses.append(loss)
            losses[ll] = np.mean(loss.cpu().numpy())
            allstims.append(sorig)

        lw = torch.norm(w[:, :NBSTIMNEURONS, :] * w[:, :NBSTIMNEURONS, :], dim=1)
        bw = bworig + LR * torch.matmul((loss - L2WDECAY * lw) / BS, w)
        alpha = alpha + LR * torch.matmul((lifelosses - L2WDECAY * lw) / BS, w)
        alpha = torch.max(torch.zeros_like(alpha), alpha)

        # Sanity check
        alpha[RESPNEURONS[0], RESPNEURONS[1]] = 0

        bw = bw - bw[RESPNEURONS[0], RESPNEURONS[1], STIMTIME - RESPONSETIME:STIMTIME] * w[RESPNEURONS[0],
                                                                                         RESPNEURONS[1], :]
        balpha = balpha - balpha[RESPNEURONS[0], RESPNEURONS[1], STIMTIME - RESPONSETIME:STIMTIME] * w[RESPNEURONS[0],
                                                                                                     RESPNEURONS[1], :]
    #
    #     if TESTING:
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(losses[:ll + 1])
    #         plt.title("Losses over time")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(bw[:, :NBSTIMNEURONS, STIMTIME - RESPONSETIME:STIMTIME].cpu().numpy())
    #         plt.title("Weight changes over time (first stim)")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(bw[:, NBSTIMNEURONS:, STIMTIME - RESPONSETIME:STIMTIME].cpu().numpy())
    #         plt.title("Weight changes over time (output neurons)")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(balpha[:, :NBSTIMNEURONS, STIMTIME - RESPONSETIME:STIMTIME].cpu().numpy())
    #         plt.title("Alpha changes over time (first stim)")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(balpha[:, NBSTIMNEURONS:, STIMTIME - RESPONSETIME:STIMTIME].cpu().numpy())
    #         plt.title("Alpha changes over time (output neurons)")
    #         plt.show()
    #
    #     if TESTING:
    #         plt.figure(figsize=(12, 12))
    #         plt.imshow(bw[:, :NBSTIMNEURONS, :].cpu().numpy())
    #         plt.colorbar()
    #         plt.title("w matrix")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.imshow(alpha[:, :NBSTIMNEURONS, :].cpu().numpy())
    #         plt.colorbar()
    #         plt.title("alpha matrix")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.imshow(balpha[:, :NBSTIMNEURONS, :].cpu().numpy())
    #         plt.colorbar()
    #         plt.title("balpha matrix")
    #         plt.show()
    #
    #     if TESTING:
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(lw[:, :NBSTIMNEURONS].cpu().numpy())
    #         plt.title("lw norm")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(lw[:, NBSTIMNEURONS:].cpu().numpy())
    #         plt.title("lw norm")
    #         plt.show()
    #
    #     plt.figure(figsize=(12, 12))
    #     plt.plot(lw.cpu().numpy())
    #     plt.title("lw norm")
    #     plt.show()
    #
    #     if TESTING:
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(losses)
    #         plt.title("Losses")
    #         plt.show()
    #
    #     plt.figure(figsize=(12, 12))
    #     plt.plot(losses)
    #     plt.title("Losses")
    #     plt.show()
    #
    #     mymap = cm.get_cmap("RdYlBu")
    #
    #     if TESTING:
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(losses)
    #         plt.title("Losses")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(bw[:, :NBSTIMNEURONS, STIMTIME - RESPONSETIME:STIMTIME].cpu().numpy())
    #         plt.title("Weight changes over time (first stim)")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(bw[:, NBSTIMNEURONS:, STIMTIME - RESPONSETIME:STIMTIME].cpu().numpy())
    #         plt.title("Weight changes over time (output neurons)")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(balpha[:, :NBSTIMNEURONS, STIMTIME - RESPONSETIME:STIMTIME].cpu().numpy())
    #         plt.title("Alpha changes over time (first stim)")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(balpha[:, NBSTIMNEURONS:, STIMTIME - RESPONSETIME:STIMTIME].cpu().numpy())
    #         plt.title("Alpha changes over time (output neurons)")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.imshow(bw[:, :NBSTIMNEURONS, :].cpu().numpy())
    #         plt.colorbar()
    #         plt.title("w matrix")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.imshow(alpha[:, :NBSTIMNEURONS, :].cpu().numpy())
    #         plt.colorbar()
    #         plt.title("alpha matrix")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.imshow(balpha[:, :NBSTIMNEURONS, :].cpu().numpy())
    #         plt.colorbar()
    #         plt.title("balpha matrix")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(lw[:, :NBSTIMNEURONS].cpu().numpy())
    #         plt.title("lw norm")
    #         plt.show()
    #
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(lw[:, NBSTIMNEURONS:].cpu().numpy())
    #         plt.title("lw norm")
    #         plt.show()
    #
    #     plt.figure(figsize=(12, 12))
    #     plt.plot(lw.cpu().numpy())
    #     plt.title("lw norm")
    #     plt.show()
    #
    #     if TESTING:
    #         plt.figure(figsize=(12, 12))
    #         plt.plot(losses)
    #         plt.title("Losses")
    #         plt.show()
    #
    #     plt.figure(figsize=(12, 12))
    #     plt.plot(losses)
    #     plt.title("Losses")
    #     plt.show()
    #
    #     mymap = cm.get_cmap("RdYlBu")
    #
    #     plt.figure(figsize=(12, 12))
    #     plt.imshow(bw[:, :NBSTIMNEURONS, :].cpu().numpy(), cmap=mymap)
    #     plt.colorbar()
    #     plt.title("w matrix")
    #     plt.show()
    #
    #     plt.figure(figsize=(12, 12))
    #     plt.imshow(alpha[:, :NBSTIMNEURONS, :].cpu().numpy(), cmap=mymap)
    #     plt.colorbar()
    #     plt.title("alpha matrix")
    #     plt.show()
    #
    #     plt.figure(figsize=(12, 12))
    #     plt.imshow(balpha[:, :NBSTIMNEURONS, :].cpu().numpy(), cmap=mymap)
    #     plt.colorbar()
    #     plt.title("balpha matrix")
    #     plt.show()
    #
    #     print("numgen", numgen)
    #     print("losses", losses)
    #     print("lifelosses", lifelosses)
    #     print("lifemselosses", lifemselosses)
    #     print("lifeblosses", lifeblosses)
    #
    # print("time in minutes", (time.time() - ticstart) / 60.0)

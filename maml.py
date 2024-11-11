import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class MAML(nn.Module):
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = 0.01
        self.meta_lr = 0.001
        self.test_num_updates = test_num_updates

        if dim_input == 1:
            self.feature_extractor = self.construct_fc_weights()
            self.classifier = nn.Linear(self.dim_hidden[-1], dim_output)
        else:
            self.feature_extractor = self.construct_conv_weights()
            self.classifier = nn.Linear(self.dim_hidden[-1] * 4 * 4, dim_output)

        self.structure_network = nn.Sequential(
            nn.Linear(self.dim_hidden[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def construct_fc_weights(self):
        layers = []
        input_dim = self.dim_input
        self.dim_hidden = [40, 40]
        for hidden_dim in self.dim_hidden:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        return nn.Sequential(*layers)

    def construct_conv_weights(self):
        conv_layers = []
        self.channels = 1
        self.dim_hidden = [64, 64, 64, 64]
        conv_layers.append(nn.Conv2d(self.channels, self.dim_hidden[0], kernel_size=3, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(self.dim_hidden[0], self.dim_hidden[1], kernel_size=3, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(self.dim_hidden[1], self.dim_hidden[2], kernel_size=3, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(self.dim_hidden[2], self.dim_hidden[3], kernel_size=3, padding=1))
        conv_layers.append(nn.ReLU())
        return nn.Sequential(*conv_layers)

    def forward(self, x):
        if len(x.shape) == 2:
            features = self.feature_extractor(x)
        else:
            features = self.feature_extractor(x)
            features = torch.flatten(features, 1)
        output = self.classifier(features)
        return output

    def task_metalearn(self, inputa, labela, inputb, labelb, num_updates):
        fast_weights = [param.clone().detach().requires_grad_(True) for param in self.classifier.parameters()]
        outputa = self.forward(inputa)
        lossa = F.mse_loss(outputa, labela)
        grads = torch.autograd.grad(lossa, fast_weights, create_graph=True)
        fast_weights = [w - self.update_lr * g for w, g in zip(fast_weights, grads)]

        for _ in range(num_updates):
            outputb = F.linear(self.feature_extractor(inputb), fast_weights[0], fast_weights[1])
            lossb = F.mse_loss(outputb, labelb)
            sparsity_loss = torch.sum(torch.abs(fast_weights[0]))
            importance_loss = torch.sum(torch.square(fast_weights[0]))
            hebbian_loss = self.genetic_algorithm_hebbian_loss(inputb, outputb)


            total_loss = lossb + 0.001 * sparsity_loss + 0.001 * importance_loss + 0.001 * hebbian_loss
            grads = torch.autograd.grad(total_loss, fast_weights, create_graph=True)
            fast_weights = [w - self.update_lr * g for w, g in zip(fast_weights, grads)]

        return lossa, total_loss

    def genetic_algorithm_hebbian_loss(self, inputb, outputb):
        population_size = 20
        generations = 50
        mutation_rate = 0.1


        population = [torch.randn_like(inputb) for _ in range(population_size)]
        
        for generation in range(generations):
            fitness_scores = []

            for individual in population:
                with torch.no_grad():
                    fitness = torch.sum(individual * outputb).item()
                    fitness_scores.append(fitness)

            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
            parents = sorted_population[:population_size // 2]

            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(parents, 2)
                crossover_point = random.randint(0, len(parent1) - 1)
                child = torch.cat((parent1[:crossover_point], parent2[crossover_point:]))

                if random.random() < mutation_rate:
                    mutation = torch.randn_like(child) * 0.1
                    child += mutation

                new_population.append(child)

            population = new_population

        best_individual = sorted_population[0]
        hebbian_loss = torch.sum(best_individual * outputb)

        return hebbian_loss

    def optimize_structure(self, input_data, label_data):
        optimizer = torch.optim.SGD(self.structure_network.parameters(), lr=self.update_lr)
        optimizer.zero_grad()
        features = self.feature_extractor(input_data)
        importance_scores = self.structure_network(features)
        threshold = torch.mean(importance_scores) * 0.5
        loss = torch.sum(torch.square(importance_scores - threshold))
        loss.backward()
        optimizer.step()

    def meta_train(self, inputa, inputb, labela, labelb):
        num_updates = self.test_num_updates
        lossa, total_loss = self.task_metalearn(inputa, labela, inputb, labelb, num_updates)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.meta_lr)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        self.optimize_structure(inputb, labelb)

        return lossa, total_loss

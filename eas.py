import numpy as np
import matplotlib.pyplot as plt

class EvolutionaryAlgorithm():
    def __init__(self, ff, gs, b, mp, rp, g, p):
        self.ff = ff # Fitness Function
        self.gs = gs # Gene Size
        self.b = b # List of barrier values for each gene
        self.mp = mp # Mutation Probability determines mutation size
        self.rp = rp # Recombination Probability
        self.g = g # Generations
        self.p = p # Population

        # Initialize population matrix (p individuals × gs genes) with zeros
        self.i = np.zeros((self.p, self.gs))
        for i in range(self.p):
            for j in range(self.gs):
                # Randomly assign each gene a value within its barrier range
                self.i[i][j] = np.random.rand() * (self.b[j][1] - self.b[j][0]) # Creates individual [-1, 1]

        self.fHistory = np.zeros(self.g) # Fitness History over generations
        self.t = self.g*self.p # Total number of tournaments per generation (for competition)
        self.gene = self.i[0] # Placeholder for best gene found

    def run(self):
        # Loop over generations
        for i in range(self.g):
            bestF = -1e9 # Initialize best fitness for this generation (very low value)

            # Run tournaments to evolve population
            for j in range(self.t):
                print(f"Running Gen {i + 1}/{self.g} Tournament {j + 1}/{self.t}")

                # Randomly select two distinct individuals for competition
                a = np.random.randint(0, self.p)
                b = np.random.randint(0, self.p)
                while a == b:
                    b = np.random.randint(0, self.p)

                # Evaluate fitness of each individual
                fA = self.ff(self.i[a])
                fB = self.ff(self.i[b])

                # Determine winner and loser based on fitness
                winner = a
                loser = b
                if fA < fB:
                    winner = b
                    loser = a

                # Update best gene if current winner is better than previous best
                if self.ff(self.i[winner]) >= bestF:
                    bestF = self.ff(self.i[winner])
                    self.gene = self.i[winner]

                # Recombination: copy genes from winner to loser with probability rp
                for k in range(self.gs):
                    if np.random.random() < self.rp:
                        self.i[loser][k] = self.i[winner][k]

                # Mutation: add Gaussian noise to loser's genes
                for k in range(self.gs):
                    self.i[loser][k] += np.random.normal(
                        self.mp*(-1*(self.b[k][1] - self.b[k][0]))/5, # Mean
                        self.mp*(self.b[k][1] - self.b[k][0]),         # Standard deviation
                        size = 1
                    )
                    # Ensure mutated gene stays within barrier limits
                    self.i[loser][k] = np.clip(self.i[loser][k], self.b[k][0], self.b[k][1])

            # Record best fitness of this generation
            self.fHistory[i] = bestF

        # Print final best fitness
        print(f"Fitness Acheived: {self.fHistory[-1]}")

        # Plot fitness progression over generations
        plt.plot(self.fHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title(f"Evolution with Population = {self.p}, Mutation Probability = {self.mp}, Recombination Probability = {self.rp}")
        plt.show()

        # Save best gene for future use
        np.save(f"gene.npy", self.gene)
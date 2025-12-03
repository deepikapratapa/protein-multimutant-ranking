import numpy as np
import random
from typing import List, Tuple, Callable, Dict
from dataclasses import dataclass
import pandas as pd
from copy import deepcopy

@dataclass
class Mutant:
    """Represents a protein mutant with multiple mutations"""
    positions: List[int]  # Mutation positions in the protein
    mutations: List[str]  # Amino acid substitutions (e.g., ['A', 'V', 'L'])
    fitness: float = None  # ΔΔG prediction from ML model
    
    def __hash__(self):
        return hash((tuple(self.positions), tuple(self.mutations)))
    
    def __eq__(self, other):
        return (self.positions == other.positions and 
                self.mutations == other.mutations)
    
    def to_dict(self):
        """Convert mutant to dictionary for feature extraction"""
        return {
            'positions': self.positions,
            'mutations': self.mutations,
            'fitness': self.fitness
        }


class ProteinGeneticAlgorithm:
    """
    Genetic Algorithm for exploring protein mutation space.
    Uses ML model predictions as fitness function to find stabilizing mutations.
    """
    
    def __init__(self,
                 ml_model,
                 feature_extractor: Callable,
                 protein_length: int,
                 n_mutations: int = 2,
                 population_size: int = 100,
                 n_generations: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism_ratio: float = 0.1,
                 tournament_size: int = 3,
                 amino_acids: List[str] = None):
        """
        Initialize Genetic Algorithm for protein mutation exploration.
        
        Parameters:
        -----------
        ml_model : trained sklearn model
            ML model that predicts ΔΔG from mutant features
        feature_extractor : callable
            Function that converts Mutant object to feature vector for ML model
            Should take Mutant object and return features compatible with ml_model
        protein_length : int
            Length of the protein sequence
        n_mutations : int
            Number of simultaneous mutations (2 for double, 3 for triple)
        population_size : int
            Number of individuals in population
        n_generations : int
            Number of generations to evolve
        mutation_rate : float
            Probability of mutation per individual
        crossover_rate : float
            Probability of crossover between parents
        elitism_ratio : float
            Fraction of top individuals to preserve
        tournament_size : int
            Number of individuals in tournament selection
        amino_acids : list
            List of amino acid single letter codes (default: 20 standard AA)
        """
        self.ml_model = ml_model
        self.feature_extractor = feature_extractor
        self.protein_length = protein_length
        self.n_mutations = n_mutations
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size
        
        # 20 standard amino acids (excluding the wild-type at each position)
        if amino_acids is None:
            self.amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        else:
            self.amino_acids = amino_acids
        
        # Tracking
        self.population = []
        self.history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }
        self.evaluated_mutants = set()  # Track unique mutants
        
    def initialize_population(self) -> List[Mutant]:
        """Create initial random population of mutants"""
        population = []
        
        for _ in range(self.population_size):
            # Randomly select mutation positions (no duplicates)
            positions = sorted(random.sample(range(self.protein_length), 
                                           self.n_mutations))
            
            # Randomly select amino acids for each position
            mutations = [random.choice(self.amino_acids) 
                        for _ in range(self.n_mutations)]
            
            mutant = Mutant(positions=positions, mutations=mutations)
            population.append(mutant)
        
        return population
    
    def evaluate_fitness(self, mutant: Mutant) -> float:
        """
        Evaluate fitness using ML model prediction.
        Lower ΔΔG = more stable = higher fitness (we negate for minimization)
        """
        if mutant.fitness is not None:
            return mutant.fitness
        
        # Extract features using provided function
        features = self.feature_extractor(mutant)
        
        # Predict ΔΔG using ML model
        ddg_prediction = self.ml_model.predict([features])[0]
        
        # Convert to fitness (negative ΔΔG = stabilizing = better)
        # We want to MINIMIZE ΔΔG, so fitness = -ΔΔG
        fitness = -ddg_prediction
        mutant.fitness = fitness
        
        return fitness
    
    def evaluate_population(self, population: List[Mutant]) -> None:
        """Evaluate fitness for entire population"""
        for mutant in population:
            self.evaluate_fitness(mutant)
            self.evaluated_mutants.add(mutant)
    
    def tournament_selection(self, population: List[Mutant]) -> Mutant:
        """Select parent using tournament selection"""
        tournament = random.sample(population, self.tournament_size)
        winner = max(tournament, key=lambda m: m.fitness)
        return winner
    
    def crossover(self, parent1: Mutant, parent2: Mutant) -> Tuple[Mutant, Mutant]:
        """
        Perform crossover between two parents.
        Exchange mutation positions and corresponding amino acids.
        """
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Single-point crossover
        crossover_point = random.randint(1, self.n_mutations - 1)
        
        # Create offspring
        child1_positions = (parent1.positions[:crossover_point] + 
                          parent2.positions[crossover_point:])
        child1_mutations = (parent1.mutations[:crossover_point] + 
                          parent2.mutations[crossover_point:])
        
        child2_positions = (parent2.positions[:crossover_point] + 
                          parent1.positions[crossover_point:])
        child2_mutations = (parent2.mutations[:crossover_point] + 
                          parent1.mutations[crossover_point:])
        
        # Ensure positions are sorted and unique
        child1_positions, child1_mutations = self._fix_duplicate_positions(
            child1_positions, child1_mutations)
        child2_positions, child2_mutations = self._fix_duplicate_positions(
            child2_positions, child2_mutations)
        
        child1 = Mutant(positions=child1_positions, mutations=child1_mutations)
        child2 = Mutant(positions=child2_positions, mutations=child2_mutations)
        
        return child1, child2
    
    def _fix_duplicate_positions(self, positions: List[int], 
                                 mutations: List[str]) -> Tuple[List[int], List[str]]:
        """Handle duplicate positions after crossover"""
        if len(set(positions)) == len(positions):
            return sorted(positions), [m for _, m in sorted(zip(positions, mutations))]
        
        # If duplicates exist, randomly replace them
        seen = set()
        new_positions = []
        new_mutations = []
        
        for pos, mut in zip(positions, mutations):
            if pos not in seen:
                new_positions.append(pos)
                new_mutations.append(mut)
                seen.add(pos)
            else:
                # Replace with new random position
                available = set(range(self.protein_length)) - seen
                if available:
                    new_pos = random.choice(list(available))
                    new_positions.append(new_pos)
                    new_mutations.append(mut)
                    seen.add(new_pos)
        
        # Sort by position
        sorted_pairs = sorted(zip(new_positions, new_mutations))
        return [p for p, _ in sorted_pairs], [m for _, m in sorted_pairs]
    
    def mutate(self, mutant: Mutant) -> Mutant:
        """
        Apply mutation operator.
        Can change either position or amino acid substitution.
        """
        if random.random() > self.mutation_rate:
            return mutant
        
        new_mutant = deepcopy(mutant)
        mutation_type = random.choice(['position', 'amino_acid'])
        
        if mutation_type == 'position':
            # Change one mutation position
            idx = random.randint(0, self.n_mutations - 1)
            available_positions = (set(range(self.protein_length)) - 
                                 set(new_mutant.positions))
            if available_positions:
                new_mutant.positions[idx] = random.choice(list(available_positions))
                new_mutant.positions.sort()
        else:
            # Change one amino acid
            idx = random.randint(0, self.n_mutations - 1)
            new_mutant.mutations[idx] = random.choice(self.amino_acids)
        
        new_mutant.fitness = None  # Reset fitness
        return new_mutant
    
    def calculate_diversity(self, population: List[Mutant]) -> float:
        """Calculate population diversity based on unique genotypes"""
        unique_mutants = len(set(population))
        return unique_mutants / len(population)
    
    def evolve(self, verbose: bool = True) -> List[Mutant]:
        """
        Main evolution loop.
        
        Returns:
        --------
        List of top mutants ranked by fitness
        """
        # Initialize population
        self.population = self.initialize_population()
        self.evaluate_population(self.population)
        
        for generation in range(self.n_generations):
            # Sort by fitness (descending)
            self.population.sort(key=lambda m: m.fitness, reverse=True)
            
            # Track statistics
            best_fitness = self.population[0].fitness
            avg_fitness = np.mean([m.fitness for m in self.population])
            diversity = self.calculate_diversity(self.population)
            
            self.history['generation'].append(generation)
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['diversity'].append(diversity)
            
            if verbose and generation % 5 == 0:
                print(f"Generation {generation}: "
                      f"Best ΔΔG = {-best_fitness:.3f}, "
                      f"Avg ΔΔG = {-avg_fitness:.3f}, "
                      f"Diversity = {diversity:.3f}")
            
            # Elitism: preserve top individuals
            n_elite = int(self.elitism_ratio * self.population_size)
            new_population = self.population[:n_elite]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            self.population = new_population[:self.population_size]
            
            # Evaluate new individuals
            self.evaluate_population(self.population)
        
        # Final sort
        self.population.sort(key=lambda m: m.fitness, reverse=True)
        
        if verbose:
            print(f"\nEvolution complete!")
            print(f"Total unique mutants evaluated: {len(self.evaluated_mutants)}")
            print(f"Best mutant ΔΔG: {-self.population[0].fitness:.3f}")
        
        return self.population
    
    def get_top_mutants(self, n: int = 25) -> pd.DataFrame:
        """
        Get top N mutants as a ranked DataFrame.
        
        Returns:
        --------
        DataFrame with columns: rank, positions, mutations, predicted_ddg
        """
        top_mutants = self.population[:n]
        
        results = []
        for rank, mutant in enumerate(top_mutants, 1):
            mutation_str = ','.join([f"{pos}{mut}" 
                                    for pos, mut in 
                                    zip(mutant.positions, mutant.mutations)])
            results.append({
                'rank': rank,
                'positions': mutant.positions,
                'mutations': mutant.mutations,
                'mutation_notation': mutation_str,
                'predicted_ddg': -mutant.fitness,  # Convert back to ΔΔG
                'fitness': mutant.fitness
            })
        
        return pd.DataFrame(results)
    
    def get_history_df(self) -> pd.DataFrame:
        """Get evolution history as DataFrame for plotting"""
        return pd.DataFrame(self.history)


# Monte Carlo baseline for comparison
class MonteCarloSearch:
    """
    Random Monte Carlo search baseline for comparison with GA.
    """
    
    def __init__(self,
                 ml_model,
                 feature_extractor: Callable,
                 protein_length: int,
                 n_mutations: int = 2,
                 n_samples: int = 5000,
                 amino_acids: List[str] = None):
        """Initialize Monte Carlo search"""
        self.ml_model = ml_model
        self.feature_extractor = feature_extractor
        self.protein_length = protein_length
        self.n_mutations = n_mutations
        self.n_samples = n_samples
        
        if amino_acids is None:
            self.amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        else:
            self.amino_acids = amino_acids
        
        self.evaluated_mutants = []
    
    def search(self, verbose: bool = True) -> List[Mutant]:
        """
        Perform random search.
        
        Returns:
        --------
        List of mutants ranked by fitness
        """
        self.evaluated_mutants = []
        seen = set()
        
        for i in range(self.n_samples):
            # Generate random mutant
            positions = sorted(random.sample(range(self.protein_length), 
                                           self.n_mutations))
            mutations = [random.choice(self.amino_acids) 
                        for _ in range(self.n_mutations)]
            
            mutant = Mutant(positions=positions, mutations=mutations)
            
            # Skip duplicates
            if mutant in seen:
                continue
            seen.add(mutant)
            
            # Evaluate
            features = self.feature_extractor(mutant)
            ddg_prediction = self.ml_model.predict([features])[0]
            mutant.fitness = -ddg_prediction
            
            self.evaluated_mutants.append(mutant)
            
            if verbose and (i + 1) % 1000 == 0:
                best_so_far = min([m.fitness for m in self.evaluated_mutants])
                print(f"Samples: {i+1}/{self.n_samples}, "
                      f"Best ΔΔG so far: {-best_so_far:.3f}")
        
        # Sort by fitness
        self.evaluated_mutants.sort(key=lambda m: m.fitness, reverse=True)
        
        if verbose:
            print(f"\nMonte Carlo search complete!")
            print(f"Unique mutants evaluated: {len(self.evaluated_mutants)}")
            print(f"Best mutant ΔΔG: {-self.evaluated_mutants[0].fitness:.3f}")
        
        return self.evaluated_mutants
    
    def get_top_mutants(self, n: int = 25) -> pd.DataFrame:
        """Get top N mutants as DataFrame"""
        top_mutants = self.evaluated_mutants[:n]
        
        results = []
        for rank, mutant in enumerate(top_mutants, 1):
            mutation_str = ','.join([f"{pos}{mut}" 
                                    for pos, mut in 
                                    zip(mutant.positions, mutant.mutations)])
            results.append({
                'rank': rank,
                'positions': mutant.positions,
                'mutations': mutant.mutations,
                'mutation_notation': mutation_str,
                'predicted_ddg': -mutant.fitness,
                'fitness': mutant.fitness
            })
        
        return pd.DataFrame(results)


# Example usage template
if __name__ == "__main__":
    """
    Example of how to use the Genetic Algorithm.
    
    You need to provide:
    1. Your trained ML model
    2. A feature extraction function that converts Mutant -> feature vector
    """
    
    # Dummy example - replace with your actual model and feature extractor
    from sklearn.ensemble import RandomForestRegressor
    
    # Create dummy model (replace with your trained model)
    dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randn(100)
    dummy_model.fit(X_dummy, y_dummy)
    
    # Define feature extractor (replace with your actual feature extraction)
    def dummy_feature_extractor(mutant: Mutant) -> np.ndarray:
        """
        Example feature extractor.
        In your actual implementation, this should:
        - Extract BLOSUM62 scores
        - Calculate hydrophobicity
        - Get structural features (SASA, secondary structure)
        - Include physics-based features (FoldX ΔΔG if available)
        """
        # Dummy: just create random features
        features = np.random.randn(10)
        return features
    
    # Initialize GA
    print("=== Genetic Algorithm ===")
    ga = ProteinGeneticAlgorithm(
        ml_model=dummy_model,
        feature_extractor=dummy_feature_extractor,
        protein_length=100,  # Length of your protein
        n_mutations=2,       # Double mutants
        population_size=50,
        n_generations=20,
        mutation_rate=0.1,
        crossover_rate=0.7
    )
    
    # Run evolution
    best_mutants = ga.evolve(verbose=True)
    
    # Get top 25 mutants
    top_25 = ga.get_top_mutants(25)
    print("\n=== Top 25 Mutants ===")
    print(top_25.head(10))
    
    # Get evolution history
    history = ga.get_history_df()
    print("\n=== Evolution History ===")
    print(history.tail())
    
    # Monte Carlo baseline
    print("\n\n=== Monte Carlo Baseline ===")
    mc = MonteCarloSearch(
        ml_model=dummy_model,
        feature_extractor=dummy_feature_extractor,
        protein_length=100,
        n_mutations=2,
        n_samples=1000  # Same number of evaluations as GA
    )
    
    mc_results = mc.search(verbose=True)
    mc_top_25 = mc.get_top_mutants(25)
    print("\n=== Top 25 Mutants (MC) ===")
    print(mc_top_25.head(10))
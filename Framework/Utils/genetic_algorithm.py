import numpy as np


def selection(pop, scores, k=3):
    selection_x = np.randint(len(pop))
    for x in np.random.randint(0, len(pop), k - 1):
        if scores(x) < scores[selection_x]:
            selection_x = x
    return pop[selection_x]


def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    if np.rand() < r_cross:
        pt = np.randint(1, len(p1) - 2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def mutation(b_string, r_mut):
    for i in range(len(b_string)):
        if np.rand() < r_mut:
            b_string[i] = 1 - b_string[i]


def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    pop = [np.randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    best, best_eval = 0, objective(pop[0])
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.2f" % (gen, pop[i], scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]

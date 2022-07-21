import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time

from test_data import generate_data as grd

# params for test purposes
k = 2  # number of cluster
var = 5  # var in RFB kernel
iteration_counter = 0
data_file = grd.generate_xor(k)  # choice : generate_random, generate_tor, generate_xor, generate_linear
data_testing = np.loadtxt(data_file, delimiter=" ")


# Initiate cluster kernels
def init_cluster(data_input, n_cluster):
    list_cluster_member = [[] for i in range(n_cluster)]
    shuffled_data_in = data_input
    np.random.shuffle(shuffled_data_in)
    for i in range(0, data_input.shape[0]):
        list_cluster_member[i % n_cluster].append(data_input[i, :])

    return list_cluster_member


# New kernel position
def rbf_kernel(data1, data2, sigma):
    delta = abs(np.subtract(data1, data2))
    squared_euclidean = (np.square(delta).sum(axis=1)) # -> Mauvaise traduction en C++ // A revoir
    result = np.exp(-(squared_euclidean) / (2 * sigma ** 2))
    return result


# Compute third term of the mathematical function
def third_term(cluster_member):
    result = 0
    for i in range(0, cluster_member.shape[0]):
        for j in range(0, cluster_member.shape[0]):
            result = result + rbf_kernel(cluster_member[i, :], cluster_member[j, :], var)
    result = result / (cluster_member.shape[0] ** 2)

    return result

# Compute second term of the mathematical function
def second_term(data_I, cluster_member):
    result = 0
    for i in range(0, cluster_member.shape[0]):
        result = result + rbf_kernel(data_I, cluster_member[i, :], var)
    result = 2 * result / cluster_member.shape[0]

    return result


# Display
def plot_result(list_cluster_members, centroid, iteration, converged):
    n = list_cluster_members.__len__()
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    plt.figure("result")
    plt.clf()
    plt.title("Iteration-" + iteration)
    for i in range(n):
        col = next(color)
        cluster_member = np.asmatrix(list_cluster_members[i])
        plt.scatter(np.ravel(cluster_member[:, 0]), np.ravel(cluster_member[:, 1]), marker=".", s=100, color=col)
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    for i in range(n):
        col = next(color)
        plt.scatter(np.ravel(centroid[i, 0]), np.ravel(centroid[i, 1]),
                    marker="*", s=400,
                    color=col, edgecolors="black")
    if (converged == 0):
        plt.ion()
        plt.show()
        plt.pause(0.1)
    if (converged == 1):
        plt.show(block=True)


# Main function
def k_means_kernel(data):
    global iteration_counter
    member_init = init_cluster(data, k)
    n_cluster = member_init.__len__()
    # looping until converged
    while (True):
        # Calculate centroid, only for visualization purpose
        centroid = np.ndarray(shape=(0, data.shape[1]))
        for i in range(0, n_cluster):
            cluster_member = np.asmatrix(member_init[i])
            centroid_cluster = cluster_member.mean(axis=0)
            centroid = np.concatenate((centroid, centroid_cluster), axis=0)
        # Plot result in every iteration
        plot_result(member_init, centroid, str(iteration_counter), 0)
        old_time = np.around(time.time(), decimals=0)
        kernel_result_cluster_all_cluster = np.ndarray(shape=(data.shape[0], 0))
        # Assign data to cluster for closest centroid
        for i in range(0, n_cluster):  # repeat for all cluster
            term_3 = third_term(np.asmatrix(member_init[i]))
            matrix_term_3 = np.repeat(term_3, data.shape[0], axis=0);
            matrix_term_3 = np.asmatrix(matrix_term_3)
            matrix_term_2 = np.ndarray(shape=(0, 1))
            for j in range(0, data.shape[0]):  # repeat for all data
                term_2 = second_term(data[j, :], np.asmatrix(member_init[i]))
                matrix_term_2 = np.concatenate((matrix_term_2, term_2), axis=0)
            matrix_term_2 = np.asmatrix(matrix_term_2)
            kernel_result_cluster_I = np.add(-1 * matrix_term_2, matrix_term_3)
            kernel_result_cluster_all_cluster = \
                np.concatenate((kernel_result_cluster_all_cluster, kernel_result_cluster_I), axis=1)
        cluster_matrix = np.ravel(np.argmin(np.matrix(kernel_result_cluster_all_cluster), axis=1))
        list_cluster_member = [[] for l in range(k)]
        for i in range(0, data.shape[0]):  # assign data to cluster regarding cluster matrix
            list_cluster_member[cluster_matrix[i].item()].append(data[i, :])
        for i in range(0, n_cluster):
            print("Cluster member numbers-", i, ": ", list_cluster_member[0].__len__())
        # Break when converged
        bool_acc = True
        for m in range(0, n_cluster):
            prev = np.asmatrix(member_init[m])
            current = np.asmatrix(list_cluster_member[m])
            if (prev.shape[0] != current.shape[0]):
                bool_acc = False
                break
            if (prev.shape[0] == current.shape[0]):
                bool_per_cluster = (prev == current).all()
            bool_acc = bool_acc and bool_per_cluster
            if (bool_acc == False):
                break
        if (bool_acc == True):
            break
        iteration_counter += 1
        # Update new cluster member
        member_init = list_cluster_member
        new_time = np.around(time.time(), decimals=0)
        print("Iteration-", iteration_counter, ": ", new_time - old_time, " seconds")

    return list_cluster_member, centroid


if __name__ == "__main__":
    cluster_result, centroid = k_means_kernel(data_testing)
    plot_result(cluster_result, centroid, str(iteration_counter) + ' (converged)', 1)
    print("Converged !")

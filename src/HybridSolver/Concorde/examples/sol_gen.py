import random


input_file = "./TSPLIB/u1060.sol"
output_file = "../src/PROBS/probs_"

def read_input():
    n_nodes = 0
    tour = []

    with open(input_file, "r") as file:
        n_nodes = int(file.readline())
        for line in file:
            tour.extend(map(int, line.split()))
        tour.append(tour[0])

    return n_nodes, tour


def create_sol(n_nodes, tour):
    sol = {}
    for i in range(len(tour) - 1):
        src = tour[i]
        dest = tour[i + 1]
        prob = round(random.uniform(0.9, 1), 4)
        sol[(int(src), int(dest))] = prob

    sequence = []
    for n1 in range(0, n_nodes):
        for n2 in range(n1 + 1, n_nodes):

            if (n1, n2) in sol:
                sequence.append((n1, n2, sol[(n1, n2)]))
            elif (n2, n1) in sol:
                sequence.append((n1, n2, sol[(n2, n1)]))
            else:
                prob = random.random() * 0.05
                sequence.append((n1, n2, prob))

    return sequence

def write_file_sol(output):
    global output_file
    instance_name = input_file.split("/")[-1].split(".")[0]
    output_file = f"{output_file}{instance_name}.txt"

    with open(output_file, "w") as file:
        for n1, n2, prob in output:
            file.write(f"{n1} {n2} {prob}\n")

    print(f"Sequence saved to {output_file}")


if __name__ == "__main__":
    n_nodes, tour = read_input()
    output = create_sol(n_nodes, tour)
    write_file_sol(output)

from SemanticNetsAgent import SemanticNetsAgent
import time

def time_function(my_function):
    # Timing the function
    start_time = time.time()  # Get the current time

    my_function()  # Call your function here

    end_time = time.time()  # Get the current time again

    execution_time = end_time - start_time  # Calculate the execution time

    print(f"Execution time: {execution_time} seconds")


def test():
    #This will test your SemanticNetsAgent
	#with seven initial test cases.
    test_agent = SemanticNetsAgent()

    time1 = time.time()
    print(test_agent.solve(1, 1))
    print("\n")
    print(test_agent.solve(2, 2))
    print("\n")
    print(test_agent.solve(3, 3))
    print("\n")
    print(test_agent.solve(5, 3))
    print("\n")
    print(test_agent.solve(6, 3))
    print("\n")
    print(test_agent.solve(7, 3))
    print("\n")
    print(test_agent.solve(5, 5))

def plot_time():
    #This will plot the time it takes for your agent
    #to solve the problem for different initial states.
    cases = [(1, 1), (2, 2), (3, 3), (5, 3), (6, 3), (7, 3), (5, 5)]
    # cases = [(1, 1), (2, 2), (3, 3), (5, 3), (6, 3), (7, 3), (8, 3),(9, 3), (10, 3), (11, 3), (12, 3), (13, 3), (14, 3), (15, 3)]
    times = []
    test_agent = SemanticNetsAgent()
    for case in cases:
        start_time = time.time()
        test_agent.solve(case[0], case[1])
        end_time = time.time()
        times.append(end_time - start_time)
    import matplotlib.pyplot as plt
    plt.plot([sum(c) for c in cases], times)
    plt.xlabel("# or animals")
    plt.ylabel("Time taken to solve problem")
    plt.savefig("Figure 1: Runtime vs. Total Animals.png")
    plt.clf()

def plot_time_more_cases():
    cases = [(1, 1), (2, 2), (3, 3), (5, 3), (6, 3), (7, 3), (8, 3),(9, 3), (10, 3), (11, 3), (12, 3), (13, 3), (14, 3), (15, 3)]
    times = []
    test_agent = SemanticNetsAgent()
    for case in cases:
        start_time = time.time()
        test_agent.solve(case[0], case[1])
        end_time = time.time()
        times.append(end_time - start_time)
    import matplotlib.pyplot as plt
    plt.plot([sum(c) for c in cases], times)
    plt.xlabel("# or animals")
    plt.ylabel("Time taken to solve problem")
    plt.savefig("Figure 2: Runtime vs. Total Animals.png")
    plt.clf()

if __name__ == "__main__":
    test()
    plot_time()
    plot_time_more_cases()
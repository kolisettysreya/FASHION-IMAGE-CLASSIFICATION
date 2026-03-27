import matplotlib.pyplot as plt
def plot_optimizer_results(results):
    optimizers = list(results.keys())
    accuracies = [results[opt]["accuracy"] for opt in optimizers]
    losses = [results[opt]["loss"] for opt in optimizers]

    plt.figure()

    plt.subplot(1,2,1)
    plt.bar(optimizers, accuracies, color="skyblue")
    plt.title("Optimizer vs Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0,100)
    plt.grid(axis="y",linestyle='--',alpha=0.7)

    plt.subplot(1,2,2)
    plt.bar(optimizers, losses, color='salmon')
    plt.title("Optimizer vs Loss")
    plt.ylabel("Loss")
    plt.grid(axis="y",linestyle='--',alpha=0.7)

    plt.tight_layout()
    plt.show()
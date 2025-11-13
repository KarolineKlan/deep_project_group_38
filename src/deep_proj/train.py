from deep_proj.model import Model
from deep_proj.data import MyDataset
from deep_proj.visualize import plot_training_progress, plot_final_results, plot_dirichlet_simplex_nD

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
    
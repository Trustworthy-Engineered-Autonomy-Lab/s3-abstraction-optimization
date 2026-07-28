import pickle as pkl
import main_mountain_car_model_checking
import cegar_loop_mc
import abstraction
import refine_whole_space

if __name__ == "__main__":

    with open("cegar/mountain-car/mountaincar_60x60.pkl", "rb") as f:
        data = pkl.load(f)

    print(type(data))
    
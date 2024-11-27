import numpy as np


# Not used anymore
# can be used to print out results to txt file after running
def write_res(args, tr, val, te):
    result_txt = open(
        "./stat_res/{}_{}_mean_results_split50.txt".format(args['task'], args['model']), "w")

    result_txt.write(
        f"Training mean: {np.mean(tr, axis=0)}\nValidation mean: {np.mean(val, axis=0)}\nTesting mean: {np.mean(te, axis=0)}"
    )
    result_txt.close()

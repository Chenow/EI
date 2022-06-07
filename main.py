import pickle
# load : get the data from file
data = pickle.load(open("données/Irish_synchrone_sample2_to_predict_na.pkl", "rb"))
print(data)
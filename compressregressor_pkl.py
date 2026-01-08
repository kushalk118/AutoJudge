import pickle
import gzip

input_file = "regressor.pkl"       # <-- your existing pkl file name
output_file = "regressor.pkl.gz"   # <-- new compressed file

with open(input_file, "rb") as f:
    data = pickle.load(f)

with gzip.open(output_file, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Compression completed")

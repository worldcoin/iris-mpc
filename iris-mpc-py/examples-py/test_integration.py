import os
from iris_mpc_py import PyIrisCode, PyPlaintextStore, PyGraphStore, PyHnswSearcher

vector_filename = "./data/vector.ndjson"
graph_filename = "./data/graph1.dat"
if not os.path.exists("./data/"):
    os.makedirs("./data/")

print("Generating 100k uniform random iris codes...")
vector_init = PyPlaintextStore()
iris0 = PyIrisCode.uniform_random()
iris_id = vector_init.insert(iris0)
for i in range(1,100000):
	vector_init.insert(PyIrisCode.uniform_random())

# write vector store to file
print("Writing vector store to file...")
vector_init.write_to_ndjson(vector_filename)

print("Generating HNSW graphs for 10k imported iris codes...")
hnsw = PyHnswSearcher(32, 64, 32)
vector1 = PyPlaintextStore()
graph1 = PyGraphStore()
hnsw.fill_from_ndjson_file(vector_filename, vector1, graph1, 10000)

print("Imported length:", vector1.len())

retrieved_iris = vector1.get(iris_id)
print("Retrieved iris0 base64 == original iris0 base64:", iris0.code.to_base64() == retrieved_iris.code.to_base64() and iris0.mask.to_base64() == retrieved_iris.mask.to_base64())

query = PyIrisCode.uniform_random()
print("Search for random query iris code:", hnsw.search(query, vector1, graph1))

# write graph store to file
print("Writing graph store to file...")
graph1.write_to_bin(graph_filename)

# read HNSW graphs from disk
print("Reading vector and graph stores from file...")
vector2 = PyPlaintextStore.read_from_ndjson(vector_filename, 10000)
graph2 = PyGraphStore.read_from_bin(graph_filename)

print("Search for random query iris code:", hnsw.search(query, vector2, graph2))

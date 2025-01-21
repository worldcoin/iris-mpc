# Python Bindings

This package provides Python bindings for some functionalities in the `iris-mpc` workspace, currently focused on execution of the HNSW k-nearest neighbors graph search algorithm over plaintext iris codes for testing and data analysis.  For compatibility, compilation of this crate is disabled from the workspace root, but enabled from within the crate subdirectory via the Cargo default feature flag `enable`.

## Installation

Installation of Python bindings from the PyO3 library code can be accomplished using the Maturin Python package as follows:

- Install Maturin in the target Python environment, e.g. the venv used for data analysis, using `pip install maturin`

- Optionally install `patchelf` library with `pip install patchelf` for support for patching wheel files that link other shared libraries

- Build and install current bindings as a module in the current Python environment by navigating to the `iris-mpc-py` directory and running `maturin develop --release`

- Build a wheel file suitable for installation using `pip install` by instead running `maturin build --release`; the `.whl` file is specific to the building architecture and Python version, and can be found in `iris_mpc/target/wheels` directory

See the [Maturin User Guide Tutorial](https://www.maturin.rs/tutorial#build-and-install-the-module-with-maturin-develop) for additional details.

## Usage

Once successfully installed, the native rust module `iris_mpc_py` can be imported in your Python environment as usual with `import iris_mpc_py`.  Example usage:

```python
from iris_mpc_py import PyHnswSearcher, PyPlaintextStore, PyGraphStore, PyIrisCode

hnsw = PyHnswSearcher(32, 64, 32) # M, ef_constr, ef_search
vector = PyPlaintextStore()
graph = PyGraphStore()

hnsw.fill_uniform_random(1000, vector, graph)

iris = PyIrisCode.uniform_random()
iris_id = hnsw.insert(iris, vector, graph)
print("Inserted iris id:", iris_id)

nearest_id, nearest_dist = hnsw.search(iris, vector, graph)
print("Nearest iris id:", nearest_id) # should be iris_id
print("Nearest iris distance:", nearest_dist) # should be 0.0
```

To write the HNSW vector and graph indices to file and read them back:

```python
hnsw.write_to_json("searcher.json")
vector.write_to_ndjson("vector.ndjson")
graph.write_to_bin("graph.dat")

hnsw2 = PyHnswSearcher.read_from_json("searcher.json")
vector2 = PyPlaintextStore.read_from_ndjson("vector.ndjson")
graph2 = PyGraphStore.read_from_bin("graph.dat")
```

As an efficiency feature, the data from the vector store is read in a streamed fashion.  This means that for a large database of iris codes, the first `num` can be read from file without loading the entire database into memory.  This can be used in two ways; first, a vector store can be initialized from the large databse file for use with a previously generated HNSW index:

```python
# Serialized HNSW graph constructed from the first 10k entries of database file
vector = PyPlaintextStore.read_from_ndjson("large_vector_database.ndjson", 10000)
graph = PyGraphStore.read_from_bin("graph.dat")
```

Second, to construct an HNSW index dynamically from streamed database entries:

```python
hnsw = PyHnswSearcher(32, 64, 32)
vector = PyPlaintextStore()
graph = PyGraphStore()
hnsw.fill_from_ndjson_file("large_vector_database.ndjson", vector, graph, 10000)
```

To generate a vector database directly for use in this way:

```python
# Generate 100k uniform random iris codes
vector_init = PyPlaintextStore()
for i in range(1,100000):
	vector_init.insert(PyIrisCode.uniform_random())
vector_init.write_to_ndjson("vector.ndjson")
```

Basic interoperability with Open IRIS iris templates is provided by way of a common base64 encoding scheme, provided by the `iris.io.dataclasses.IrisTemplate` methods `serialize` and `deserialize`.  These methods use a base64 encoding of iris code and mask code arrays represented as a Python `dict` with base64-encoded fields `iris_codes`, `mask_codes`, and a version string `iris_code_version` to check for compatibility.  The `PyIrisCode` class interacts with this representation as follows:

```python
serialized_iris_code = {
	"iris_codes": "...",
	"mask_codes": "...",
	"iris_code_version": "1.0",
}

iris = PyIrisCode.from_open_iris_template_dict(serialized_iris_code)
reserialized_iris_code = iris.to_open_iris_template_dict("1.0")
```

Note that the `to_open_iris_template_dict` method takes an optional argument which fills the `iris_code_version` field of the resulting Python `dict` since the `PyIrisCode` object does not preserve this data.

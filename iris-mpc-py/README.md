# Python Bindings

This package provides Python bindings for some functionalities in the `iris-mpc` workspace, currently focused on execution of the HNSW k-nearest neighbors graph search algorithm over plaintext iris codes for testing and data analysis.

## Installation

Installation of Python bindings from the PyO3 library code can be accomplished using the Maturin Python package as follows:

- Install Maturin in the target Python environment, e.g. the venv used for data analysis, using `pip install maturin`

- Optionally install `patchelf` library with `pip install patchelf` for support for patching wheel files that link other shared libraries

- Build and install current bindings as a module in the current Python environment by navigating to the `iris-mpc-py` directory and running `maturin develop --release --features enable`

- Build a wheel file suitable for installation using `pip install` by instead running `maturin build --release --features enable`; the `.whl` file is specific to the building architecture and Python version, and can be found in `iris_mpc/target/wheels` directory

See the [Maturin User Guide Tutorial](https://www.maturin.rs/tutorial#build-and-install-the-module-with-maturin-develop) for additional details.

## Usage

Once successfully installed, the native rust module `iris_mpc_py` can be imported in your Python environment as usual with `import iris_mpc_py`.  Example usage:

```python
from iris_mpc_py import PyHnsw, PyIrisCode, PyIrisCodeArray

hnsw = PyHnsw(32, 32) # M, ef
hnsw.fill_uniform_random(1000)

code = PyIrisCodeArray.uniform()
mask = PyIrisCodeArray.uniform()

iris = PyIrisCode(code, mask)

iris_id = hnsw.insert(iris)
print("Inserted iris id:", iris_id)

nearest_id, nearest_dist = hnsw.search(iris)
print("Nearest iris id:", nearest_id) # should be iris_d
print("Nearest iris distance:", nearest_dist) # should be 0.0

hnsw.write_to_file("hnsw_example.dat")
hnsw_again = PyHnsw.read_from_file("hnsw_example.dat")
```

Basic interoperability with Open IRIS iris templates is implemented but not yet tested.  Usage should be something like the following:

```python
# Type of object is: iris.io.dataclasses.IrisTemplate
oi_template = TEMPLATE_FROM_OPEN_IRIS

# transform directly from template object
iris_method_1 = PyIrisCode.from_open_iris_template(oi_template)

# or, using lower level primitives
b64_encoding = oi_template.serialize()

code = PyIrisCodeArray.from_b64(b64_encoding["iris_codes"])
mask = PyIrisCodeArray.from_b64(b64_encoding["mask_codes"])
iris_method_2 = PyIrisCode(code, mask)

# You can now use the imported iris code object as demonstrated above
```
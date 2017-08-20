Parallel Implementation for Partitioning HTS Read Datasets and Associated de Bruijn Graph
========================================================================

## Code organization

### Note

- The current version corresponds to an alpha release of the software
- Additional usage instructions are to follow
- External (third-party) softwares, included in the [`ext/`](`ext/`) directory, are covered under their repective licenses

### Dependencies

- `cmake` version >= 2.6
- `g++` (version 4.8.1+)
- an `MPI` implementation supporting `MPI-2` or `MPI-3`
- external (third-party) softwares are included in the [`ext/`](`ext/`) directory of this project

### Download and compile


The repository and external submodules can be downloaded using the recursive clone.

```sh
git clone --recursive <URL>
```

Set up external submodule (ext/bliss/ext/sparsehash).

```sh
./configure
make
```

Compile using the cmake tool.

```sh
mkdir build_directory && cd build_directory
cmake ../new_readpartitioning
make -j <THREAD COUNT>
```

### Run

Inside the build directory, 

```sh
mpirun -np <COUNT OF PROCESSES> ./bin/<EXECUTABLE> <ARGUMENTS>
```

# Label_Propagation_SYCL
Label propagation algorithm in sycl

docker pull intel/oneapi-hpckit:latest
docker run -it --rm -v $(pwd):/workspace -w /workspace intel/oneapi-hpckit:latest /bin/bash
dpcpp -o main main.cpp
./main

To check the version of the compiler: which dpcpp

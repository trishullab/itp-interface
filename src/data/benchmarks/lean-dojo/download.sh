if [[ ! -d "src/itp_interface/scripts" ]]; then
    # Raise an error if the scripts directory is not present
    echo "Please run this script from the root of the repository, cannot find src/scripts"
    exit 1
fi
if [[ ! -d "src/.log" ]]; then
    mkdir src/.log
fi
if [[ ! -d "src/.log/benchmarks" ]]; then
    mkdir src/.log/benchmarks
fi
# Check if the file is already downloaded
if [[ ! -f "src/.log/benchmarks/leandojo_benchmark_v1.tar.gz" ]]; then
    echo "Downloading leandojo_benchmark_v1.tar.gz..."
    wget "https://zenodo.org/records/8016386/files/leandojo_benchmark_v1.tar.gz?download=1" -O src/.log/benchmarks/leandojo_benchmark_v1.tar.gz
    echo "Downloaded leandojo_benchmark_v1.tar.gz successfully!"
else
    echo "leandojo_benchmark_v1.tar.gz already exists, skipping download..."
fi
# Check if the file is already extracted
if [[ ! -d "src/.log/benchmarks/leandojo_benchmark_v1" ]]; then
    echo "Extracting leandojo_benchmark_v1.tar.gz..."
    tar -xzf src/.log/benchmarks/leandojo_benchmark_v1.tar.gz -C src/.log/benchmarks
    echo "Extracted leandojo_benchmark_v1.tar.gz successfully!"
else
    echo "leandojo_benchmark_v1.tar.gz already extracted, skipping extraction..."
fi
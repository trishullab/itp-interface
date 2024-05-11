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
file_link="https://zenodo.org/records/10929138/files/leandojo_benchmark_4.tar.gz?download=1"
file_version="leandojo_benchmark_4"
file_name="$file_version.tar.gz"
if [[ ! -f "src/.log/benchmarks/$file_name" ]]; then
    echo "Downloading $file_name..."
    wget $file_link -O src/.log/benchmarks/$file_name
    echo "Downloaded $file_name successfully!"
else
    echo "$file_name already exists, skipping download..."
fi
# Check if the file is already extracted
if [[ ! -d "src/.log/benchmarks/$file_version" ]]; then
    echo "Extracting $file_name..."
    mkdir src/.log/benchmarks/$file_version
    tar -xzf src/.log/benchmarks/$file_name -C src/.log/benchmarks/$file_version
    echo "Extracted $file_name successfully!"
else
    echo "$file_name already extracted, skipping extraction..."
fi
#!/bin/bash

# Check if the base directory is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <base_notebook_directory>"
    exit 1
fi

# Base directory for notebooks
base_dir=$1

# Define notebook directories or patterns
declare -a notebooks=(
    "$base_dir/examples/tools/*.ipynb"
    "$base_dir/examples/plotting/*.ipynb"
    "$base_dir/examples/image/*.ipynb"
    "$base_dir/examples/graph/*.ipynb"
    "$base_dir/tutorials/*.ipynb"
)

# Initialize an array to hold valid notebook paths
declare -a valid_notebooks

# Gather all valid notebook files from the patterns
echo "Gathering notebooks..."
for pattern in "${notebooks[@]}"; do
    for nb in $pattern; do
        if [[ -f "$nb" ]]; then  # Check if the file exists
            valid_notebooks+=("$nb")  # Add to the list of valid notebooks
        fi
    done
done

# Check if we have any notebooks to run
if [ ${#valid_notebooks[@]} -eq 0 ]; then
    echo "No notebooks found to run."
    exit 1
fi

# Echo the notebooks that will be run for clarity
echo "Preparing to run the following notebooks:"
for nb in "${valid_notebooks[@]}"; do
    echo "$nb"
done

# Initialize a flag to track the success of all commands
all_success=true

# Execute all valid notebooks
for nb in "${valid_notebooks[@]}"; do
    echo "Running $nb"
    jupytext -k squidpy --execute "$nb" || {
        echo "Failed to run $nb"
        all_success=false
    }
done

# Check if any executions failed
if [ "$all_success" = false ]; then
    echo "One or more notebooks failed to execute."
    exit 1
fi

echo "All notebooks executed successfully."
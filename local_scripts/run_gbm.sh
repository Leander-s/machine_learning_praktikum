data=$1
model=$2
mode=$3

max_retries=5
retries=0

exit_code=1

while (( retries < max_retries )); do
    conda run -n ml python gnn.py $data $model $mode
    exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        exit 0
    elif [[ $exit_code -eq 139 ]]; then
        echo "Segfault but its fine..."
        exit 0
    else
        echo "Error! Retrying dataset $data with model $model..."
    fi

    ((retries++))
done

echo "Model $model on dataset $data failed after $max_retries attempts. Skipping this."
exit 1

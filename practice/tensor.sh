DATA_DIR=/home/ken/our_project/practice/Flowers_data
python /home/ken/models/research/slim/download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
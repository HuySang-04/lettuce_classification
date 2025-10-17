#cd /path/to/lettuce_project    # vào thư mục project
#bash scripts/run_lettuce_disease_train.sh

python3 src/train_stage1.py \
    --data_path ../../data/Lettuce_disease_datasets \
    --epochs 30 \
    --batch_size 4 \
    --image_size 224 \
    --lr 1e-5 \
    --random_state 42 \
    --test_size 0.2 \
    --create_data True
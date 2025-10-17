#cd /path/to/lettuce_project    # vào thư mục project
#bash scripts/run_lettuce_health_train.sh

python3 src/lettuce_health_train.py \
    --data_path ../../data/plant-health \
    --epochs 30 \
    --batch_size 4 \
    --image_size 224 \
    --lr 1e-5 \
    --random_state 42 \
    --test_size 0.2

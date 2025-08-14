export PYTHONPATH="//home/fit02/dien-workspace/viamr/src:$PYTHONPATH"
echo "Running infer script..."

# python3 -m src.main \
#     --input_file "/home/fit02/dien-workspace/viamr/src/data/test.txt" \
#     --output_file "/home/fit02/dien-workspace/viamr/results.txt" \
#     --model_name "/home/fit02/dien-workspace/viamr/outputs/Qwen-1.7B-SFT" \
#     --my_test 1

python3 -m src.main \
    --input_file "/home/fit02/dien-workspace/viamr/src/data/public_test.txt" \
    --output_file "/home/fit02/dien-workspace/viamr/results.txt" \
    --model_name "/home/fit02/dien-workspace/viamr/outputs/Qwen-1.7B-SFT-2" \
    --my_test 0 \
    2>&1 | tee main.log
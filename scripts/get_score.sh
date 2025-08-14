export PYTHONPATH="//home/fit02/dien-workspace/viamr/src:$PYTHONPATH"
echo "Running get_score script..."

python3 -m src.get_score \
    --predict_file "/home/fit02/dien-workspace/viamr/src/data/test.txt" \
    --gold_file "/home/fit02/dien-workspace/viamr/results.txt"
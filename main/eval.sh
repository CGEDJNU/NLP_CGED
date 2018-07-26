# output predict

EXPERIMENT_NUM=1
EMBEDDING_DIM=30
HIDDEN_DIM=20
TRAINING_RATIO=0.2
python test.py --embedding_dim $EMBEDDING_DIM --hidden_dim $HIDDEN_DIM --checkpoint $1 --model_dir ../data/models/${EXPERIMENT_NUM}_${EMBEDDING_DIM}_${HIDDEN_DIM}_${TRAINING_RATIO}/

# get eval 
java -jar ../data/CGED-Test-2016/nlptea16cged.jar -r ../data/CGED-Test-2016/CGED16_HSK_Test_Output_.txt -t ../data/CGED-Test-2016/CGED16_HSK_Test_Truth.txt -o ../data/CGED-Test-2016/HSK_Evaluation.txt

# show results
cat ../data/CGED-Test-2016/HSK_Evaluation.txt

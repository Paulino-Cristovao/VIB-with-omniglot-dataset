

#
LR=0.001
NUMBER_SAMPLES=1

LATENT_LIST="16 32 64 128 256 512 1024"
EPOCHS_LIST="50 100 200"

for LATENT in $LATENT_LIST ; do
for EPOCHS in $EPOCHS_LIST ; do

mkdir -p $LATENT

python pretrain.py --save $LATENT --epochs $EPOCHS --lr $LR
python imprint.py --model $LATENT/$EPOCHS/pretrain_checkpoint/model_best.pth.tar --epochs $EPOCHS --numsample $NUMBER_SAMPLES --dimension $LATENT
#python imprint_ft.py --model $LATENT/$EPOCHS/pretrain_checkpoint/model_best.pth.tar --batch_size $LATENT --epochs $EPOCHS --lr $LR --numsample $NUMBER_SAMPLES

done;
done;



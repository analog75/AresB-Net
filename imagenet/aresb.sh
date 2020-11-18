batchsize=256
epochs=100
weightdecay=0.00001
lr=0.001
echo "batchsize: ${batchsize}"
echo "weightdecay: ${weightdecay}"
echo "lr: ${lr}"
nn_array=( 
aresbnet18   
#aresbnet34   
#aresbnet10   
)

for nn in ${nn_array[@]}
do
  echo "nn : ${nn}"
  #python aresb_main_adam.py --trainedfile ./train/${batchsize}/${nn}adamda2.best.pth.tar --epochs ${epochs} --batch-size ${batchsize} --arch ${nn} --trainout ./output/${batchsize}/${nn}adamda2train.out --valout ./output/${batchsize}/${nn}adamda2val.out --weight-decay ${weightdecay} --lr ${lr}
  python aresb_main_adam.py --evaluate --pretrained ./train/${batchsize}/pretrained/${nn}adamda2.best.pth.tar --epochs ${epochs} --batch-size ${batchsize} --arch ${nn} --trainout ./output/${batchsize}/${nn}train_eval.out --valout ./output/${batchsize}/${nn}val_eval.out --lr ${lr} 
done


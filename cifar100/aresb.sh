batchsize=256
weightdecay=0.00001
lr=0.1
echo "trainedfile: ${trainedfile}"
echo "batchsize: ${batchsize}"
echo "weightdecay: ${weightdecay}"
echo "lr: ${lr}"
nn_array=( 
aresbnet18   
#aresbnet10   
#aresbnet34   
)

for nn in ${nn_array[@]}
do
  echo "nn : ${nn}"
  #python aresb_main.py --trainedfile ./train/${batchsize}/${nn}.360epochs.pth.tar --batch-size ${batchsize} --arch ${nn} --outputfile ./output/${batchsize}/${nn}360epochs.out --weight-decay ${weightdecay} --lr ${lr}
  python aresb_main.py --evaluate --pretrained ./train/${batchsize}/pretrained/${nn}.360epochs.pth.tar --batch-size ${batchsize} --arch ${nn} --outputfile ./output/${batchsize}/${nn}360epochs.out --weight-decay ${weightdecay} --lr ${lr}
done


for DATASET in dtd eurosat ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
do
    bash scripts/mmrlpp/few_shot.sh $DATASET
done

for DATASET in dtd eurosat ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
do
    bash scripts/mmrl/few_shot.sh $DATASET
done
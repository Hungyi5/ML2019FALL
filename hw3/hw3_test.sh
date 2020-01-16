wget https://www.dropbox.com/s/cd1ltsoxd9kg9lx/model_wide_resnet50_2_ep49_SGD_lr1-3_m9.pkl?dl=1 -O ./model/wide_resnet50_2_ep49_SGD_lr1-3_m9.pkl
python3 model.py $1 $2 --test

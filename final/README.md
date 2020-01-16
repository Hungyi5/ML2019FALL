# How to run
```
cd models/best/
wget https://www.dropbox.com/s/6rsf08vkzii9vrq/model.pkl.13?dl=1 -O model.pkl.13
python3.7 predict.py ../models/best/model.pkl.13
```

# if you want training model
1. Prepare the dataset and pre-trained embeddings (FastText is used here) in `./data`:

```
./data/train.json
./data/valid.json
./data/test.json
./data/crawl-300d-2M.vec
```
crawl-300d-2M.vec is download at https://fasttext.cc/docs/en/english-vectors.html

2. Preprocess the data
```
cd src/
python3.7 make_dataset.py ../data/
```

3. To train model as follow:
```
python train.py ../models/best/
```

4. To predict, run
```
python3.7 predict.py ../models/best/model.pkl.13
```

5. To plot, run
```
python3.7 plot.py ../models/best/model.pkl.13
```


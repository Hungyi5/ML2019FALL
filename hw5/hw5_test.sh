wget https://www.dropbox.com/s/5oe9m4h7j1j63qo/gru.pkl?dl=1 -O gru.pkl
wget https://www.dropbox.com/s/vb9pfto19uuqo42/word2vec_seg_iter20_rand2266.model?dl=1 -O word2vec_seg_iter20_rand2266.model
python3 model.py $1 $2 --test
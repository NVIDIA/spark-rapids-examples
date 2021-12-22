nvidia-docker run -d -it \
-v $PWD/data:/data \
-v $PWD/models:/models \
-p 8080:8080 \
-p 4040:4040 \
-p 8899:8888 \
spark-dl:test \
bash
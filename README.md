to create docker image on local host type:
    sudo docker run -it --rm -p 8501:8501 -v "`pwd`/models/cnn:/models/cnn" -e MODEL_NAME=cnn tensorflow/serving 
    
localhost url : 
    'http://localhost:8501/v1/models/cnn:predict'

To run client lunch client.py script:
 1. create virtual env with make_env.sh bash script 
 2. run client.py script with proper command line arguments (set test dataset file path)
 
    python client.py <test.txt path>




sudo docker build --platform linux/amd64,linux/arm64 -t moseiik_tests .
sudo docker run --platform linux/amd64 moseiik_tests 
sudo docker run --platform linux/arm64 moseiik_tests

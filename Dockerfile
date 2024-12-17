#Using rust image, might change late for smaller image
FROM rust:latest
WORKDIR /app

#Download the tiles' zip
ADD https://nasext-vaader.insa-rennes.fr/ietr-vaader/moseiik_test_images.zip /app

#Installing necessary things to unzip, unzipping then removin the zip
RUN apt-get update && apt-get install unzip && unzip /app/moseiik_test_images.zip -d /app/moseiik_test_images && rm /app/moseiik_test_images.zip

#Copy all the files in the folder except what's precised in .dockerignore
COPY . .

#Running the tests
ENTRYPOINT ["cargo", "test", "--release"]

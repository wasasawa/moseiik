# Start with the official Alpine Linux image
FROM alpine:latest

# Set environment variables for Rust
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

# Install dependencies, Rust, and unzip
RUN apk add --no-cache \
        build-base \
	bash \
        curl \
        openssl-dev \
        unzip \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && chmod -R a+w $RUSTUP_HOME $CARGO_HOME

WORKDIR /app

#Download the tiles' zip
ADD https://nasext-vaader.insa-rennes.fr/ietr-vaader/moseiik_test_images.zip /app



# Unzipping then removin the zip
RUN  unzip /app/moseiik_test_images.zip -d /app/moseiik_test_images && rm /app/moseiik_test_images.zip

#Copy all the files in the folder except what's precised in .dockerignore
COPY . .

#Running the tests
ENTRYPOINT ["cargo", "test", "--release"]

FROM rust:latest
WORKDIR /app

COPY . .

RUN ls

CMD ["ls"]
ENTRYPOINT ["cargo", "test", "--release"]

FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt 
EXPOSE 8085
CMD ["waitress-serve", "--listen=*:8085", "app:app"]

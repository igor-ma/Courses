FROM python:3.7-slim

ARG BAU
ARG BAP

ENV BASIC_AUTH_USERNAME=$BAU
ENV BASIC_AUTH_PASSWORD=$BAP

COPY ./requirements.txt /usr/requirements.txt
WORKDIR /usr
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./src /usr/src
COPY ./models /usr/models

ENTRYPOINT [ "python3" ]
CMD [ "src/app/main.py" ]
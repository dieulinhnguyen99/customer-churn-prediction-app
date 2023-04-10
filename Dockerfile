FROM python:3.10.0-slim
COPY requirements.txt /tmp/requirements.txt
COPY *.png /usr/bin/local/scripts/
COPY *.csv /usr/bin/local/scripts/
COPY *.py /usr/bin/local/scripts/
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
RUN chmod +x /usr/bin/local/scripts/
WORKDIR /usr/bin/local/scripts/
ENV PATH="/usr/bin/local:${PATH}"
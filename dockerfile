FROM condaforge/mambaforge:4.9.2-5
WORKDIR /home/
COPY . .
RUN apt-get --allow-releaseinfo-change update

COPY . .

RUN \
    pip install -U pi && \
    . /root/.bashrc && \
    conda init bash && \
    conda env create -f environment.yml && \
    conda activate covid_project && \
    pip install --editable .
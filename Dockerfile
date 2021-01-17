FROM tensorflow/tensorflow:latest-gpu

WORKDIR /ml_project/

RUN apt update && apt install -y wget && \
    wget --progress=bar:force:noscroll -nc https://competitions.codalab.org/my/datasets/download/4aaa1591-b5b2-42ce-a6f5-a2031a2439a1 -O modified_public.zip && \
    unzip -o modified_public.zip && \
    mv modified_public/* data/ && \
    rm modified_public.zip && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install \
    numpy \
    keras \
    jupyterlab  \
    opencv-python \
    sklearn \
    matplotlib \
    pandas

RUN git clone https://github.com/mhd-medfa/NoisyStudent-Based-Object-Recognition.git

CMD jupyter-lab NoisyStudent-Based-Object-Recognition/ --allow-root --ip 0.0.0.0 --port 8080
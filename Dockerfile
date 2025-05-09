FROM quay.io/jupyter/scipy-notebook:python-3.11

USER root
WORKDIR /tmp

RUN apt-get update && \
 apt-get install -yq --no-install-recommends cmake && \
 apt-get clean && \
 rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
  tar -xzf ta-lib-0.6.4-src.tar.gz && \
  cd ta-lib-0.6.4/ && \
  ./configure --prefix=/usr && \
  make && \
  make install
RUN rm -R ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

USER ${NB_UID}

RUN pip install uv

ADD ./vectorbtpro ./vectorbtpro
ADD pyproject.toml ./
ADD LICENSE ./
ADD README.md ./
RUN uv pip install --system --no-cache-dir ".[all]"
RUN uv pip install --system --no-cache-dir --no-deps 'universal-portfolios'

# https://github.com/plotly/dash/issues/2580
RUN jupyter lab build --minimize=False

WORKDIR "$HOME/work"

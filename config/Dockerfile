ARG base_image
FROM $base_image

LABEL maintainer="lucasolivdavid@gmail.com"

ADD actions/install.sh /tmp
RUN /tmp/install.sh

ADD requirements.txt /tmp
RUN pip install -qr /tmp/requirements.txt

ADD config/notebook.json /root/.jupyter/nbconfig/notebook.json

ARG jupyter_theme
RUN jt -t ${jupyter_theme}

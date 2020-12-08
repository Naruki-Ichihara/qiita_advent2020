FROM continuumio/anaconda3:2019.03
RUN pip install --upgrade pip 
WORKDIR /workdir
EXPOSE 8888
ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
CMD ["--notebook-dir=/workdir"]